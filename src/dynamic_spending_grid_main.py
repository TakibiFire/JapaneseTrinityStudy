"""
このスクリプトは、異なるDynamic Spendingの上限（upper_limit）と下限（lower_limit）、
およびダイナミックリバランスの有無を組み合わせたグリッドシミュレーションを実行します。

設定詳細:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 信託報酬: 0.05775%
- 無リスク資産: 利回り 4% (ダイナミックリバランス有効時に使用)
- インフレ率: 0.0 (DynamicSpendingの仕様上考慮しないため)
- 税率: 20.315%
- 試行回数: 5000回
- シミュレーション期間: 50年
- リバランス: 1年ごと (12ヶ月)
- シード値: 42
- 初期出費額: 400万円 (4%)

出力フォーマット:
data/dynamic_spending_grid_comp.csv に以下のカラムを持つCSVを出力します。
- upper_limit: 上限 (例: 0.05)
- lower_limit: 下限 (例: -0.015)
- is_dynamic_rebalance: ダイナミックリバランス有効か (1: 有効, 0: 無効)
- 1, 2, ..., 50: 各経過年数における生存確率 (0.0〜1.0)
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core import (DynamicSpending, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.dynamic_rebalance import calculate_optimal_strategy


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  seed = 42
  initial_money = 10000
  tax_rate = 0.20315
  fee_acwi = 0.0005775
  zero_risk_yield = 0.04

  # 共通アセット名
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  zr_name = "無リスク資産(4%)"

  # 1. 価格推移の生成
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name,
                 dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)),
      Asset(name=acwi_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=fee_acwi,
            forex=fx_name)
  ]

  print(f"月次価格の推移を生成中... (試行回数: {n_sim}, 期間: {years}年)")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=years * 12,
                                                       seed=seed)

  zr_asset = ZeroRiskAsset(name=zr_name, yield_rate=zero_risk_yield)

  # 変数: 上限、下限、ダイナミックリバランスの有無
  upper_limits = [0.02, 0.03, 0.04, 0.05, 0.06]
  lower_limits = [
      0.02, 0.015, 0.01, 0.005, 0.0, -0.005, -0.01, -0.015, -0.02
  ]
  dynamic_rebalance_options = [0, 1]

  # ダイナミック最適比率用のコールバック
  def dynamic_optimal_fn(net_value: np.ndarray, annual_spend: np.ndarray,
                         remaining_years: float) -> Dict[str, Union[float, np.ndarray]]:
    safe_net_value = np.maximum(net_value, 1e-10)
    s_rate = annual_spend / safe_net_value
    # calculate_optimal_strategyはインフレ率0.0177を前提にチューニングされているため、
    # 常に0.0177を渡す必要がある。
    ratio_array = calculate_optimal_strategy(s_rate,
                                             remaining_years,
                                             base_yield=zero_risk_yield,
                                             tax_rate=tax_rate,
                                             inflation_rate=0.0177)
    return {acwi_name: ratio_array, zr_name: 1.0 - ratio_array}

  results: list[dict[str, Any]] = []

  total_combinations = len(upper_limits) * len(lower_limits) * len(
      dynamic_rebalance_options)
  count = 0

  print("各戦略のシミュレーションを実行中...")
  for is_dyn in dynamic_rebalance_options:
    for upper in upper_limits:
      for lower in lower_limits:
        count += 1
        print(
            f"Processing dynamic_reb: {is_dyn}, upper: {upper:.1%}, lower: {lower:.1%} ({count}/{total_combinations})"
        )

        initial_ratio: Dict[Union[str, ZeroRiskAsset], float] = {acwi_name: 1.0}
        dynamic_fn = None
        selling_priority = [acwi_name]

        if is_dyn == 1:
          dynamic_fn = dynamic_optimal_fn
          selling_priority = [zr_name, acwi_name]
          # 初期比率は100%オルカン (期待値)
          initial_r = 1.0
          initial_ratio = {acwi_name: float(initial_r), zr_asset: 1.0 - float(initial_r)}

        # DynamicSpendingの仕様上、インフレ調整は名目前年支出額に対して行われるため、
        # Strategy内部の共通インフレ調整機能は無効(0.0)にする必要がある。
        strategy = Strategy(
            name=f"DR{is_dyn}/U{upper:.1%}/L{lower:.1%}",
            initial_money=initial_money,
            initial_loan=0,
            yearly_loan_interest=0,
            initial_asset_ratio=initial_ratio,
            annual_cost=DynamicSpending(target_ratio=0.04,
                                        upper_limit=upper,
                                        lower_limit=lower),
            inflation_rate=0.0,
            tax_rate=tax_rate,
            selling_priority=selling_priority,
            rebalance_interval=12,
            dynamic_rebalance_fn=dynamic_fn)

        res = simulate_strategy(strategy, monthly_asset_prices)

        # 生存確率の計算
        row: dict[str, Any] = {
            "upper_limit": upper,
            "lower_limit": lower,
            "is_dynamic_rebalance": is_dyn,
        }
        for year in range(1, years + 1):
          bankrupt_count = (res.sustained_months < year * 12).sum()
          survival_rate = 1.0 - (bankrupt_count / n_sim)
          row[str(year)] = survival_rate

        results.append(row)

  df = pd.DataFrame(results)
  os.makedirs("data", exist_ok=True)
  csv_path = "data/dynamic_spending_grid_comp.csv"
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"✅ {csv_path} に保存しました。")
  print(df.head())


if __name__ == "__main__":
  main()
