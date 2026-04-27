"""
このスクリプトは、異なるDynamic Spendingの上限（upper_limit）と下限（lower_limit）、
およびダイナミックリバランスの有無を組み合わせたグリッドシミュレーションを実行します。

設定詳細:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 信託報酬: 0.05775%
- 無リスク資産: 利回り 4%
- インフレ率: AR(12)モデルによる動的変動 (Japan_CPI)
- 税率: 20.315%
- 試行回数: 5000回
- シミュレーション期間: 50年
- リバランス: 1年ごと (12ヶ月)
- シード値: 42
- 初期出費額: 270万円 (2.7%) または 400万円 (4%)

出力フォーマット:
data/dynamic_rebalance/{exp_name}.csv に以下のカラムを持つCSVを出力します。
- upper_limit: 上限 (例: 0.05)
- lower_limit: 下限 (例: -0.015)
- is_dynamic_rebalance: ダイナミックリバランス有効か (1: 有効, 0: 無効)
- 1, 2, ..., 50: 各経過年数における生存確率 (0.0〜1.0) または 実質支出額
"""

import argparse
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core import (DynamicSpending, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)
from src.lib.dp_predictor import DPOptimalStrategyPredictor
from src.lib.dynamic_rebalance import calculate_optimal_strategy
from src.lib.simulation_defaults import get_cpi_ar12_config


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="Dynamic Spendingの各パラメータでの生存確率をグリッドシミュレーションする")
  parser.add_argument(
      "--exp_name",
      type=str,
      default="4p",
      help="実験名 (4p, 2.7p, 2.7p_dp, 1p_1.5p_spend)。カンマ区切りで複数指定可能")
  args = parser.parse_args()

  exp_names = [name.strip() for name in args.exp_name.split(",")]

  for exp_name in exp_names:
    run_experiment(exp_name)


def run_experiment(exp_name: str):
  # 実験設定
  n_sim = 5000
  years = 50
  start_age = 40
  seed = 42
  initial_money = 10000
  tax_rate = 0.20315
  fee_acwi = 0.0005775
  zero_risk_yield = 0.04
  models_path = "data/optimal_strategy_v2_models.json"

  # 実験パラメータの決定
  if exp_name == "4p":
    target_ratio = 0.04
    initial_spend = 400.0
    dynamic_rebalance_options = [0]
    upper_limits = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    lower_limits = [-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, -0.0]
    is_spend_dump = False
  elif exp_name == "2.7p":
    target_ratio = 0.027
    initial_spend = 270.0
    dynamic_rebalance_options = [0]
    upper_limits = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    lower_limits = [-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, -0.0]
    is_spend_dump = False
  elif exp_name == "2.7p_dp":
    target_ratio = 0.027
    initial_spend = 270.0
    dynamic_rebalance_options = [1]
    upper_limits = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    lower_limits = [-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, -0.0]
    is_spend_dump = False
  elif exp_name == "1p_1.5p_spend":
    target_ratio = 0.027
    initial_spend = 270.0
    dynamic_rebalance_options = [1]
    upper_limits = [0.01]
    lower_limits = [-0.015]
    is_spend_dump = True
  else:
    print(f"Skipping unknown exp_name: {exp_name}")
    return

  # 共通アセット名
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  zr_name = "無リスク資産(4%)"
  cpi_name = "Japan_CPI"

  # 1. 価格推移の生成
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name,
                 dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)),
      Asset(name=acwi_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=fee_acwi,
            forex=fx_name),
      get_cpi_ar12_config(name=cpi_name)
  ]

  print(f"月次価格の推移を生成中... (試行回数: {n_sim}, 期間: {years}年)")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=years * 12,
                                                       seed=seed)

  zr_asset = ZeroRiskAsset(name=zr_name, yield_rate=zero_risk_yield)

  # DP予測器の準備
  dp_predictor = None
  if 1 in dynamic_rebalance_options:
    dp_predictor = DPOptimalStrategyPredictor(models_path)

  # ダイナミック最適比率用のコールバック
  def dynamic_optimal_fn(
      net_value: np.ndarray, annual_spend: np.ndarray, remaining_years: float,
      post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    if dp_predictor is not None:
      # 年齢の算出
      elapsed_years = int(round(years - (remaining_years - 0.25)))
      predict_age = start_age + elapsed_years
      # DP予測器を使用
      ratio_array = dp_predictor.get_a_opt_with_winning_threshold(
          predict_age, post_tax_net, annual_spend)
    else:
      # 従来のV1ヒューリスティック (もし使用する場合)
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

  # 1. キャッシュフロールールの定義 (基本支出のベース設定)
  spend_config = BaseSpendConfig(name="生活費",
                                 amount=initial_spend,
                                 cpi_name=cpi_name)
  monthly_cashflows = generate_cashflows([spend_config], monthly_asset_prices,
                                         n_sim, years * 12)

  print(f"各戦略のシミュレーションを実行中... (実験: {exp_name})")
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
          initial_ratio = {
              acwi_name: float(initial_r),
              zr_asset: 1.0 - float(initial_r)
          }

        # DynamicSpendingの仕様上、インフレ調整は名目前年支出額に対して行われるため、
        # BaseSpendConfigにcpi_nameを渡すことで、Vanguardルールのインフレ調整が有効になる。
        ds_handler = DynamicSpending(initial_annual_spend=initial_spend,
                                     target_ratio=target_ratio,
                                     upper_limit=upper,
                                     lower_limit=lower)
        cashflow_rules = [
            CashflowRule(source_name=spend_config.name,
                         cashflow_type=CashflowType.REGULAR,
                         dynamic_handler=ds_handler)
        ]

        strategy = Strategy(name=f"DR{is_dyn}/U{upper:.1%}/L{lower:.1%}",
                            initial_money=initial_money,
                            initial_loan=0,
                            yearly_loan_interest=0,
                            initial_asset_ratio=initial_ratio,
                            cashflow_rules=cashflow_rules,
                            tax_rate=tax_rate,
                            selling_priority=selling_priority,
                            rebalance_interval=12,
                            dynamic_rebalance_fn=dynamic_fn,
                            record_annual_spend=is_spend_dump)

        res = simulate_strategy(strategy,
                                monthly_asset_prices,
                                monthly_cashflows=monthly_cashflows)

        if is_spend_dump:
          # 支出額のダンプ
          nominal_spends = res.annual_spends
          assert nominal_spends is not None

          # 実質支出額の計算
          real_spends = np.zeros_like(nominal_spends)
          for y in range(years):
            cpi_at_year_start = monthly_asset_prices[cpi_name][:, y * 12]
            real_spends[:, y] = nominal_spends[:, y] / cpi_at_year_start

          # CSV用に整形 (pathごとに1行)
          for path_idx in range(n_sim):
            spend_row: dict[str, Any] = {
                "upper_limit": upper,
                "lower_limit": lower,
                "is_dynamic_rebalance": is_dyn,
                "path_id": path_idx,
                "value_type": "real_spend"
            }
            for y in range(years):
              spend_row[str(y + 1)] = real_spends[path_idx, y]
            results.append(spend_row)
        else:
          # 生存確率の計算
          survival_row: dict[str, Any] = {
              "upper_limit": upper,
              "lower_limit": lower,
              "is_dynamic_rebalance": is_dyn,
              "value_type": "survival"
          }
          for year in range(1, years + 1):
            bankrupt_count = (res.sustained_months < year * 12).sum()
            survival_rate = 1.0 - (bankrupt_count / n_sim)
            survival_row[str(year)] = survival_rate
          results.append(survival_row)

  df = pd.DataFrame(results)
  data_dir = "data/dynamic_rebalance"
  os.makedirs(data_dir, exist_ok=True)
  csv_path = os.path.join(data_dir, f"{exp_name}.csv")
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"✅ {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
