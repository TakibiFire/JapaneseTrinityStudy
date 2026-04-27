"""
このスクリプトは、異なる支出率（spend_ratio）と資産配分（orukan_ratio）の組み合わせに対して、
生存確率のグリッドシミュレーションを実行します。

設定詳細:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 信託報酬: 0.05775%
- 無リスク資産: 利回り 4%
- インフレ率: 年率 1.77% (固定、Japan_CPI_1.77pct)
- 税率: 20.315%
- 試行回数: 5000回
- シミュレーション期間: 60年
- リバランス: 1年ごと (12ヶ月)
- 売却順序: 1.無リスク資産, 2.株式（オルカン）
- シード値: 42

出力フォーマット:
data/withdrawal_rate_grid_comp.csv に以下のカラムを持つCSVを出力します。
- spend_ratio: 年間の支出率 (例: 0.04)
- orukan_ratio: オルカンの配分比率 (例: 0.8)
- 1, 2, ..., 60: 各経過年数における生存確率 (0.0〜1.0)
"""

import os
from typing import Any, Dict, List, Union

import pandas as pd

from src.core import Strategy, ZeroRiskAsset, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 60
  seed = 42
  initial_money = 10000
  tax_rate = 0.20315
  inflation_rate_mu = 0.0177
  fee_acwi = 0.0005775
  zero_risk_yield = 0.04

  # 共通アセット名
  cpi_name = "Japan_CPI_1.77pct"
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  zero_risk_asset_name = "無リスク資産(4%)"

  # 資産モデル設定
  ork_dist = YearlyLogNormalArithmetic(mu=0.07, sigma=0.15)
  fx_dist = YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)

  # 1. 価格推移の生成
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name, dist=fx_dist),
      Asset(name=acwi_name, dist=ork_dist, trust_fee=fee_acwi, forex=fx_name),
      CpiAsset(name=cpi_name,
               dist=YearlyLogNormalArithmetic(mu=inflation_rate_mu, sigma=0.0))
  ]

  print(f"月次価格の推移を生成中... (試行回数: {n_sim}, 期間: {years}年)")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=years * 12,
                                                       seed=seed)

  zero_risk_asset = ZeroRiskAsset(name=zero_risk_asset_name,
                                  yield_rate=zero_risk_yield)

  spending_rates = [
      (0.3333333, "33.3% (x3)"),
      (0.25, "25% (x4)"),
      (0.20, "20% (x5)"),
      (0.15, "15% (x6.7)"),
      (0.10, "10% (x10)"),
      (0.0666666, "6.67% (x15)"),
      (0.05, "5% (x20)"),
      (0.04, "4% (x25)"),
      (0.035714, "3.57% (x28)"),
      (0.033333, "3.333% (x30)"),
      (0.03, "3% (x33.3)"),
      (0.028571, "2.86% (x35)"),
      (0.025, "2.5% (x40)"),
      (0.022222, "2.222% (x45)"),
      (0.02, "2% (x50)"),
  ]
  stock_ratios = [
      1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
      0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0
  ]

  results: list[dict[str, Any]] = []

  total_combinations = len(spending_rates) * len(stock_ratios)
  count = 0

  print("各戦略のシミュレーションを実行中...")
  for spending_rate, spending_label in spending_rates:
    annual_cost = initial_money * spending_rate
    print(
        f"Processing spending rate: {spending_label} ({count}/{total_combinations})"
    )

    # 1. キャッシュフロールールの定義
    spend_config = BaseSpendConfig(name="生活費",
                                   amount=annual_cost,
                                   cpi_name=cpi_name)
    cashflow_rules = [
        CashflowRule(source_name=spend_config.name,
                     cashflow_type=CashflowType.REGULAR)
    ]
    monthly_cashflows = generate_cashflows([spend_config], monthly_asset_prices,
                                           n_sim, years * 12)

    for ratio in stock_ratios:
      zr_ratio = 1.0 - ratio

      # 初期資産配分の設定
      initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float] = {
          acwi_name: ratio
      }
      if zr_ratio > 0:
        initial_asset_ratio[zero_risk_asset] = zr_ratio

      # 売却順序: 無リスク資産を優先
      if zr_ratio > 0:
        selling_priority = [zero_risk_asset_name, acwi_name]
      else:
        selling_priority = [acwi_name]

      strategy = Strategy(
          name=f"支出{spending_label} / オルカン{int(ratio*100)}%",
          initial_money=initial_money,
          initial_loan=0,
          yearly_loan_interest=0,
          initial_asset_ratio=initial_asset_ratio,
          cashflow_rules=cashflow_rules,
          tax_rate=tax_rate,
          selling_priority=selling_priority,
          rebalance_interval=12  # 1年ごとのリバランス
      )

      res = simulate_strategy(strategy,
                              monthly_asset_prices,
                              monthly_cashflows=monthly_cashflows)

      # 生存確率の計算 (生存確率 = 100% - 破産確率)
      row: dict[str, Any] = {
          "spend_ratio": spending_rate,
          "orukan_ratio": ratio,
      }
      for year in range(1, years + 1):
        bankrupt_count = (res.sustained_months < year * 12).sum()
        survival_rate = 1.0 - (bankrupt_count / n_sim)
        row[str(year)] = survival_rate

      results.append(row)
      count += 1

  df = pd.DataFrame(results)
  os.makedirs("data", exist_ok=True)
  csv_path = "data/withdrawal_rate_grid_comp.csv"
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"✅ {csv_path} に保存しました。")
  print(df.head())


if __name__ == "__main__":
  main()
