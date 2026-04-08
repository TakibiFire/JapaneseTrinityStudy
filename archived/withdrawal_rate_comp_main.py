"""
このスクリプトは、異なる支出率（spend_ratio）と資産配分（orukan_ratio）の組み合わせに対して、
生存確率のグリッドシミュレーションを実行します。

主な設定:
- インフレ率 2% 固定
- 無リスク資産（利回り4%, 税引後）を併用
- 1年ごとのリバランス
- 売却順序: 1.無リスク資産, 2.株式（オルカン）

出力フォーマット:
data/withdrawal_rate_comp.csv に以下のカラムを持つCSVを出力します。
- spend_ratio: 年間の支出率 (例: 0.04)
- orukan_ratio: オルカンの配分比率 (例: 0.8)
- 1, 2, ..., 50: 各経過年数における生存確率 (0.0〜1.0)
"""

import os
import re
import sys
from typing import Any, Dict, Union

import pandas as pd

from core import (Asset, Forex, Strategy, ZeroRiskAsset, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)


def main():
  # 為替の定義
  forexes = [Forex(name="USDJPY", mu=0.0, sigma=0.1053)]
  forex_paths = generate_forex_paths(forexes)

  # 資産の定義 (オルカン)
  asset_name = "オルカン"
  assets_def = [
      Asset(name=asset_name,
            trust_fee=0.0005775,
            mu=0.07,
            sigma=0.15,
            leverage=1,
            forex="USDJPY"),
  ]

  print("月次価格推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets_def,
                                                       forex_paths=forex_paths)

  # 無リスク資産の定義 (4%)
  rf = ZeroRiskAsset(name="無リスク資産(4%)", yield_rate=0.04)

  spending_rates = [
      (0.0666666, "5% (x15)"),
      (0.05, "5% (x20)"),
      (0.04, "4% (x25)"),
      (0.035714, "3.57% (x28)"),
      (0.033333, "3.333% (x30)"),
      (0.03, "3% (x33.3)"),
      (0.028571, "2.86% (x35)"),
      (0.025, "2.5% (x40)"),
      (0.022222, "2.222% (x45)"),
      (0.02, "2.222% (x50)"),
  ]
  stock_ratios = [
      1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
      0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0
  ]

  initial_money = 10000

  results: list[dict[str, Any]] = []

  print("各戦略のシミュレーションを実行中...")
  for spending_rate, spending_label in spending_rates:
    annual_cost = initial_money * spending_rate
    for ratio in stock_ratios:
      rf_ratio = 1.0 - ratio

      # initial_asset_ratio setup
      initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float] = {}
      if ratio == 1.0:
        initial_asset_ratio[asset_name] = 1.0
        selling_priority = [asset_name]
      else:
        initial_asset_ratio[asset_name] = ratio
        initial_asset_ratio[rf] = rf_ratio
        selling_priority = [rf.name, asset_name]

      strategy = Strategy(
          name=f"支出{spending_label} / オルカン{int(ratio*100)}%",
          initial_money=initial_money,
          initial_loan=0,
          yearly_loan_interest=0,
          initial_asset_ratio=initial_asset_ratio,
          annual_cost=annual_cost,
          inflation_rate=0.02,
          tax_rate=0.20315,
          selling_priority=selling_priority,
          rebalance_interval=12  # 1年ごとのリバランス
      )

      res = simulate_strategy(strategy, monthly_asset_prices)

      # 生存確率の計算 (生存確率 = 100% - 破産確率)
      row: dict[str, Any] = {
          "spend_ratio": spending_rate,
          "orukan_ratio": ratio,
      }
      for year in range(1, 51):
        bankrupt_count = (res.sustained_months < year * 12).sum()
        survival_rate = 1.0 - (bankrupt_count / len(res.sustained_months))
        row[str(year)] = survival_rate

      results.append(row)

  df = pd.DataFrame(results)
  os.makedirs("data", exist_ok=True)
  csv_path = "data/withdrawal_rate_comp.csv"
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"✅ {csv_path} に保存しました。")
  print(df.head())


if __name__ == "__main__":
  main()
