"""
ボラティリティ比較のシミュレーションを実行し、結果のサマリーを出力するスクリプト。

オルカンの期待リターンは固定とし、ボラティリティ（シグマ）のみを変化させた場合の
複数のセットアップを比較します。
"""

import sys

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, MonthlyLogNormal,
                                     generate_monthly_asset_prices)
from visualize import create_styled_summary


def main():
  sigmas = [0, 11, 13, 15, 17]

  # 新エンジンはシミュレーションの月数とパス数をシミュレーション時に指定するため、
  # 変数として定義しておきます。
  N_SIM = 1000
  YEARS = 50
  N_MONTHS = YEARS * 12
  SEED = 42

  # 1. 資産の定義
  # 新エンジンでは MonthlyLogNormal を使用し、年率パラメータを渡します。
  assets = [
      Asset(name=f"オルカン v{v}%",
            dist=MonthlyLogNormal(mu=0.07, sigma=v / 100.0),
            trust_fee=0,
            leverage=1) for v in sigmas
  ]

  # 2. 戦略(Plan)の定義
  strategies = [
      Strategy(name=f"オルカン, ボラ={v}%",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={f"オルカン v{v}%": 1.0},
               annual_cost=0.0,
               inflation_rate=None,
               tax_rate=0.0,
               selling_priority=[f"オルカン v{v}%"]) for v in sigmas
  ]

  # 3. シミュレーションの実行
  print("新エンジン: 月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=N_SIM,
                                                       n_months=N_MONTHS,
                                                       seed=SEED)

  results = {}
  print("新エンジン: 各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # 4. サマリーの出力
  print("\n--- シミュレーション結果 ---")
  formatted_df, raw_df = create_styled_summary(
      results,  # type: ignore
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[])

  # Markdown形式で表示
  print(
      formatted_df.to_markdown(colalign=("left",) +
                               ("right",) * len(formatted_df.columns)))


if __name__ == "__main__":
  main()
