"""
ボラティリティ比較のシミュレーションを実行し、結果のサマリーを出力するスクリプト。

オルカンの期待リターンは固定とし、ボラティリティ（シグマ）のみを変化させた場合の
複数のセットアップを比較します。
"""

import os
import sys

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, MonthlyLogNormal,
                                     generate_monthly_asset_prices)
from src.lib.visualize import create_styled_summary, visualize_and_save


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

  # 結果の可視化と保存
  visualize_and_save(
      results,
      html_file="temp/volatility_comp_result.html",
      distribution_image_file="docs/imgs/volatility_comp_result.svg",
      survival_image_file=None,  # このスクリプトでは生存確率グラフは不要の場合
      title="ボラティリティ比較のシミュレーション結果",
      distribution_title="50年後の資産の分布 (ボラティリティ比較)",
      summary_title="最終評価額サマリー (1,000回試行)",
      bankruptcy_years=[])

  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[])
  # Markdown形式で表示
  print(
      formatted_df.to_markdown(colalign=("left",) +
                               ("right",) * len(formatted_df.columns)))

  # CSVとして保存
  csv_dir = "docs/data"
  os.makedirs(csv_dir, exist_ok=True)
  csv_path = os.path.join(csv_dir, "volatility_result.csv")
  formatted_df.to_csv(csv_path)
  print(f"✅ CSVデータを {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
