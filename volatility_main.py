"""
ボラティリティ比較のシミュレーションを実行し、HTMLレポートを生成するスクリプト。

オルカンの期待リターンは固定とし、ボラティリティ（シグマ）のみを変化させた場合の
複数のセットアップを比較します。
"""

from core import (Asset, Strategy, generate_monthly_asset_prices,
                  simulate_strategy)
from visualize import visualize_and_save


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  sigmas = [0, 11, 13, 15, 17]
  
  assets = [
      Asset(name=f"オルカン v{v}%", yearly_cost=0, mu=0.07, sigma=v / 100.0, leverage=1)
      for v in sigmas
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      Strategy(name=f"オルカン, ボラ={v}%",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={f"オルカン v{v}%": 1.0},
               annual_cost=0,
               annual_cost_inflation=0,
               selling_priority=[f"オルカン v{v}%"])
      for v in sigmas
  ]

  # ---------------------------------------------------------------------------
  # 3. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 4. 可視化と保存
  # ---------------------------------------------------------------------------
  visualize_and_save(
      results=results,
      html_file='temp/volatility_result.html',
      image_file='imgs/volatility_result.svg',
      title='ボラティリティ違いによる50年後の最終評価額',
      summary_title='ボラティリティ比較サマリー（1,000回試行）',
      bankruptcy_years=[]
  )

if __name__ == "__main__":
  main()
