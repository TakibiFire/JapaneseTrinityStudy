"""
日本版トリニティ・スタディの最適化スクリプト。

`optimization.py` を用いて、複数の最適化ターゲット（破産確率の最小化など）に対して
最適なパラメータ（資産配分、初期借入額、リバランス間隔）を算出し、
その結果のサマリーをターミナルに出力する。
"""

import time
from typing import Dict

import numpy as np
import pandas as pd

from core import (Asset, Strategy, create_styled_summary,
                  generate_monthly_asset_prices, simulate_strategy)
from optimization import OptimizationTarget, create_strategy, optimize_strategy


def main() -> None:
  """
  最適化モジュールの使用例となるメイン処理。
  
  複数の最適化ターゲット（破産確率最小化、50パーセンタイル最大化など）に対して
  scipy.optimize.brute を用いて探索を実行し、得られた最適パラメータとその結果
  （計算時間、最適化後のシミュレーションサマリー）をターミナルに表示する。
  """
  print("資産の価格推移をシミュレーションしています...")

  # 資産の定義
  assets = [
      Asset(name="オルカン", yearly_cost=0.05775 / 100, leverage=1),
      Asset(name="レバカン", yearly_cost=0.422 / 100, leverage=2)
  ]

  # 月次価格の生成
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  targets = [
      OptimizationTarget.MINIMIZE_RUIN_PROBABILITY,
      # OptimizationTarget.MAXIMIZE_10_PERCENTILE,
      OptimizationTarget.MAXIMIZE_50_PERCENTILE,
  ]

  for target in targets:
    print(f"\n--- 最適化ターゲット: {target.value} ---")
    start_time = time.time()

    # 最適化の実行
    best_i, best_j, best_k, best_r, best_score = optimize_strategy(
        monthly_asset_prices, target)

    elapsed_time = time.time() - start_time

    print(f"計算時間: {elapsed_time:.2f} 秒")
    print(f"最適化結果:")
    print(f"  オルカン初期投資割合 (i): {best_i:.1f}")
    print(f"  レバカン初期投資割合 (j): {best_j:.1f}")
    print(f"  初期借入額の係数 (k)    : {best_k}")
    print(f"  リバランス間隔 (r)      : {best_r} ヶ月")

    if target == OptimizationTarget.MINIMIZE_RUIN_PROBABILITY:
      print(f"  最小破産確率           : {best_score:.2f}%")
    else:
      print(f"  最大化されたスコア       : {best_score:.2f} 万円")

    # 最適化されたパラメータを用いたシミュレーション
    optimized_strategy = create_strategy(best_i,
                                         best_j,
                                         best_k,
                                         best_r,
                                         name=f"Opt ({target.value})")

    net_values = simulate_strategy(optimized_strategy, monthly_asset_prices)
    df_results = pd.DataFrame({optimized_strategy.name: net_values})

    styled_summary = create_styled_summary(df_results)

    print("\n  --- シミュレーション サマリー ---")

    # ターミナルで見やすいように転置して文字列化
    summary_data = styled_summary.data
    formatted_series = pd.Series(index=summary_data.columns, dtype=str)

    for col in summary_data.columns:
      val = summary_data.loc[optimized_strategy.name, col]
      if "確率" in col:
        formatted_series[col] = f"{val:.1f}%"
      else:
        formatted_series[col] = f"約 {val / 10000:.1f}億円"

    formatted_series.name = optimized_strategy.name
    print(formatted_series.to_string())


if __name__ == "__main__":
  main()
