"""
DP Experimental Approaches の結果を分析・可視化するスクリプト。
Altair を用いて、8つの全組み合わせを一つのグラフで比較します。
"""

import os

import altair as alt
import pandas as pd


def main():
  csv_path = "data/experiment_dp_grid/eval.csv"
  img_dir = "docs/imgs/all_60yr"
  os.makedirs(img_dir, exist_ok=True)

  if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    return

  df = pd.read_csv(csv_path)

  # 表示用にラベルを加工
  df['Combination'] = df['mode'] + " (miny=" + df['min_y'].astype(
      str) + ") / " + df['win_type']

  # 各組み合わせのポイントを計算
  combinations = df['Combination'].unique()
  points = {c: 0 for c in combinations}
  rules = df['spending_rule'].unique()

  print(f"Analyzing {len(rules)} rules for Prediction Modes...")
  print(f"{'Rule':>5} | {'Max SR':>7} | Top Strategies (within 0.3%)")
  print("-" * 70)

  for rule in rules:
    df_rule = df[df['spending_rule'] == rule]
    max_sr = df_rule['survival_rate'].max()

    # 閾値 (最高値 - 1%)
    threshold = max_sr - 0.01
    top_tier = df_rule[df_rule['survival_rate'] >= threshold]
    top_tier_names = top_tier['Combination'].tolist()

    print(f"{rule:5.2f} | {max_sr:7.4f} | {', '.join(top_tier_names)}")

    for c in top_tier_names:
      points[c] += 1

  print("\n=== Final Strategy Rankings (Points) ===")
  sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
  for c, p in sorted_points:
    print(f"{c:<25}: {p} points")

  print("\n=== Final Strategy Rankings (Sum of Survival Rates) ===")
  sum_survival = df.groupby('Combination')['survival_rate'].sum().sort_values(
      ascending=False)
  for c, s in sum_survival.items():
    print(f"{c:<25}: {s:.4f}")

  # ポイント順に並べ替えて上位6つを抽出。
  # ただし、baseline (legacy / V1) は比較のために常に含める。
  baseline = "legacy (miny=0) / V1"
  top_6_strategies = [c for c, p in sorted_points[:6]]

  plot_strategies = list(top_6_strategies)
  if baseline in points and baseline not in plot_strategies:
    plot_strategies.append(baseline)
    print(f"Added baseline to plot: {baseline}")

  print(f"\nPlot Strategies: {plot_strategies}")

  # 対象の戦略のみを保持
  df_plot = df[df['Combination'].isin(plot_strategies)].copy()

  # y軸の範囲をデータの最小値・最大値に合わせる
  y_min = df_plot['survival_rate'].min()
  y_max = df_plot['survival_rate'].max()
  # 少しマージンを持たせる
  y_range = [max(0, y_min - 0.05), min(1.0, y_max + 0.05)]

  chart = alt.Chart(df_plot).mark_line(point=True).encode(
      x=alt.X('spending_rule:Q', title='年間支出率 (%)'),
      y=alt.Y('survival_rate:Q',
              title='生存確率 (95歳開始時点)',
              scale=alt.Scale(domain=y_range)),
      color=alt.Color('Combination:N',
                      title='Approach Combinations',
                      legend=alt.Legend(columns=1, symbolLimit=0)),
      tooltip=[
          'spending_rule', 'survival_rate', 'win_type', 'mode', 'min_y'
      ]).properties(
          width=600,
          height=500,
          title='Comparison of DP Experimental Approaches (Top 6 + Baseline)'
      ).interactive()

  output_path = os.path.join(img_dir, "exp_dp_full_comparison_single.svg")
  chart.save(output_path)
  print(f"Saved single comparison plot to {output_path}")


if __name__ == "__main__":
  main()
