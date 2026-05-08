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

  # 結果のダンプ
  print("--- Experimental Results ---")
  print(
      df.sort_values(
          ['robust', 'n_sim_train', 'disable_win_threshold',
           'spending_rule']).to_string(index=False))
  print("----------------------------")

  # 表示用にラベルを加工
  df['WinThreshold'] = df['disable_win_threshold'].apply(lambda x: 'Win:OFF'
                                                         if x else 'Win:ON')
  df['Growth'] = df['robust'].apply(lambda x: 'Robust' if x else 'Net')
  df['Train'] = df['n_sim_train'].apply(lambda x: f'N={x}')

  # 8通りの組み合わせを一つのラベルに統合
  df['Combination'] = df['WinThreshold'] + ' / ' + df['Growth'] + ' / ' + df[
      'Train']

  # y軸の範囲をデータの最小値・最大値に合わせる
  y_min = df['survival_rate'].min()
  y_max = df['survival_rate'].max()
  # 少しマージンを持たせる
  y_range = [max(0, y_min - 0.05), min(1.0, y_max + 0.05)]

  chart = alt.Chart(df).mark_line(point=True).encode(
      x=alt.X('spending_rule:Q', title='年間支出率 (%)'),
      y=alt.Y('survival_rate:Q',
              title='生存確率 (95歳開始時点)',
              scale=alt.Scale(domain=y_range)),
      color=alt.Color('Combination:N',
                      title='Approach Combinations',
                      legend=alt.Legend(columns=1, symbolLimit=0)),
      tooltip=[
          'spending_rule', 'survival_rate', 'WinThreshold', 'Growth', 'Train'
      ]).properties(
          width=600,
          height=500,
          title='Comparison of all 8 DP Experimental Approaches').interactive()

  output_path = os.path.join(img_dir, "exp_dp_full_comparison_single.svg")
  chart.save(output_path)
  print(f"Saved single comparison plot to {output_path}")


if __name__ == "__main__":
  main()
