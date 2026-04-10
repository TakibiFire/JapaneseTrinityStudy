"""
data/dynamic_rebalance_summary.csv を読み込み、
(spend_ratio, target_years, strategy) ごとの生存確率をピボットテーブル形式で表示する。

以前のバージョン (archived/analyze_dynamic_rebalance_main.py) と異なり、
新しい src/dynamic_rebalance_comp_main.py が直接 summary.csv を出力するため、
このスクリプトは主に表示と書式の整形を担当します。
"""

import os
from typing import List, Tuple, cast

import altair as alt
import pandas as pd


def main():
  # 新しいシミュレーションスクリプトが出力する CSV
  csv_path = "data/dynamic_rebalance_summary.csv"

  if not os.path.exists(csv_path):
    print(f"Error: {csv_path} が見つかりません。先に src/dynamic_rebalance_comp_main.py を実行してください。")
    return

  # CSVを読み込む
  df = pd.read_csv(csv_path)

  # 指定された順序で戦略を定義
  strategy_order = [
      "固定最適比率",
      "ダイナミック最適比率",
      "110-年齢 (30歳開始)",
      "110-年齢 (40歳開始)",
      "110-年齢 (50歳開始)",
      "110-年齢 (60歳開始)",
  ]
  # 存在する戦略のみに限定
  existing_strategies = [s for s in strategy_order if s in df["strategy"].unique()]
  df = df[df["strategy"].isin(existing_strategies)]

  # ピボットテーブルの作成
  pivot_df = df.pivot(index=["spend_ratio", "target_years"],
                      columns="strategy",
                      values="survival_probability")
  pivot_df = pivot_df[existing_strategies]

  # 表示用にインデックスをソート (支出率、年数の順)
  pivot_df = pivot_df.sort_index(level=["spend_ratio", "target_years"])

  print("\n--- ダイナミックリバランス戦略の比較 (生存確率) ---")
  print(pivot_df)

  # --- Markdown 形式で出力 ---
  md_lines = []
  
  # ヘッダー
  headers = ["支出率", "目標年数"] + list(pivot_df.columns)
  md_lines.append("| " + " | ".join(headers) + " |")
  md_lines.append("| " + " | ".join([":---"] * 2 + [":---:"] * (len(headers) - 2)) + " |")
  
  for index_val, row in pivot_df.iterrows():
    s_rate, t_years = cast(Tuple[float, float], index_val)
    # 最大値を見つける (許容誤差を含める)
    max_val = row.max()
    is_max = row >= (max_val - 1e-6)
    
    formatted_row = [f"{s_rate:.2%}", f"{int(t_years)}年"]
    for i, val in enumerate(row):
      cell = f"{val:.2%}"
      if is_max.iloc[i]:
        cell = f"**{cell}**"
      formatted_row.append(cell)
    
    md_lines.append("| " + " | ".join(formatted_row) + " |")

  md_output = "\n".join(md_lines)

  # docs/data/dynamic_rebalance/summary.md に保存
  md_dir = "docs/data/dynamic_rebalance"
  os.makedirs(md_dir, exist_ok=True)
  md_path = os.path.join(md_dir, "summary.md")
  with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_output)
  print(f"✅ Markdown テーブルを {md_path} に保存しました。")

  # --- 考察用のサマリー統計 ---
  print("\n--- 戦略ごとのサマリー統計 ---")
  summary_stats = []
  for strategy in existing_strategies:
    win_count = 0
    total_diff = 0.0
    count = 0
    for index_val, row in pivot_df.iterrows():
      # ダイナミックが「固定最適」よりどれくらい勝っているか
      if strategy == "ダイナミック最適比率":
        total_diff += (row["ダイナミック最適比率"] - row["固定最適比率"])
        if row["ダイナミック最適比率"] > row["固定最適比率"] + 1e-6:
          win_count += 1
        count += 1
    
    avg_survival = pivot_df[strategy].mean()
    stat = {"戦略": strategy, "平均生存確率": f"{avg_survival:.2%}"}
    if strategy == "ダイナミック最適比率" and count > 0:
      stat["対 固定最適 勝ち越し数"] = f"{win_count}/{count}"
      stat["対 固定最適 平均改善幅"] = f"{total_diff/count:+.2%}"
    summary_stats.append(stat)
  
  print(pd.DataFrame(summary_stats))

  # --- 可視化 (Altair) ---
  img_dir = "docs/imgs/dynamic_rebalance"
  os.makedirs(img_dir, exist_ok=True)

  # 1. 2x2グリッドグラフ (15x, 20x, 25x, 30x)
  grid_targets = [
      (0.0666666, "15x (6.67%)"),
      (0.05, "20x (5.00%)"),
      (0.04, "25x (4.00%)"),
      (0.0333333, "30x (3.33%)"),
  ]
  
  charts = []
  for s_rate, label in grid_targets:
    # 最も近い支出率を探す
    actual_s = min(df["spend_ratio"].unique(), key=lambda x: abs(x - s_rate))
    
    chart = alt.Chart(df[df["spend_ratio"] == actual_s]).mark_line(point=True).encode(
        x=alt.X("target_years:Q", title="目標年数", axis=alt.Axis(values=list(range(0, 51, 10)))),
        y=alt.Y("survival_probability:Q", title="生存確率", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("strategy:N", title="戦略", sort=existing_strategies),
        tooltip=["strategy", "target_years", "survival_probability"]
    ).properties(
        title=f"支出率 {label} の比較",
        width=300,
        height=250
    )
    charts.append(chart)
  
  grid_chart = alt.vconcat(
      alt.hconcat(charts[0], charts[1]),
      alt.hconcat(charts[2], charts[3])
  ).resolve_scale(color='shared')

  grid_path = os.path.join(img_dir, "strategy_comparison_grid.svg")
  grid_chart.save(grid_path)
  print(f"✅ 比較グリッドグラフを {grid_path} に保存しました。")


if __name__ == "__main__":
  main()


if __name__ == "__main__":
  main()
