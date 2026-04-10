"""
data/dynamic_rebalance_summary.csv を読み込み、
(spend_ratio, target_years, strategy) ごとの生存確率をピボットテーブル形式で表示する。

以前のバージョン (archived/analyze_dynamic_rebalance_main.py) と異なり、
新しい src/dynamic_rebalance_comp_main.py が直接 summary.csv を出力するため、
このスクリプトは主に表示と書式の整形を担当します。
"""

import os

import pandas as pd


def main():
  # 新しいシミュレーションスクリプトが出力する CSV
  csv_path = "data/dynamic_rebalance_summary.csv"

  if not os.path.exists(csv_path):
    print(f"Error: {csv_path} が見つかりません。先に src/dynamic_rebalance_comp_main.py を実行してください。")
    return

  # CSVを読み込む
  df = pd.read_csv(csv_path)

  # ピボットテーブルの作成
  # インデックス: spend_ratio, target_years
  # カラム: strategy
  # 値: survival_probability
  pivot_df = df.pivot(index=["spend_ratio", "target_years"],
                      columns="strategy",
                      values="survival_probability")

  # 指定された順序でカラムを並べ替え
  column_order = [
      "固定最適比率",
      "ダイナミック最適比率",
      "110-年齢 (30歳開始)",
      "110-年齢 (40歳開始)",
      "110-年齢 (50歳開始)",
      "110-年齢 (60歳開始)",
  ]
  # 存在するカラムのみに限定
  column_order = [c for c in column_order if c in pivot_df.columns]
  pivot_df = pivot_df[column_order]

  # 表示設定
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)

  print("\n--- ダイナミックリバランス戦略の比較 (生存確率) ---")
  print(pivot_df)

  # 表示用にインデックスをソート (支出率、年数の順)
  pivot_df = pivot_df.sort_index(level=["spend_ratio", "target_years"])

  # Markdown 形式で出力 (最も確率が高いものを太字にする)
  md_lines = []
  
  # ヘッダー
  headers = ["支出率", "目標年数"] + list(pivot_df.columns)
  md_lines.append("| " + " | ".join(headers) + " |")
  md_lines.append("| " + " | ".join([":---"] * 2 + [":---:"] * (len(headers) - 2)) + " |")
  
  for (s_rate, t_years), row in pivot_df.iterrows():
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
  
  print(f"\n✅ Markdown テーブルを {md_path} に保存しました。")


if __name__ == "__main__":
  main()
