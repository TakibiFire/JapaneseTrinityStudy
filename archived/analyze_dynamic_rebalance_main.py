"""
data/dynamic_rebalance_comp.csv を読み込み、
(spend_ratio, target_years, strategy) ごとに target_years 年目の生存確率を抽出し、
戦略ごとの比較表を作成・表示します。
"""

import pandas as pd


def main():
  csv_path = "data/dynamic_rebalance_comp.csv"

  # BOM付きのCSVを想定して utf-8-sig で読み込む
  df = pd.read_csv(csv_path, encoding="utf-8-sig")

  # (spend_ratio, target_years, strategy) ごとに target_years 年目の生存確率を抽出
  results = []
  for _, row in df.iterrows():
    target_years = int(row["target_years"])
    spend_ratio = row["spend_ratio"]
    strategy = row["strategy"]

    # target_years カラムの値を抽出
    # カラム名は文字列 "10", "20", ...
    survival_rate = row[str(target_years)]

    results.append({
        "spend_ratio": spend_ratio,
        "target_years": target_years,
        "strategy": strategy,
        "survival_rate": survival_rate
    })

  res_df = pd.DataFrame(results)

  # ピボットテーブルの作成
  # インデックス: spend_ratio, target_years
  # カラム: strategy
  # 値: survival_rate
  pivot_df = res_df.pivot(index=["spend_ratio", "target_years"],
                          columns="strategy",
                          values="survival_rate")

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

  print("--- ダイナミックリバランス戦略の比較 (目標年数時点の生存確率) ---")
  print(pivot_df)

  # CSVへ保存
  output_path = "data/dynamic_rebalance_summary.csv"
  pivot_df.to_csv(output_path, encoding="utf-8-sig")
  print(f"\n✅ {output_path} に保存しました。")


if __name__ == "__main__":
  main()
