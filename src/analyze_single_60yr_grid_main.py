"""
data/single_60yr/ の結果を分析・可視化するスクリプト（単身世帯版）。

内容:
1. 最適な組み合わせの分析 (受給開始年齢 × Dynamic Spending)
2. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
"""

import argparse
import json
import os

import pandas as pd

from src.lib.survival_contours import (generate_rule_of_thumb,
                                       generate_smooth_contour_data,
                                       get_contour_anchor_points,
                                       save_contour_charts)
from src.lib.survival_formula_analysis import run_survival_formula_analysis
from src.lib.visualize_all_yr import (calculate_preference_order,
                                      create_heatmap,
                                      create_optimal_pension_heatmap,
                                      create_pension_survival_curve,
                                      prepare_heatmap_labels,
                                      run_optimal_pension_age_analysis)

# 設定
IMG_DIR = "docs/imgs/single_60yr"
DATA_OUT_DIR = "docs/data/single_60yr"
NUM_YEARS = 35
START_AGE = 60


def run_optimal_pension_analysis(df_all: pd.DataFrame, target_year: str):
  """
  最適な年金受給開始年齢を分析する。
  """
  print(f"\n\n{'='*20} 最適な年金受給開始年齢の分析 {'='*20}")

  # 1. グラフ作成 (m=1, r=4% と m=1, r=5%)
  create_pension_survival_curve(df_all,
                                multiplier=1.0,
                                rule=4.0,
                                title="受給開始年齢別 生存確率推移 (支出レベル1.0, 初年度支出率4%)",
                                output_path=os.path.join(
                                    IMG_DIR,
                                    "survival_curve_pension_m1_r4.svg"),
                                start_age=START_AGE,
                                num_years=NUM_YEARS)

  create_pension_survival_curve(df_all,
                                multiplier=1.0,
                                rule=5.0,
                                title="受給開始年齢別 生存確率推移 (支出レベル1.0, 初年度支出率5%)",
                                output_path=os.path.join(
                                    IMG_DIR,
                                    "survival_curve_pension_m1_r5.svg"),
                                start_age=START_AGE,
                                num_years=NUM_YEARS)

  # 共通の分析関数を呼び出し (ラベル短縮を有効化)
  run_optimal_pension_age_analysis(df_all,
                                   target_year,
                                   IMG_DIR,
                                   START_AGE,
                                   NUM_YEARS,
                                   shorten_labels=True)


def run_formula_analysis(df_all: pd.DataFrame, target_year: str):
  """
  全年金開始年齢の中から最良の生存確率を選択し、そのグリッドに対して詳細分析を実行する。
  """
  print(f"\n\n{'='*20} 統合生存確率グリッドの分析 (最良年金開始年齢選択) {'='*20}")

  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    print("Error: Survival data not found.")
    return

  # 1. 各 (multiplier, rule) において最良の年金開始年齢を選択
  dim_cols = ['spend_multiplier', 'spending_rule']
  best_rows = []
  for _, group in df_survival.groupby(dim_cols):
    # ターゲット年における生存確率が最大のものを選ぶ
    best_row = group.loc[group[target_year].idxmax()].copy()
    best_rows.append(best_row)

  df_best_grid = pd.DataFrame(best_rows)

  # 2. ヒートマップ
  df_h, m_order, r_order = prepare_heatmap_labels(df_best_grid)
  title = f"単身世帯 最適化生存確率 (全年金開始年齢のうち最大値) - {target_year}年後"
  output_path = os.path.join(IMG_DIR, "best_combined_survival_heatmap.svg")

  create_heatmap(df_h,
                 target_col=target_year,
                 title=title,
                 x_col="rule_label",
                 x_title="初期支出率 (%ルール)",
                 y_col="multiplier_label",
                 y_title="支出レベル",
                 output_path=output_path,
                 x_sort=r_order,
                 y_sort=m_order)

  # 3. 生存達成データの生成
  target_probs = [0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.65, 0.60]
  plot_data = []
  for p in target_probs:
    anchors = get_contour_anchor_points(df_best_grid, p, target_year)
    plot_data.extend(generate_smooth_contour_data(anchors, f"{p*100:g}%"))
  df_plot_survival = pd.DataFrame(plot_data)

  # 4. グラフ保存
  save_contour_charts(df_plot_survival,
                      target_probs,
                      img_dir=IMG_DIR,
                      prefix="single_combined_",
                      rule_range=(2.5, 25.0))

  # 5. Rule of Thumb
  generate_rule_of_thumb(df_best_grid, target_probs, target_year)

  # 6. 詳細な近似モデルの分析
  coeffs = run_survival_formula_analysis(df_best_grid, target_year)

  # 7. JSON出力
  if coeffs:
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    out_json = {
        "start_age": START_AGE,
        "household": "single",
        "target_age": START_AGE + int(target_year),
        **coeffs
    }
    json_path = os.path.join(DATA_OUT_DIR, "formula.json")
    with open(json_path, "w") as f:
      json.dump(out_json, f, indent=2)
    print(f"✅ {json_path} を保存しました。")


def main():
  parser = argparse.ArgumentParser(
      description="60歳リタイア開始・95歳までの分析・可視化スクリプト（単身世帯版）。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="optimal-pension",
                      help="実験設定 (optimal-pension)")
  args = parser.parse_args()

  exp_types = args.exp_type.split(",")
  target_year = str(NUM_YEARS)

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_OUT_DIR, exist_ok=True)

  for et in exp_types:
    et = et.strip()
    csv_path = f"data/single_60yr/{et}.csv"
    if not os.path.exists(csv_path):
      print(f"Warning: {csv_path} が見つかりません。スキップします。")
      continue

    print(f"\nProcessing experiment type: {et}")
    df_all = pd.read_csv(csv_path)

    if et == "optimal-pension":
      run_optimal_pension_analysis(df_all, target_year)
      run_formula_analysis(df_all, target_year)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
