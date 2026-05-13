"""
data/single_50yr/ の結果を分析・可ビジュアル化するスクリプト（単身世帯版）。

内容:
1. 最適な組み合わせの分析 (受給開始年齢 × Dynamic Spending)
2. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
"""

import argparse
import os

import pandas as pd

from src.lib.visualize_all_yr import (create_pension_survival_curve,
                                      prepare_heatmap_labels,
                                      run_common_formula_analysis,
                                      run_optimal_pension_age_analysis)

# 設定
IMG_DIR = "docs/imgs/single_50yr"
DATA_OUT_DIR = "docs/data/single_50yr"
NUM_YEARS = 45
START_AGE = 50


def run_optimal_pension_analysis(df_all: pd.DataFrame, target_year: str):
  """
  最適な年金受給開始年齢を分析する。
  """
  print(f"\n\n{'='*20} 最適な年金受給開始年齢の分析 {'='*20}")

  # 1. グラフ作成 (m=1, r=4% と m=1.5, r=7%)
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
                                multiplier=1.5,
                                rule=7.0,
                                title="受給開始年齢別 生存確率推移 (支出レベル1.5, 初年度支出率7%)",
                                output_path=os.path.join(
                                    IMG_DIR,
                                    "survival_curve_pension_m1_5_r7.svg"),
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
  df_best_grid["household"] = "single"

  title = f"単身世帯 最適化生存確率 (全年金開始年齢のうち最大値) - {target_year}年後"
  target_probs = [0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.65, 0.60]

  run_common_formula_analysis(
      df_best_grid,
      target_year,
      IMG_DIR,
      DATA_OUT_DIR,
      START_AGE,
      pension_start=0,  # 最良を選択しているため
      title=title,
      prefix="single_combined_",
      target_probs=target_probs,
      output_json="formula.json",
      generate_heatmap=True)


def main():
  parser = argparse.ArgumentParser(
      description="50歳リタイア開始・95歳までの分析・可視化スクリプト（単身世帯版）。")
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
    csv_path = f"data/single_50yr/{et}.csv"
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
