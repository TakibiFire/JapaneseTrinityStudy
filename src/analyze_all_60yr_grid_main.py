"""
data/all_60yr/ の結果を分析・可視化するスクリプト。

内容:
1. 最適な組み合わせの分析 (受給開始年齢 × Dynamic Spending)
2. 支出額パーセンタイル推移の生成
3. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
4. 予測モデルの評価 (R2 Score)
5. ステップワイズ特徴量選択による生存確率の近似式算出
6. 生存達成データの生成
7. 生存確率達成ラインのグラフ保存
"""

import argparse
import os

import pandas as pd

from src.lib.visualize_all_yr import (create_pension_survival_curve,
                                      generate_dp_calc_json_common,
                                      run_common_formula_analysis,
                                      run_ds_comparison_analysis,
                                      run_lifeplan_analysis,
                                      run_optimal_pension_age_analysis)

# 設定
IMG_DIR = "docs/imgs/all_60yr"
DATA_OUT_DIR = "docs/data/all_60yr"
TEMP_DIR = "temp/all_60yr"
NUM_YEARS = 35
START_AGE = 60


def run_optimal_pension_analysis(df_all: pd.DataFrame, target_year: str):
  """
  最適な年金受給開始年齢を分析する。
  """
  print(f"\n\n{'='*20} 最適な年金受給開始年齢の分析 {'='*20}")

  # 1. グラフ作成
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

  create_pension_survival_curve(df_all,
                                multiplier=3.0,
                                rule=5.0,
                                title="受給開始年齢別 生存確率推移 (支出レベル2.0, 初年度支出率5%)",
                                output_path=os.path.join(
                                    IMG_DIR,
                                    "survival_curve_pension_m3_r5.svg"),
                                start_age=START_AGE,
                                num_years=NUM_YEARS)

  # 共通の分析関数を呼び出し
  run_optimal_pension_age_analysis(df_all, target_year, IMG_DIR, START_AGE,
                                   NUM_YEARS)


def run_pen70_lifeplan_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-lifeplan の分析を実行する。
  """
  run_lifeplan_analysis(df_all, target_year, IMG_DIR)


def run_pen70_formula_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-formula の詳細分析を実行する。
  """
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    return

  title = f"60歳リタイア・年金70歳・{target_year}年後生存確率(%) (R70-aware)"
  run_common_formula_analysis(df_survival,
                              target_year,
                              IMG_DIR,
                              DATA_OUT_DIR,
                              START_AGE,
                              pension_start=70,
                              title=title,
                              prefix="pen70_formula_",
                              output_json="formula.json")


def run_pen70_ds_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-ds の詳細分析を実行する。
  """
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    return

  formula_path = "data/all_60yr/pen70-formula.csv"
  if not os.path.exists(formula_path):
    return

  df_formula = pd.read_csv(formula_path)
  df_f_surv = df_formula[df_formula["value_type"] == "survival"].copy()

  title = f"60歳リタイア・年金70歳・{target_year}年後生存確率(%) (R70 + SpendAwareDS)"

  run_ds_comparison_analysis(df_survival,
                             df_f_surv,
                             target_year,
                             IMG_DIR,
                             NUM_YEARS,
                             START_AGE,
                             title_main=title,
                             output_prefix="pen70_ds_")


def generate_dp_calc_json(df_all: pd.DataFrame):
  """
  生存確率計算機（DP版）のための設定JSONを生成する。
  """
  generate_dp_calc_json_common(df_all,
                               DATA_OUT_DIR,
                               START_AGE,
                               NUM_YEARS,
                               model_prefix="re60_pen70_95")


def main():
  parser = argparse.ArgumentParser(description="60歳リタイア開始・95歳までの分析・可視化スクリプト。")
  parser.add_argument(
      "--exp_type",
      type=str,
      default="optimal-pension",
      help=
      "実験設定 (comma separated: optimal-pension, pen70-lifeplan, pen70-formula, pen70-ds)"
  )
  args = parser.parse_args()

  exp_types = args.exp_type.split(",")
  target_year = str(NUM_YEARS)

  for et in exp_types:
    et = et.strip()
    csv_path = f"data/all_60yr/{et}.csv"
    if not os.path.exists(csv_path):
      print(f"Warning: {csv_path} が見つかりません。スキップします。")
      continue

    print(f"\nProcessing experiment type: {et}")
    df_all = pd.read_csv(csv_path)

    if et == "optimal-pension":
      run_optimal_pension_analysis(df_all, target_year)
    elif et == "pen70-lifeplan":
      run_pen70_lifeplan_analysis(df_all, target_year)
    elif et == "pen70-formula":
      run_pen70_formula_analysis(df_all, target_year)
      generate_dp_calc_json(df_all)
    elif et == "pen70-ds":
      run_pen70_ds_analysis(df_all, target_year)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
