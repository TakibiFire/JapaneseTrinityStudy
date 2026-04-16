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

import os
from typing import Any, Dict, List, Optional, cast

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.lib.fitting_all_yr import (FeatureSetType, run_fitting_analysis,
                                    run_stepwise_fitting_analysis,
                                    run_survival_curve_analysis,
                                    save_survival_charts)
from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels,
                                      run_best_combination_analysis)

# 設定
IMG_DIR = "docs/imgs/all_60yr"
TEMP_DIR = "temp/all_60yr"
BASE_SPEND_ANNUAL = 540.0


def run_p60_d1_heatmap(df_survival: pd.DataFrame):
  """
  P60, D1 のヒートマップを作成する。
  """
  print(f"\n\n{'='*20} P60, D1 ヒートマップ生成 {'='*20}")

  if df_survival.empty:
    return

  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)

  year_target = "35"
  title = f"60歳リタイア・年金60歳・{year_target}年後生存確率(%) (ダイナミックスペンディングON)"
  output_name = f"grid_heatmap_{year_target}yr_p60_dyn_on.svg"
  output_path = os.path.join(IMG_DIR, output_name)

  create_heatmap(df_h,
                 target_col=year_target,
                 title=title,
                 x_col="rule_label",
                 x_title="初期支出率 (%ルール)",
                 y_col="multiplier_label",
                 y_title="支出レベル",
                 output_path=output_path,
                 x_sort=r_order,
                 y_sort=m_order)


def main():
  P_D_RANGE_CSV = "data/all_60yr/P-D-RANGE.csv"
  if not os.path.exists(P_D_RANGE_CSV):
    print(f"Error: {P_D_RANGE_CSV} が見つかりません。")
    return

  df_p_d_all = pd.read_csv(P_D_RANGE_CSV)
  df_p_d_survival = df_p_d_all[df_p_d_all["value_type"] == "survival"].copy()

  P60_D1_CSV = "data/all_60yr/P60-D1.csv"
  if not os.path.exists(P60_D1_CSV):
    print(f"Error: {P60_D1_CSV} が見つかりません。")
    return

  df_p60_d1_all = pd.read_csv(P60_D1_CSV)
  df_p60_d1_survival = df_p60_d1_all[df_p60_d1_all["value_type"] ==
                                     "survival"].copy()

  # 1. 最適な組み合わせの分析 (35年後)
  run_best_combination_analysis(
      df_p_d_survival,
      target_year="35",
      img_dir=IMG_DIR,
      temp_dir=TEMP_DIR,
      title_prefix="60歳リタイア",
      threshold=0.02,
      pref_order=["P60_D1", "P65_D1", "P60_D0", "P65_D0"],  # 優先順位: 60歳ありを最優先
      width=500,
      height=450)

  # 2. 支出額パーセンタイル推移の生成
  # Only care about P60, D1, Mult1, Rule4 case.
  df_plot = df_p_d_all[(df_p_d_all["pension_start_age"] == 60) &
                       (df_p_d_all["spend_multiplier"] == 1.0) &
                       (df_p_d_all["spending_rule"] == 4.0)]
  if not df_plot.empty:
    title = "年間支出額推移: 60歳リタイア, 年金60歳, 初期540万円/年, 初期支出率4%"
    output_path = os.path.join(IMG_DIR, "spend_percentiles_60yr_p60_m1_r4.svg")
    create_spend_percentile_chart(df_plot,
                                  title,
                                  output_path,
                                  start_age=60,
                                  num_years=35)

  # 3. df_p60_d1_survival からヒートマップを作成
  run_p60_d1_heatmap(df_p60_d1_survival)

  # 4. 予測モデルの評価
  target_col = "35"
  fitting_results = run_fitting_analysis(df_p60_d1_survival, target_col)

  # 5. ステップワイズ特徴量選択による生存確率の近似式算出
  # fitting_results の中から最も Adj R2 が高い Logit 手法を選択する
  logit_results = [r for r in fitting_results if r["use_logit"]]
  best_eval = max(logit_results, key=lambda x: x["adj_r2"])

  model_sw, selected_sw, poly_sw = run_stepwise_fitting_analysis(
      df_p60_d1_survival,
      target_col,
      max_adj_r2=float(best_eval["adj_r2"]),
      poly_deg=int(best_eval["poly_deg"]),
      interaction_only=bool(best_eval["interaction_only"]),
      use_logit=True)

  # 6. 生存達成データの生成 (97, 95, 90, 80, 70%)
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  df_plot_survival, base_cost = run_survival_curve_analysis(
      df_p60_d1_survival,
      model_sw,
      selected_sw,
      poly_sw,
      use_logit=True,
      target_probs=target_probs)

  # 7. 3つのグラフを保存
  save_survival_charts(df_plot_survival, base_cost, target_probs, img_dir=IMG_DIR)


if __name__ == "__main__":
  main()
