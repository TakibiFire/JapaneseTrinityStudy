"""
data/all_40yr/P-D-RANGE.csv の結果を分析・可視化するスクリプト。

内容:
1. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
2. 年金開始年齢 (60 vs 65) の比較分析
3. Dynamic Spending (ON vs OFF) の比較分析
4. 支出額のパーセンタイル推移可視化
5. (受給年齢 × Dynamic Spending) の最適な組み合わせの抽出・分析

COMMENT: Keep the above up to date.
"""

import os
from typing import List

import numpy as np
import pandas as pd

from src.lib.fitting_all_yr import (FeatureSetType, run_fitting_analysis,
                                    run_rule_of_thumb_analysis,
                                    run_stepwise_fitting_analysis,
                                    run_survival_curve_analysis,
                                    save_survival_charts)
from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels,
                                      run_best_combination_analysis)

# 設定
IMG_DIR = "docs/imgs/all_40yr"
TEMP_IMG_DIR = "temp/all_40yr"


def run_percentile_analysis(df_all: pd.DataFrame):
  """
  支出額パーセンタイル推移の生成。
  """
  print(f"\n\n{'='*20} 支出額パーセンタイル推移グラフを生成中... {'='*20}")

  household_sizes = sorted(df_all["household_size"].unique())
  pension_start_ages = sorted(df_all["pension_start_age"].unique())
  spend_multipliers = sorted(df_all["spend_multiplier"].unique())
  spending_rules = sorted(df_all["spending_rule"].unique())

  # Overwrite:
  household_sizes = [1]
  pension_start_ages = [60]
  spend_multipliers = [1.0]
  spending_rules = [4]

  for h_size in household_sizes:
    for p_age in pension_start_ages:
      for s_mult in spend_multipliers:
        for rule in spending_rules:
          mask = (df_all["household_size"] == h_size) & \
                 (df_all["pension_start_age"] == p_age) & \
                 (df_all["spend_multiplier"] == s_mult) & \
                 (df_all["spending_rule"] == rule)

          df_plot = df_all[mask]
          if df_plot.empty:
            continue

          # 初期支出額を取得してタイトルに使用
          init_cost = df_plot["initial_annual_cost"].iloc[0]
          h_label = "2人世帯" if h_size == 2 else "単身世帯"
          title = f"年間支出額推移: {h_label}, 年金{p_age}歳, 初期{int(round(init_cost))}万円/年, 初期支出率{rule:g}%"
          output_name = f"spend_percentiles_h{h_size}_p{p_age}_m{s_mult:g}_r{rule:g}.svg"
          output_path = os.path.join(IMG_DIR, output_name)

          create_spend_percentile_chart(df_plot,
                                        title,
                                        output_path,
                                        start_age=40,
                                        num_years=55)


def run_p60_d1_heatmap(df_survival: pd.DataFrame, target_year: str):
  """
  P60, D1, H1 のヒートマップを作成する。
  """
  print(f"\n\n{'='*20} P60, D1, H1 ヒートマップ生成 {'='*20}")

  if df_survival.empty:
    return

  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)

  title = f"40歳開始・単身世帯・年金60歳・{target_year}年後生存確率(%) (ダイナミックスペンディングON)"
  output_name = f"grid_heatmap_{target_year}yr_h1_p60_dyn_on.svg"
  output_path = os.path.join(IMG_DIR, output_name)

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


def main():
  P_D_RANGE_CSV = "data/all_40yr/P-D-RANGE.csv"
  if not os.path.exists(P_D_RANGE_CSV):
    print(f"Error: {P_D_RANGE_CSV} が見つかりません。")
    return

  df_p_d_all = pd.read_csv(P_D_RANGE_CSV)
  df_p_d_survival = df_p_d_all[df_p_d_all["value_type"] == "survival"].copy()

  P60_D1_CSV = "data/all_40yr/P60-D1.csv"
  if not os.path.exists(P60_D1_CSV):
    print(f"Error: {P60_D1_CSV} が見つかりません。")
    return

  df_p60_d1_all = pd.read_csv(P60_D1_CSV)
  df_p60_d1_survival = df_p60_d1_all[df_p60_d1_all["value_type"] ==
                                     "survival"].copy()

  target_year = "55"
  # 1. 最適組み合わせ分析
  run_best_combination_analysis(df_p_d_survival,
                                target_year=target_year,
                                img_dir=IMG_DIR,
                                temp_dir=TEMP_IMG_DIR,
                                title_prefix="単身世帯",
                                output_name="best_strategy_h1.svg")

  # 2. 支出額パーセンタイル推移の生成
  run_percentile_analysis(df_p_d_all)

  # 3. df_p60_d1_survival からヒートマップを作成
  run_p60_d1_heatmap(df_p60_d1_survival, target_year)

  # 4. 予測モデルの評価
  fitting_results = run_fitting_analysis(df_p60_d1_survival, target_year)

  # # 5. ステップワイズ特徴量選択による生存確率の近似式算出
  # fitting_results の中から最も Adj R2 が高い Logit 手法を選択する (境界線の算出には Logit が適しているため)
  logit_results = [r for r in fitting_results if r["use_logit"]]
  best_eval = max(logit_results, key=lambda x: x["adj_r2"])

  model_sw, selected_sw, poly_sw = run_stepwise_fitting_analysis(
      df_p60_d1_survival,
      target_year,
      max_adj_r2=float(best_eval["adj_r2"]),
      poly_deg=int(best_eval["poly_deg"]),
      interaction_only=bool(best_eval["interaction_only"]),
      use_logit=True)

  # 6. 生存達成データの生成 (97, 95, 90, 80, 70%)
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  df_plot, base_cost = run_survival_curve_analysis(df_p60_d1_survival,
                                                   model_sw,
                                                   selected_sw,
                                                   poly_sw,
                                                   use_logit=True,
                                                   target_probs=target_probs)

  # 7. 3つのグラフを保存
  save_survival_charts(df_plot, base_cost, target_probs, img_dir=IMG_DIR)

  # 8. 資産と支出額のみを用いたモデル評価
  asset_spend_results = run_fitting_analysis(
      df_p60_d1_survival,
      target_year,
      feature_set_type=FeatureSetType.ASSET_SPEND)

  # 9. 資産と支出額のみを用いたステップワイズ近似式算出
  # 最も高い Adj R2 を持つ Logit モデルを選択
  logit_as_results = [r for r in asset_spend_results if r["use_logit"]]
  best_as_eval = max(logit_as_results, key=lambda x: x["adj_r2"])

  run_stepwise_fitting_analysis(df_p60_d1_survival,
                                target_year,
                                max_adj_r2=float(best_as_eval["adj_r2"]),
                                poly_deg=int(best_as_eval["poly_deg"]),
                                interaction_only=bool(
                                    best_as_eval["interaction_only"]),
                                use_logit=True,
                                feature_set_type=FeatureSetType.ASSET_SPEND)

  # 10. 初期支出率を求める公式 (Rule of Thumb) の出力
  run_rule_of_thumb_analysis(df_p60_d1_survival, target_year, target_probs)


if __name__ == "__main__":
  main()
