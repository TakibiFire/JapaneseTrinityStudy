"""
data/all_60yr/ の結果を分析・可視化するスクリプト。

内容:
1. 予測モデルの評価 (P vs Logit)
2. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
3. 最適な組み合わせの分析 (受給開始年齢 × Dynamic Spending)
"""

import os
from typing import Any, Dict, List, Optional, cast

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels,
                                      run_best_combination_analysis)

# 設定
IMG_DIR = "docs/imgs/all_60yr"
TEMP_DIR = "temp/all_60yr"
BASE_SPEND_ANNUAL = 540.0


def run_fitting_analysis(df: pd.DataFrame, target_col: str):
  """
  予測モデルの評価を行い STDOUT に出力する。
  """
  print(f"\n--- 予測モデルの評価 (R2 Score) - {target_col}年生存確率 ---")

  y_target = np.asarray(df[target_col].to_numpy(), dtype=float)
  # M = initial_money, S = spending_rule, Mult = spend_multiplier
  M_val = np.asarray(df["initial_money"].to_numpy(), dtype=float)
  S_val = np.asarray(df["spending_rule"].to_numpy(), dtype=float)
  Mult_val = np.asarray(df["spend_multiplier"].to_numpy(), dtype=float)

  # 対数オッズ空間
  y_clipped = np.clip(y_target, 0.001, 0.999)
  logit_y = np.log(y_clipped / (1 - y_clipped))

  # 特徴量セット
  feats = pd.DataFrame({
      "M": M_val,
      "S": S_val,
      "Mult": Mult_val,
      "invM": 1.0 / np.maximum(M_val, 1.0),
      "invS": 1.0 / np.maximum(S_val, 0.1),
      "logM": np.log(np.maximum(M_val, 1.0)),
      "logS": np.log(np.maximum(S_val, 0.1))
  })

  def evaluate(name, poly_deg, interaction_only, use_logit):
    poly = PolynomialFeatures(degree=poly_deg,
                              interaction_only=interaction_only)
    X_poly = poly.fit_transform(feats)

    target = logit_y if use_logit else y_target
    model = LinearRegression()
    model.fit(X_poly, target)

    pred_raw = model.predict(X_poly)
    if use_logit:
      y_pred = 1 / (1 + np.exp(-pred_raw))
    else:
      y_pred = np.clip(pred_raw, 0, 1)

    r2 = r2_score(y_target, y_pred)
    print(f"{name:<40} | R2: {r2:.4f} | 特徴量数: {X_poly.shape[1]}")

  evaluate("確率Pに対する線形モデル", 1, False, False)
  evaluate("Logitに対する線形モデル", 1, False, True)
  evaluate("確率Pに対する2次モデル (交互作用のみ)", 2, True, False)
  evaluate("Logitに対する2次モデル (交互作用のみ)", 2, True, True)


def run_optimization_analysis(df: pd.DataFrame,
                              target_col: str,
                              target_prob: float = 0.97):
  """
  指定した生存確率(target_prob)を達成する支出率(spending_rule)を、
  multiplierごとに最適化して求める。
  """
  print(
      f"\n--- {target_col}年生存確率 {target_prob*100:.0f}% を達成する支出率の算出 (線形モデル) ---")

  y_target = np.asarray(df[target_col].to_numpy(), dtype=float)
  M_val = np.asarray(df["initial_money"].to_numpy(), dtype=float)
  S_val = np.asarray(df["spending_rule"].to_numpy(), dtype=float)
  Mult_val = np.asarray(df["spend_multiplier"].to_numpy(), dtype=float)

  # 特徴量セット (run_fitting_analysis と合わせる)
  feats_df = pd.DataFrame({
      "M": M_val,
      "S": S_val,
      "Mult": Mult_val,
      "invM": 1.0 / np.maximum(M_val, 1.0),
      "invS": 1.0 / np.maximum(S_val, 0.1),
      "logM": np.log(np.maximum(M_val, 1.0)),
      "logS": np.log(np.maximum(S_val, 0.1))
  })

  poly = PolynomialFeatures(degree=1)
  X_poly = poly.fit_transform(feats_df)

  model = LinearRegression()
  model.fit(X_poly, y_target)

  def predict_prob(s, mult):
    m = (BASE_SPEND_ANNUAL * mult) / (s / 100.0)
    f = pd.DataFrame({
        "M": [m],
        "S": [s],
        "Mult": [mult],
        "invM": [1.0 / max(m, 1.0)],
        "invS": [1.0 / max(s, 0.1)],
        "logM": [np.log(max(m, 1.0))],
        "logS": [np.log(max(s, 0.1))]
    })
    X = poly.transform(f)
    return model.predict(X)[0]

  # 対象となる支出レベル
  target_multipliers = [0.36, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]

  # 1. 実際のデータ範囲で target_prob を達成可能な multiplier の範囲を特定
  crossable_multipliers = []
  for mult in sorted(df["spend_multiplier"].unique()):
    df_m = df[df["spend_multiplier"] == mult]
    if not df_m.empty:
      if df_m[target_col].min() <= target_prob <= df_m[target_col].max():
        crossable_multipliers.append(mult)

  # 最適化の実行 (dict用)
  calculated_dict = {}
  for mult in target_multipliers:
    if any(abs(mult - c) < 1e-6 for c in crossable_multipliers):
      try:
        opt_s = brentq(lambda s: predict_prob(s, mult) - target_prob, 1.0, 20.0)
        calculated_dict[mult] = round(opt_s, 2)
      except ValueError:
        pass

  # 結果の構築 (dict用)
  results_all = {}
  if calculated_dict:
    max_rule = max(calculated_dict.values())
    for mult in target_multipliers:
      if mult in calculated_dict:
        results_all[mult] = calculated_dict[mult]
      else:
        # 算出できなかった（＝安全すぎる）場合は最大値を採用
        results_all[mult] = max_rule

  # 結果の出力
  print("\noptimized_rules = {")
  for mult in sorted(results_all.keys()):
    print(f"    {mult}: {results_all[mult]},")
  print("}")

  # 2. グラフ用データの算出 (特定した範囲内を 0.1 step で)
  plot_data = []
  if crossable_multipliers:
    m_start = min(crossable_multipliers)
    m_end = max(crossable_multipliers)
    fine_multipliers = np.arange(m_start, m_end + 0.01, 0.1)

    for mult in fine_multipliers:
      try:
        opt_s = brentq(lambda s: predict_prob(s, mult) - target_prob, 1.0, 20.0)
        plot_data.append({"multiplier": mult, "optimal_rule": opt_s})
      except ValueError:
        pass

  df_plot = pd.DataFrame(plot_data)

  if not df_plot.empty:
    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x=alt.X(
            'multiplier:Q',
            title='支出レベル',
            axis=alt.
            Axis(labelExpr
                 =f"format(datum.value * {BASE_SPEND_ANNUAL}, '.0f') + '万(x' + datum.value + ')'"
                )),
        y=alt.Y('optimal_rule:Q',
                title=f'{target_prob*100:.0f}%生存達成ルール (%)',
                scale=alt.Scale(zero=False)),
        tooltip=['multiplier', 'optimal_rule']).properties(
            title=f'支出レベルごとの{target_prob*100:.0f}%生存達成ルール (35年・低倍率は常に97%超)',
            width=600,
            height=350)

    os.makedirs(IMG_DIR, exist_ok=True)
    output_path = os.path.join(IMG_DIR, "survival_97_rule_curve.svg")
    chart.save(output_path)
    print(f"✅ {output_path} に保存しました。")


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

  # 3. df_p60_d1_survival からヒートマップの作成


if __name__ == "__main__":
  main()
