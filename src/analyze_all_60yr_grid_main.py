"""
data/all_combinations/all_60yr_grid.csv の結果を分析・可視化するスクリプト。

内容:
1. 予測モデルの評価 (P vs Logit)
2. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
※ダイナミックスペンディングのON/OFF別に分析を行う。
"""

import os
from typing import Any, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# 設定
CSV_PATH = "data/all_combinations/all_60yr_grid.csv"
IMG_DIR = "docs/imgs/all_60yr"
BASE_SPEND_ANNUAL = 540.0


def run_fitting_analysis(df: pd.DataFrame, target_col: str):
  """
  予測モデルの評価を行い STDOUT に出力する。
  """
  print(f"\n--- 予測モデルの評価 (R2 Score) - {target_col}年生存確率 ---")

  y_target = df[target_col].values.astype(float)
  # M = initial_money, S = spending_rule, Mult = spend_multiplier
  M_val = df["initial_money"].values.astype(float)
  S_val = df["spending_rule"].values.astype(float)
  Mult_val = df["spend_multiplier"].values.astype(float)

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


def create_heatmap(df: pd.DataFrame,
                   target_col: str,
                   title: str,
                   x_col: str,
                   x_title: str,
                   y_col: str,
                   y_title: str,
                   output_name: str,
                   x_sort: Optional[List[Any]] = None,
                   y_sort: Optional[List[Any]] = None):
  """
  ヒートマップを作成して保存する。
  """
  plot_df = df.copy()
  plot_df["survival_rate"] = plot_df[target_col]
  plot_df["survival_rate_pct"] = plot_df["survival_rate"] * 100

  base = alt.Chart(plot_df).encode(
      x=alt.X(f'{x_col}:O', title=x_title, sort=x_sort),
      y=alt.Y(f'{y_col}:O', title=y_title, sort=y_sort),
  )

  heatmap = base.mark_rect().encode(color=alt.Color(
      'survival_rate:Q',
      title='生存確率',
      scale=alt.Scale(
          domain=[0.0, 0.8, 0.9, 0.94, 0.97, 1.0],
          range=['#d73027', '#fee08b', '#ffffbf', 'yellowgreen', 'lightgreen', 'green'])))

  text = base.mark_text(baseline='middle').encode(
      text=alt.Text('survival_rate_pct:Q', format='.1f'),
      color=alt.condition(alt.datum.survival_rate > 0.6, alt.value('black'),
                          alt.value('white')))

  chart = (heatmap + text).properties(title=title, width=400, height=300)

  # STDOUT出力
  print(f"\n--- {title} ---")
  pivot = plot_df.pivot_table(index=y_col,
                              columns=x_col,
                              values="survival_rate_pct")
  if y_sort:
    pivot = pivot.reindex(index=y_sort)
  if x_sort:
    pivot = pivot.reindex(columns=x_sort)
  print(pivot.to_string())

  output_path = os.path.join(IMG_DIR, output_name)
  os.makedirs(IMG_DIR, exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def run_optimization_analysis(df: pd.DataFrame,
                              target_col: str,
                              target_prob: float = 0.97):
  """
  指定した生存確率(target_prob)を達成する支出率(spending_rule)を、
  multiplierごとに最適化して求める。
  """
  print(
      f"\n--- {target_col}年生存確率 {target_prob*100:.0f}% を達成する支出率の算出 (線形モデル) ---"
  )

  y_target = df[target_col].values.astype(float)
  M_val = df["initial_money"].values.astype(float)
  S_val = df["spending_rule"].values.astype(float)
  Mult_val = df["spend_multiplier"].values.astype(float)

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

  # 対象となる支出レベル (dict出力用)
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
        x=alt.X('multiplier:Q',
                title='支出レベル',
                axis=alt.Axis(
                  labelExpr=f"format(datum.value * {BASE_SPEND_ANNUAL}, '.0f') + '万(x' + datum.value + ')'")),
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


def main():
  if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} が見つかりません。")
    return

  df_all = pd.read_csv(CSV_PATH)

  # ラベルの日本語化
  df_all["multiplier_label"] = df_all["spend_multiplier"].map(
      lambda x: f"{BASE_SPEND_ANNUAL * x:g}万円/年 (x{x:g})")
  df_all["rule_label"] = df_all["spending_rule"].map(lambda x: f"{x:g}%")

  m_order = [
      f"{BASE_SPEND_ANNUAL * x:g}万円/年 (x{x:g})"
      for x in [3.0, 2.0, 1.5, 1.2, 1.0, 0.75, 0.5, 0.36]
  ]
  # ルール順: 1.5% から 6.0% まで
  r_order = [f"{x:g}%" for x in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]]

  for use_dyn in [0, 1]:
    dyn_label = "dyn_on" if use_dyn == 1 else "dyn_off"
    dyn_title_suffix = "(ダイナミックスペンディングON)" if use_dyn == 1 else "(支出トレンド適用)"

    print(f"\n\n{'='*20} {dyn_label.upper()} {'='*20}")
    df = df_all[df_all["use_dynamic_spending"] == use_dyn].copy()

    for year_target in ["30", "35"]:
      # 1. 分析の実行
      run_fitting_analysis(df, year_target)

      # 2. 可視化
      create_heatmap(
          df,
          target_col=year_target,
          title=f"60歳リタイア開始・{year_target}年後の生存確率(%) {dyn_title_suffix}",
          x_col="rule_label",
          x_title="初期支出率 (%ルール)",
          y_col="multiplier_label",
          y_title="支出レベル",
          output_name=f"grid_heatmap_{year_target}yr_{dyn_label}.svg",
          x_sort=r_order,
          y_sort=m_order)

    # 3. 97%生存確率の最適化 (DYN_OFF, 35年のみ)
    if use_dyn == 0:
      run_optimization_analysis(df, "35", 0.97)


if __name__ == "__main__":
  main()
