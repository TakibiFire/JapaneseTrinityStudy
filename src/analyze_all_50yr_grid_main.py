"""
data/all_50yr/all_50yr_grid.csv の結果を分析・可視化するスクリプト。

内容:
1. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
2. 年金開始年齢 (60 vs 65) の比較分析
3. Dynamic Spending (ON vs OFF) の比較分析
4. 支出額のパーセンタイル推移可視化
5. (受給年齢 × Dynamic Spending) の最適な組み合わせの抽出・分析

NOW: Fix the order.
"""

import os
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.lib.fitting_all_yr import (FeatureSetType, run_fitting_analysis,
                                    run_stepwise_fitting_analysis)
from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels,
                                      run_best_combination_analysis)

# 設定
IMG_DIR = "docs/imgs/all_50yr"
TEMP_IMG_DIR = "temp/all_50yr"


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
                                         start_age=50,
                                         num_years=45)


def run_p60_d1_heatmap(df_survival: pd.DataFrame):
  """
  P60, D1, H1 のヒートマップを作成する。
  """
  print(f"\n\n{'='*20} P60, D1, H1 ヒートマップ生成 {'='*20}")

  if df_survival.empty:
    return

  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)

  year_target = "45"
  title = f"50歳開始・単身世帯・年金60歳・{year_target}年後生存確率(%) (ダイナミックスペンディングON)"
  output_name = f"grid_heatmap_{year_target}yr_h1_p60_dyn_on.svg"
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


def run_survival_curve_analysis(df: pd.DataFrame,
                                model: LinearRegression,
                                selected_feats: List[str],
                                poly: PolynomialFeatures,
                                use_logit: bool,
                                target_probs: List[float]):
  """
  近似式を用いて特定の生存確率を達成する曲線を生成し、データフレームを返す。
  """
  print(f"\n\n{'='*20} 6. 近似式による生存確率達成データの生成 {'='*20}")

  # 支出率の範囲 (2.8% から 7.0%)
  # グラフが綺麗になるよう、少し細かめに刻む
  fine_rules = np.arange(2.8, 7.01, 0.1)

  # 基準となる支出額 (multiplier=1.0 の時)
  base_cost = df[np.isclose(df["spend_multiplier"], 1.0)]["initial_annual_cost"].iloc[0]

  def get_pred_val(s, mult):
    m_money = base_cost * mult / (s / 100.0)
    b_feats = pd.DataFrame({
        "M": [m_money],
        "S": [s],
        "Mult": [mult],
        "invM": [1.0 / max(m_money, 1.0)],
        "invS": [1.0 / max(s, 0.1)],
        "logM": [np.log(max(m_money, 1.0))],
        "logS": [np.log(max(s, 0.1))]
    })
    X_poly = poly.transform(b_feats)
    feature_names = poly.get_feature_names_out(b_feats.columns)
    c_df = pd.DataFrame(X_poly, columns=feature_names)
    X_sel = c_df[selected_feats]
    return model.predict(X_sel)[0]

  plot_data = []
  # 探索範囲
  m_min, m_max = 0.05, 10.0

  for target_p in target_probs:
    target_val = np.log(target_p / (1 - target_p)) if use_logit else target_p
    label = f"{target_p*100:g}%"

    for s in fine_rules:
      try:
        s_val = float(s)
        v_min = get_pred_val(s_val, m_min) - target_val
        v_max = get_pred_val(s_val, m_max) - target_val

        if v_min * v_max <= 0:
          opt_m = brentq(lambda m: get_pred_val(s_val, m) - target_val, m_min,
                         m_max)
          spend = opt_m * base_cost
          plot_data.append({
              "spending_rule": s_val,
              "multiplier": opt_m,
              "annual_spend_man": spend,
              "initial_money": spend / (s_val / 100.0),
              "target_prob": label
          })
      except Exception:
        pass

  df_plot = pd.DataFrame(plot_data)
  return df_plot, base_cost


def save_survival_charts(df_plot: pd.DataFrame, base_cost: float, target_probs: List[float]):
  """
  生成されたデータから3つのグラフを作成して保存する。
  """
  if df_plot.empty:
    print("曲線を描画できるデータが見つかりませんでした。")
    return

  prob_order = [f"{p*100:g}%" for p in target_probs]

  # 1. 支出率 (S) vs 支出レベル (Spend)
  chart1 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('spending_rule:Q',
              title='初期支出率 (%)',
              scale=alt.Scale(domain=[2.8, 7.0])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, base_cost * 3.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (初期支出率 vs 支出レベル)", width=600, height=400)

  path1 = os.path.join(IMG_DIR, "survival_rule_vs_spend.svg")
  chart1.save(path1)
  print(f"✅ {path1} に保存しました。")

  # 2-a: x=総資産, y=支出レベル
  chart2 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('initial_money:Q',
              title='総資産 (万円)',
              scale=alt.Scale(domain=[0, 30000])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, base_cost * 3.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (総資産 vs 支出レベル)", width=600, height=400)

  path2 = os.path.join(IMG_DIR, "survival_asset_vs_spend.svg")
  chart2.save(path2)
  print(f"✅ {path2} に保存しました。")

  # 2-b: x=総資産, y=初期支出率
  chart3 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('initial_money:Q',
              title='総資産 (万円)',
              scale=alt.Scale(domain=[0, 30000])),
      y=alt.Y('spending_rule:Q',
              title='初期支出率 (%)',
              scale=alt.Scale(domain=[2.8, 7.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (総資産 vs 初期支出率)", width=600, height=400)

  path3 = os.path.join(IMG_DIR, "survival_asset_vs_rule.svg")
  chart3.save(path3)
  print(f"✅ {path3} に保存しました。")


def run_rule_of_thumb_analysis(df: pd.DataFrame, target_probs: List[float]):
  """
  Step 10: 資産と支出額の2つの特徴量のみを用いた簡略式の算出とテーブル出力。
  """
  print(f"\n\n{'='*20} 10. 初期支出率を求める公式 (Rule of Thumb) {'='*20}")

  target_col = "45"
  y_raw = df[target_col].to_numpy().astype(float)
  # Logitターゲット
  y_clipped = np.clip(y_raw, 0.0001, 0.9999)
  y_logit = np.log(y_clipped / (1 - y_clipped))

  M_val = df["initial_money"].to_numpy().astype(float)
  Spend_val = df["initial_annual_cost"].to_numpy().astype(float)

  # 指定された2つの特徴量のみを使用
  X = pd.DataFrame({
      "M invSpend": M_val / Spend_val,
      "invSpend": 1.0 / Spend_val
  })

  model = LinearRegression().fit(X, y_logit)
  
  c_m_inv = model.coef_[0]
  c_inv = model.coef_[1]
  intercept = model.intercept_

  print("| 目標生存確率 | 初期支出率を求める公式 |")
  print("| --: | --- |")

  for p in target_probs:
    logit_p = np.log(p / (1 - p))
    k = logit_p - intercept
    
    # Spend = (c1*M + c2) / K
    # Spend = (c1/K)*M + (c2/K)
    slope = c_m_inv / k
    offset = c_inv / k
    
    # 支出率 S = Spend/M * 100 = (slope*M + offset)/M * 100 = slope*100 + (offset*100)/M
    s_base = slope * 100
    s_const = offset

    print(f"| {p*100:g}% | 総資産の {s_base:.1f}% + {s_const:.0f}万円 |")


def main():
  P_D_RANGE_CSV = "data/all_50yr/P-D-RANGE-H1.csv"
  if not os.path.exists(P_D_RANGE_CSV):
    print(f"Error: {P_D_RANGE_CSV} が見つかりません。")
    return

  df_p_d_all = pd.read_csv(P_D_RANGE_CSV)
  df_p_d_survival = df_p_d_all[df_p_d_all["value_type"] == "survival"].copy()

  P60_D1_CSV = "data/all_50yr/P60-D1-H1.csv"
  if not os.path.exists(P60_D1_CSV):
    print(f"Error: {P60_D1_CSV} が見つかりません。")
    return

  df_p60_d1_all = pd.read_csv(P60_D1_CSV)
  df_p60_d1_survival = df_p60_d1_all[df_p60_d1_all["value_type"] ==
                                     "survival"].copy()

  # 1. 最適組み合わせ分析
  run_best_combination_analysis(
      df_p_d_survival,
      target_year="45",
      img_dir=IMG_DIR,
      temp_dir=TEMP_IMG_DIR,
      title_prefix="単身世帯",
      output_name="best_strategy_h1.svg")

  # 2. 支出額パーセンタイル推移の生成
  run_percentile_analysis(df_p_d_all)

  # 3. df_p60_d1_survival からヒートマップを作成
  run_p60_d1_heatmap(df_p60_d1_survival)

  # 4. 予測モデルの評価
  fitting_results = run_fitting_analysis(df_p60_d1_survival, "45")

  # # 5. ステップワイズ特徴量選択による生存確率の近似式算出
  # fitting_results の中から最も Adj R2 が高い Logit 手法を選択する (境界線の算出には Logit が適しているため)
  logit_results = [r for r in fitting_results if r["use_logit"]]
  best_eval = max(logit_results, key=lambda x: x["adj_r2"])

  model_sw, selected_sw, poly_sw = run_stepwise_fitting_analysis(
      df_p60_d1_survival,
      "45",
      max_adj_r2=float(best_eval["adj_r2"]),
      poly_deg=int(best_eval["poly_deg"]),
      interaction_only=bool(best_eval["interaction_only"]),
      use_logit=True)

  # # 6. 生存達成データの生成 (97, 95, 90, 80, 70%)
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  df_plot, base_cost = run_survival_curve_analysis(df_p60_d1_survival,
                                                   model_sw,
                                                   selected_sw,
                                                   poly_sw,
                                                   use_logit=True,
                                                   target_probs=target_probs)

  # 7. 3つのグラフを保存
  save_survival_charts(df_plot, base_cost, target_probs)

  # 8. 資産と支出額のみを用いたモデル評価
  asset_spend_results = run_fitting_analysis(df_p60_d1_survival, "45", feature_set_type=FeatureSetType.ASSET_SPEND)

  # 9. 資産と支出額のみを用いたステップワイズ近似式算出
  # 最も高い Adj R2 を持つ Logit モデルを選択
  logit_as_results = [r for r in asset_spend_results if r["use_logit"]]
  best_as_eval = max(logit_as_results, key=lambda x: x["adj_r2"])

  run_stepwise_fitting_analysis(
      df_p60_d1_survival,
      "45",
      max_adj_r2=float(best_as_eval["adj_r2"]),
      poly_deg=int(best_as_eval["poly_deg"]),
      interaction_only=bool(best_as_eval["interaction_only"]),
      use_logit=True,
      feature_set_type=FeatureSetType.ASSET_SPEND)

  # 10. 初期支出率を求める公式 (Rule of Thumb) の出力
  run_rule_of_thumb_analysis(df_p60_d1_survival, target_probs)


if __name__ == "__main__":
  main()
