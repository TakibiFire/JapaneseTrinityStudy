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
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 設定
IMG_DIR = "docs/imgs/all_50yr"
TEMP_IMG_DIR = "temp/all_50yr"


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
      x=alt.X(f'{x_col}:O',
              title=x_title,
              sort=x_sort,
              axis=alt.Axis(labelExpr="split(datum.label, '@')")),
      y=alt.Y(f'{y_col}:O',
              title=y_title,
              sort=y_sort,
              axis=alt.Axis(labelExpr="split(datum.label, '@')")),
  )

  heatmap = base.mark_rect().encode(
      color=alt.Color('survival_rate:Q',
                      title='生存確率',
                      scale=alt.Scale(domain=[0.0, 0.8, 0.9, 0.94, 0.97, 1.0],
                                      range=[
                                          '#d73027', '#fee08b', '#ffffbf',
                                          'yellowgreen', 'lightgreen', 'green'
                                      ])))

  text = base.mark_text(baseline='middle').encode(
      text=alt.Text('survival_rate_pct:Q', format='.1f'),
      color=alt.condition(alt.datum.survival_rate > 0.6, alt.value('black'),
                          alt.value('white')))

  chart = (heatmap + text).properties(title=title, width=500, height=400)

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


def create_best_combo_heatmap(df_best: pd.DataFrame,
                              title: str,
                              x_col: str,
                              x_title: str,
                              y_col: str,
                              y_title: str,
                              output_name: str,
                              x_sort: Optional[List[Any]] = None,
                              y_sort: Optional[List[Any]] = None):
  """
  最適な組み合わせ(Pxx_Dx)を可視化するヒートマップ。
  """
  plot_df = df_best.copy()

  # Categorical color scale for combos
  color_map = {
      "65歳,あり": "#9AE6B4",  # Light green
      "65歳,なし": "#B2F5EA",  # Light teal
      "60歳,あり": "#FBD38D",  # Light orange
      "60歳,なし": "#FEB2B2"  # Light red
  }
  domain = list(color_map.keys())
  range_ = list(color_map.values())

  base = alt.Chart(plot_df).encode(
      x=alt.X(f'{x_col}:O',
              title=x_title,
              sort=x_sort,
              axis=alt.Axis(labelExpr="split(datum.label, '@')")),
      y=alt.Y(f'{y_col}:O',
              title=y_title,
              sort=y_sort,
              axis=alt.Axis(labelExpr="split(datum.label, '@')")),
  )

  heatmap = base.mark_rect().encode(
      color=alt.Color('display_combo:N',
                      title='選択された戦略',
                      scale=alt.Scale(domain=domain, range=range_)))

  # Text shows strategy, its prob, and dyn flip gap with newline
  text = base.mark_text(baseline='middle',
                        lineBreak='\n').encode(text=alt.Text('combo_label:N'),
                                               color=alt.value('black'))

  chart = (heatmap + text).properties(title=title, width=500, height=450)

  output_path = os.path.join(IMG_DIR, output_name)
  os.makedirs(IMG_DIR, exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def create_spend_percentile_chart(df: pd.DataFrame, title: str,
                                  output_name: str):
  """
  支出額のパーセンタイル推移を可視化する（DynamicSpending ON/OFF比較）。
  """
  # 年度列（1-45）をロング形式に変換
  id_vars = [
      "household_size", "pension_start_age", "spend_multiplier",
      "use_dynamic_spending", "spending_rule", "value_type"
  ]
  if "exp_name" in df.columns:
    id_vars.append("exp_name")

  year_cols = [c for c in df.columns if c.isdigit()]

  df_long = df.melt(id_vars=id_vars,
                    value_vars=year_cols,
                    var_name="year",
                    value_name="spend")
  df_long["year"] = df_long["year"].astype(int)
  # 年数から年齢(開始50歳)に変換
  df_long["age"] = df_long["year"] + 50

  df_long["dyn_label"] = df_long["use_dynamic_spending"].map({
      1: "ON",
      0: "OFF"
  })

  # p25, p50, p75 を列に展開
  pivot_df = df_long.pivot_table(index=["dyn_label", "age"],
                                 columns="value_type",
                                 values="spend").reset_index()

  # Altairでプロット
  base = alt.Chart(pivot_df).encode(x=alt.X("age:Q", title="年齢"))

  # Area (25p-75p)
  area = base.mark_area(opacity=0.3).encode(
      y=alt.Y("spend25p:Q", title="年間支出額 (万円)"),
      y2="spend75p:Q",
      color=alt.Color("dyn_label:N",
                      scale=alt.Scale(domain=["ON", "OFF"],
                                      range=["red", "blue"]),
                      title="Dynamic Spending"))

  # Line (50p)
  line = base.mark_line().encode(y="spend50p:Q", color=alt.Color("dyn_label:N"))

  chart = (area + line).properties(title=title, width=600, height=400)

  output_path = os.path.join(IMG_DIR, output_name)
  os.makedirs(IMG_DIR, exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def run_dynamic_spending_analysis(df: pd.DataFrame):
  """
  Dynamic Spending (ON vs OFF) の比較分析を行う。
  """
  print("\n\n" + "=" * 20 + " Dynamic Spending の比較分析 (ON vs OFF) " + "=" * 20)

  target_year = "45"
  idx_cols = [
      'household_size', 'pension_start_age', 'spend_multiplier', 'spending_rule'
  ]

  # df は value_type == "survival" であることを前提とする
  df_on = df[df['use_dynamic_spending'] == 1].set_index(idx_cols)[target_year]
  df_off = df[df['use_dynamic_spending'] == 0].set_index(idx_cols)[target_year]

  diff = df_on - df_off

  print(f"\n--- {target_year}年後生存確率の差分 (ON - OFF) の統計 ---")
  print(diff.describe().to_string())

  print("\n--- ONが有利なケース (差分 > 1.0%) ---")
  print(diff[diff > 0.01].to_string())

  print("\n--- OFFが有利なケース (差分 < -1.0%) ---")
  print(diff[diff < -0.01].to_string())


def run_heatmap_analysis(df_survival: pd.DataFrame):
  """
  各種ディメンションでヒートマップを生成する。
  """
  household_sizes = sorted(df_survival["household_size"].unique())
  pension_start_ages = sorted(df_survival["pension_start_age"].unique())
  use_dyn_list = sorted(df_survival["use_dynamic_spending"].unique())

  for h_size in household_sizes:
    for p_age in pension_start_ages:
      for use_dyn in use_dyn_list:
        dyn_label = "dyn_on" if use_dyn == 1 else "dyn_off"
        dyn_title_suffix = "(ダイナミックスペンディングON)" if use_dyn == 1 else "(支出トレンド適用)"
        h_label = "2人世帯" if h_size == 2 else "単身世帯"

        print(
            f"\n\n{'='*20} {h_label} / 年金{p_age}歳開始 / {dyn_label.upper()} {'='*20}"
        )

        mask = (df_survival["household_size"] == h_size) & \
               (df_survival["pension_start_age"] == p_age) & \
               (df_survival["use_dynamic_spending"] == use_dyn)
        df_h = df_survival[mask].copy()

        if df_h.empty:
          continue

        # ラベルの日本語化と表示順の設定
        df_h["multiplier_label"] = df_h.apply(
            lambda r:
            f"{int(round(r['initial_annual_cost'])):d}万円/年@(x{r['spend_multiplier']:g})",
            axis=1)
        df_h["rule_label"] = df_h["spending_rule"].map(
            lambda x: f"{x:g}%@(x{round(100/x, 1):g})")

        actual_multipliers = sorted(df_h["spend_multiplier"].unique(),
                                    reverse=True)
        actual_rules = sorted(df_h["spending_rule"].unique())

        m_order = []
        for m in actual_multipliers:
          cost = df_h[df_h["spend_multiplier"] ==
                      m]["initial_annual_cost"].iloc[0]
          m_order.append(f"{int(round(cost)):d}万円/年@(x{m:g})")
        r_order = [f"{x:g}%@(x{round(100/x, 1):g})" for x in actual_rules]

        for year_target in ["45"]:
          title = f"50歳開始・{h_label}・年金{p_age}歳・{year_target}年後生存確率(%) {dyn_title_suffix}"
          output_name = f"grid_heatmap_{year_target}yr_h{h_size}_p{p_age}_{dyn_label}.svg"

          create_heatmap(df_h,
                         target_col=year_target,
                         title=title,
                         x_col="rule_label",
                         x_title="初期支出率 (%ルール)",
                         y_col="multiplier_label",
                         y_title="支出レベル",
                         output_name=output_name,
                         x_sort=r_order,
                         y_sort=m_order)


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
                 (df_all["spending_rule"] == rule) & \
                 (df_all["value_type"].isin(["spend25p", "spend50p", "spend75p"]))

          df_plot = df_all[mask].copy()
          if df_plot.empty:
            continue

          # 初期支出額を取得してタイトルに使用
          init_cost = df_plot["initial_annual_cost"].iloc[0]
          h_label = "2人世帯" if h_size == 2 else "単身世帯"
          title = f"年間支出額推移: {h_label}, 年金{p_age}歳, 初期{int(round(init_cost))}万円/年, 初期支出率{rule:g}%"
          output_name = f"spend_percentiles_h{h_size}_p{p_age}_m{s_mult:g}_r{rule:g}.svg"

          create_spend_percentile_chart(df_plot, title, output_name)


def run_best_combination_analysis(df_survival: pd.DataFrame):
  """
  (pension_start_age, use_dynamic_spending) の最適な組み合わせを分析する。
  """
  print("\n\n" + "=" * 20 + " 最適な組み合わせ (受給開始年齢 × Dynamic Spending) の分析 " +
        "=" * 20)

  target_year = "45"
  # 分析対象のディメンション
  dim_cols = ['household_size', 'spend_multiplier', 'spending_rule']

  # 優先順位: P60_D1, P65_D1, P60_D0, P65_D0
  PREF_ORDER = ["P60_D1", "P65_D1", "P60_D0", "P65_D0"]

  def get_selected_strategy(group: pd.DataFrame) -> pd.Series:
    # 生存確率で降順ソート
    sorted_group = group.sort_values(by=[target_year], ascending=False)
    top_prob = float(sorted_group[target_year].max())

    # 組み合わせラベルを作成
    sorted_group["combo"] = sorted_group.apply(
        lambda r:
        f"P{int(r['pension_start_age'])}_D{int(r['use_dynamic_spending'])}",
        axis=1)

    # ユーザー指定の優先順位でスキャン
    selected_row = None
    for pref in PREF_ORDER:
      match = sorted_group[sorted_group["combo"] == pref]
      if not match.empty:
        row = match.iloc[0]
        # 許容範囲: top_prob - 0.01 以上なら採用
        if float(row[target_year]) >= (top_prob - 0.01):
          selected_row = row.copy()
          break

    if selected_row is None:
      selected_row = sorted_group.iloc[0].copy()

    selected_row["best_combo"] = selected_row["combo"]

    # DynamicSpending を反転させた場合との比較 (gap)
    target_p_age = selected_row['pension_start_age']
    target_use_dyn = selected_row['use_dynamic_spending']
    flipped_dyn = 1 - target_use_dyn

    flipped_row = group[(group['pension_start_age'] == target_p_age) &
                        (group['use_dynamic_spending'] == flipped_dyn)]

    if not flipped_row.empty:
      prob_flipped = float(flipped_row.iloc[0][target_year])
      gap = float(selected_row[target_year]) - prob_flipped
      selected_row["dyn_gap"] = gap
    else:
      selected_row["dyn_gap"] = 0.0

    # 表示用ラベルの作成
    dyn_str = "あり" if selected_row['use_dynamic_spending'] == 1 else "なし"
    selected_row[
        "display_combo"] = f"{int(selected_row['pension_start_age'])}歳,{dyn_str}"

    return selected_row

  # Mypy のためにリストを使う
  results = []
  for keys, group in df_survival.groupby(dim_cols):
    results.append(get_selected_strategy(group))
  df_best = pd.DataFrame(results)

  # Heatmap text label (3rd row added: gap with flipped dyn)
  df_best["combo_label"] = df_best.apply(
      lambda r:
      f"{r['display_combo']}\n{r[target_year]*100:.1f}%\n({r['dyn_gap']*100:+.1f}%)",
      axis=1)

  # ヒートマップ用のラベル作成
  df_best["multiplier_label"] = df_best.apply(
      lambda r:
      f"{int(round(r['initial_annual_cost'])):d}万円/年@(x{r['spend_multiplier']:g})",
      axis=1)
  df_best["rule_label"] = df_best["spending_rule"].map(
      lambda x: f"{x:g}%@(x{round(100/x, 1):g})")

  actual_multipliers = sorted(df_best["spend_multiplier"].unique(),
                              reverse=True)
  actual_rules = sorted(df_best["spending_rule"].unique())

  m_order = []
  for m in actual_multipliers:
    cost = df_best[df_best["spend_multiplier"] ==
                   m]["initial_annual_cost"].iloc[0]
    m_order.append(f"{int(round(cost)):d}万円/年@(x{m:g})")
  r_order = [f"{x:g}%@(x{round(100/x, 1):g})" for x in actual_rules]

  # ヒートマップ生成
  for h_size in sorted(df_best["household_size"].unique()):
    h_label = "2人世帯" if h_size == 2 else "単身世帯"
    df_h = df_best[df_best["household_size"] == h_size].copy()

    create_best_combo_heatmap(
        df_h,
        title=f"戦略選択: {h_label} (優先順: 60歳あり > 65歳あり > 60歳なし > 65歳なし, 許容差1%)",
        x_col="rule_label",
        x_title="初期支出率 (%ルール)",
        y_col="multiplier_label",
        y_title="支出レベル",
        output_name=f"best_strategy_h{h_size}.svg",
        x_sort=r_order,
        y_sort=m_order)

  # CSV出力
  csv_output = os.path.join(TEMP_IMG_DIR, "best_strategy.csv")
  os.makedirs(TEMP_IMG_DIR, exist_ok=True)
  df_best.to_csv(csv_output, index=False)
  print(f"✅ {csv_output} にCSVを保存しました。")

  print(f"\n--- {target_year}年後生存確率を最大化する組み合わせの分布 ---")
  counts = df_best["display_combo"].value_counts().sort_index()
  print(counts.to_string())


def run_p60_d1_heatmap(df_survival: pd.DataFrame):
  """
  P60, D1, H1 のヒートマップを作成する。
  """
  print(f"\n\n{'='*20} P60, D1, H1 ヒートマップ生成 {'='*20}")

  df_h = df_survival.copy()
  if df_h.empty:
    return

  # ラベルの日本語化と表示順の設定
  df_h["multiplier_label"] = df_h.apply(
      lambda r:
      f"{int(round(r['initial_annual_cost'])):d}万円/年@(x{r['spend_multiplier']:g})",
      axis=1)
  df_h["rule_label"] = df_h["spending_rule"].map(
      lambda x: f"{x:g}%@(x{round(100/x, 1):g})")

  actual_multipliers = sorted(df_h["spend_multiplier"].unique(), reverse=True)
  actual_rules = sorted(df_h["spending_rule"].unique())

  m_order = []
  for m in actual_multipliers:
    cost = df_h[df_h["spend_multiplier"] == m]["initial_annual_cost"].iloc[0]
    m_order.append(f"{int(round(cost)):d}万円/年@(x{m:g})")
  r_order = [f"{x:g}%@(x{round(100/x, 1):g})" for x in actual_rules]

  year_target = "45"
  title = f"50歳開始・単身世帯・年金60歳・{year_target}年後生存確率(%) (ダイナミックスペンディングON)"
  output_name = f"grid_heatmap_{year_target}yr_h1_p60_dyn_on.svg"

  create_heatmap(df_h,
                 target_col=year_target,
                 title=title,
                 x_col="rule_label",
                 x_title="初期支出率 (%ルール)",
                 y_col="multiplier_label",
                 y_title="支出レベル",
                 output_name=output_name,
                 x_sort=r_order,
                 y_sort=m_order)


def run_fitting_analysis(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
  """
  予測モデルの評価を行い STDOUT に出力する。
  """
  print(f"\n\n{'='*20} 予測モデルの評価 (R2 Score) - {target_col}年生存確率 {'='*20}")

  y_target = df[target_col].to_numpy().astype(float)
  M_val = df["initial_money"].to_numpy().astype(float)
  S_val = df["spending_rule"].to_numpy().astype(float)
  Mult_val = df["spend_multiplier"].to_numpy().astype(float)

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

  results = []

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
    n = len(y_target)
    p = X_poly.shape[1] - 1  # 定数項を除く
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(
        f"{name:<40} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f} | 特徴量数: {X_poly.shape[1]}"
    )
    results.append({
        "name": name,
        "poly_deg": poly_deg,
        "interaction_only": interaction_only,
        "use_logit": use_logit,
        "adj_r2": adj_r2
    })

  evaluate("確率Pに対する2次モデル (交互作用のみ)", 2, True, False)
  evaluate("Logitに対する2次モデル (交互作用のみ)", 2, True, True)
  evaluate("確率Pに対する3次モデル (交互作用のみ)", 3, True, False)
  evaluate("Logitに対する3次モデル (交互作用のみ)", 3, True, True)

  return results


def run_stepwise_fitting_analysis(df: pd.DataFrame,
                                  target_col: str,
                                  max_adj_r2: float,
                                  poly_deg: int = 2,
                                  interaction_only: bool = True,
                                  use_logit: bool = False):
  """
  ステップワイズ特徴量選択による生存確率の近似式算出。
  """
  print(f"\n\n{'='*20} 5. ステップワイズ特徴量選択による生存確率の近似式算出 {'='*20}")
  print(
      f"ターゲット: {target_col}年生存確率, Logit使用: {use_logit}, Degree: {poly_deg}, InteractionOnly: {interaction_only}"
  )
  threshold = max_adj_r2 * 0.99
  print(f"目標 Adj R2 (99% of {max_adj_r2:.4f}): {threshold:.4f}")

  y_raw = df[target_col].to_numpy().astype(float)
  if use_logit:
    y_clipped = np.clip(y_raw, 0.0001, 0.9999)
    y_target = np.log(y_clipped / (1 - y_clipped))
  else:
    y_target = y_raw

  # 特徴量の準備
  M_val = df["initial_money"].to_numpy().astype(float)
  S_val = df["spending_rule"].to_numpy().astype(float)
  Mult_val = df["spend_multiplier"].to_numpy().astype(float)

  base_feats = pd.DataFrame({
      "M": M_val,
      "S": S_val,
      "Mult": Mult_val,
      "invM": 1.0 / np.maximum(M_val, 1.0),
      "invS": 1.0 / np.maximum(S_val, 0.1),
      "logM": np.log(np.maximum(M_val, 1.0)),
      "logS": np.log(np.maximum(S_val, 0.1))
  })

  poly = PolynomialFeatures(degree=poly_deg, interaction_only=interaction_only)
  X_all = poly.fit_transform(base_feats)
  feature_names = poly.get_feature_names_out(base_feats.columns)

  candidate_df = pd.DataFrame(X_all, columns=feature_names)
  if "1" in candidate_df.columns:
    candidate_df = candidate_df.drop(columns=["1"])

  selected: List[str] = []
  current_r2 = -1.0
  n = len(y_target)

  print(
      f"{'Step':>4} | {'追加された特徴量':<20} | {'R2 Score':>8} | {'Adj R2':>8} | {'向上幅':>8}"
  )
  print("-" * 65)

  for step in range(1, 41):
    best_feat = None
    best_r2 = -1e9
    best_adj_r2 = -1e9
    for feat in candidate_df.columns:
      if feat in selected:
        continue
      trial = selected + [feat]
      model = LinearRegression()
      model.fit(candidate_df[trial], y_target)

      pred_raw = model.predict(candidate_df[trial])
      if use_logit:
        y_pred = 1 / (1 + np.exp(-pred_raw))
      else:
        y_pred = np.clip(pred_raw, 0, 1)

      r2 = r2_score(y_raw, y_pred)
      p = len(trial)
      adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

      if adj_r2 > best_adj_r2:
        best_r2 = r2
        best_adj_r2 = adj_r2
        best_feat = feat

    if best_feat:
      improvement = best_r2 - current_r2 if current_r2 != -1.0 else best_r2
      print(
          f"{step:4d} | {best_feat:<20} | {best_r2:8.4f} | {best_adj_r2:8.4f} | {improvement:+8.4f}"
      )
      selected.append(best_feat)
      current_r2 = best_r2

      if best_adj_r2 >= threshold:
        print(f"目標 Adj R2 に達したため終了します。")
        break
    else:
      break

  # 最終モデル
  final_model = LinearRegression()
  final_model.fit(candidate_df[selected], y_target)

  print(f"\n--- 最終モデルの係数 (ターゲット: {target_col}年) ---")
  print(f"切片 (Intercept): {final_model.intercept_:.6f}")
  coef_df = pd.DataFrame({"特徴量": selected, "係数": final_model.coef_})
  print(coef_df.to_string(index=False))

  # 97% / 90% の算出
  def print_formula(target_val, label):
    parts = [f"{final_model.intercept_:.6f}"]
    for f, c in zip(selected, final_model.coef_):
      parts.append(f"({c:+.6f} * {f})")
    formula = " + ".join(parts)
    print(f"\n--- {label} 生存確率の条件式 ---")
    if use_logit:
      print(f"Logit(P) = {np.log(target_val/(1-target_val)):.4f}")
      print(f"0 = {formula} - {np.log(target_val/(1-target_val)):.4f}")
    else:
      print(f"P = {target_val:.4f}")
      print(f"0 = {formula} - {target_val:.4f}")

  print_formula(0.97, "97%")
  print_formula(0.90, "90%")

  return final_model, selected, poly


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


def run_asset_spend_fitting_analysis(df: pd.DataFrame, target_col: str):
  """
  Step 8: 総資産 (M) と支出額 (Spend/Mult) のみを用いて生存確率を予測するモデルの評価。
  """
  print(f"\n\n{'='*20} 8. 資産と支出額のみを用いたモデル評価 (R2 Score) - {target_col}年 {'='*20}")

  y_target = df[target_col].to_numpy().astype(float)
  M_val = df["initial_money"].to_numpy().astype(float)
  # 支出額 (Multiplier を使用)
  Mult_val = df["spend_multiplier"].to_numpy().astype(float)
  # 実支出額 (万円)
  Spend_val = df["initial_annual_cost"].to_numpy().astype(float)

  # 対数オッズ空間
  y_clipped = np.clip(y_target, 0.001, 0.999)
  logit_y = np.log(y_clipped / (1 - y_clipped))

  # 特徴量セット (Rule S を意図的に除外)
  feats = pd.DataFrame({
      "M": M_val,
      "Mult": Mult_val,
      "Spend": Spend_val,
      "invM": 1.0 / np.maximum(M_val, 1.0),
      "invSpend": 1.0 / np.maximum(Spend_val, 1.0),
      "logM": np.log(np.maximum(M_val, 1.0)),
      "logSpend": np.log(np.maximum(Spend_val, 1.0))
  })

  results = []

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
    n = len(y_target)
    p = X_poly.shape[1] - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(
        f"{name:<40} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f} | 特徴量数: {X_poly.shape[1]}"
    )
    results.append({
        "name": name,
        "poly_deg": poly_deg,
        "interaction_only": interaction_only,
        "use_logit": use_logit,
        "adj_r2": adj_r2
    })

  evaluate("P に対する2次 (Asset/Spendのみ)", 2, True, False)
  evaluate("Logit に対する2次 (Asset/Spendのみ)", 2, True, True)
  evaluate("P に対する3次 (Asset/Spendのみ)", 3, True, False)
  evaluate("Logit に対する3次 (Asset/Spendのみ)", 3, True, True)

  return results


def run_asset_spend_stepwise_analysis(df: pd.DataFrame,
                                      target_col: str,
                                      max_adj_r2: float,
                                      poly_deg: int = 2,
                                      interaction_only: bool = True,
                                      use_logit: bool = False):
  """
  Step 9: 資産と支出額のみを用いたステップワイズ特徴量選択による近似式算出。
  """
  print(f"\n\n{'='*20} 9. 資産と支出額のみを用いたステップワイズ近似式算出 {'='*20}")
  print(f"ターゲット: {target_col}年, Logit使用: {use_logit}, Degree: {poly_deg}")
  threshold = max_adj_r2 * 0.97
  print(f"目標 Adj R2 (97% of {max_adj_r2:.4f}): {threshold:.4f}")

  y_raw = df[target_col].to_numpy().astype(float)
  if use_logit:
    y_clipped = np.clip(y_raw, 0.0001, 0.9999)
    y_target = np.log(y_clipped / (1 - y_clipped))
  else:
    y_target = y_raw

  M_val = df["initial_money"].to_numpy().astype(float)
  Mult_val = df["spend_multiplier"].to_numpy().astype(float)
  Spend_val = df["initial_annual_cost"].to_numpy().astype(float)

  base_feats = pd.DataFrame({
      "M": M_val,
      "Mult": Mult_val,
      "Spend": Spend_val,
      "invM": 1.0 / np.maximum(M_val, 1.0),
      "invSpend": 1.0 / np.maximum(Spend_val, 1.0),
      "logM": np.log(np.maximum(M_val, 1.0)),
      "logSpend": np.log(np.maximum(Spend_val, 1.0))
  })

  poly = PolynomialFeatures(degree=poly_deg, interaction_only=interaction_only)
  X_all = poly.fit_transform(base_feats)
  feature_names = poly.get_feature_names_out(base_feats.columns)
  candidate_df = pd.DataFrame(X_all, columns=feature_names)
  if "1" in candidate_df.columns: candidate_df = candidate_df.drop(columns=["1"])

  selected: List[str] = []
  current_r2 = -1.0
  n = len(y_target)

  print(f"{'Step':>4} | {'追加された特徴量':<20} | {'R2 Score':>8} | {'Adj R2':>8} | {'向上幅':>8}")
  print("-" * 65)

  for step in range(1, 41):
    best_feat = None
    best_r2 = -1e9
    best_adj_r2 = -1e9
    for feat in candidate_df.columns:
      if feat in selected: continue
      trial = selected + [feat]
      model = LinearRegression().fit(candidate_df[trial], y_target)
      pred_raw = model.predict(candidate_df[trial])
      y_pred = 1/(1+np.exp(-pred_raw)) if use_logit else np.clip(pred_raw, 0, 1)
      r2 = r2_score(y_raw, y_pred)
      adj_r2 = 1 - (1 - r2) * (n - 1) / (n - len(trial) - 1)
      if adj_r2 > best_adj_r2:
        best_r2, best_adj_r2, best_feat = r2, adj_r2, feat

    if best_feat:
      improvement = best_r2 - current_r2 if current_r2 != -1.0 else best_r2
      print(f"{step:4d} | {best_feat:<20} | {best_r2:8.4f} | {best_adj_r2:8.4f} | {improvement:+8.4f}")
      selected.append(best_feat)
      current_r2 = best_r2
      if best_adj_r2 >= threshold:
        print(f"目標 Adj R2 に達したため終了します。")
        break
    else: break

  final_model = LinearRegression().fit(candidate_df[selected], y_target)
  print(f"\n--- 最終モデルの係数 (資産/支出のみ) ---")
  print(f"切片 (Intercept): {final_model.intercept_:.6f}")
  coef_df = pd.DataFrame({"特徴量": selected, "係数": final_model.coef_})
  print(coef_df.to_string(index=False))


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
  # run_best_combination_analysis(df_p_d_survival)

  # 2. 支出額パーセンタイル推移の生成
  # run_percentile_analysis(df_p_d_all)

  # 3. df_p60_d1_survival からヒートマップを作成
  # run_p60_d1_heatmap(df_p60_d1_survival)

  # 4. 予測モデルの評価
  # fitting_results = run_fitting_analysis(df_p60_d1_survival, "45")

  # # 5. ステップワイズ特徴量選択による生存確率の近似式算出
  # # fitting_results の中から最も Adj R2 が高い Logit 手法を選択する (境界線の算出には Logit が適しているため)
  # logit_results = [r for r in fitting_results if r["use_logit"]]
  # best_eval = max(logit_results, key=lambda x: x["adj_r2"])

  # model_sw, selected_sw, poly_sw = run_stepwise_fitting_analysis(
  #     df_p60_d1_survival,
  #     "45",
  #     max_adj_r2=float(best_eval["adj_r2"]),
  #     poly_deg=int(best_eval["poly_deg"]),
  #     interaction_only=bool(best_eval["interaction_only"]),
  #     use_logit=True)

  # # 6. 生存達成データの生成 (97, 95, 90, 80, 70%)
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  # df_plot, base_cost = run_survival_curve_analysis(df_p60_d1_survival,
  #                                                  model_sw,
  #                                                  selected_sw,
  #                                                  poly_sw,
  #                                                  use_logit=True,
  #                                                  target_probs=target_probs)

  # 7. 3つのグラフを保存
  # save_survival_charts(df_plot, base_cost, target_probs)

  # 8. 資産と支出額のみを用いたモデル評価
  asset_spend_results = run_asset_spend_fitting_analysis(df_p60_d1_survival, "45")

  # 9. 資産と支出額のみを用いたステップワイズ近似式算出
  # 最も高い Adj R2 を持つ Logit モデルを選択
  logit_as_results = [r for r in asset_spend_results if r["use_logit"]]
  best_as_eval = max(logit_as_results, key=lambda x: x["adj_r2"])

  run_asset_spend_stepwise_analysis(
      df_p60_d1_survival,
      "45",
      max_adj_r2=float(best_as_eval["adj_r2"]),
      poly_deg=int(best_as_eval["poly_deg"]),
      interaction_only=bool(best_as_eval["interaction_only"]),
      use_logit=True)

  # 10. 初期支出率を求める公式 (Rule of Thumb) の出力
  run_rule_of_thumb_analysis(df_p60_d1_survival, target_probs)


if __name__ == "__main__":
  main()
