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
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd

from src.lib.fitting_all_yr import (FeatureSetType, run_fitting_analysis,
                                    run_rule_of_thumb_analysis,
                                    run_stepwise_fitting_analysis,
                                    run_survival_curve_analysis,
                                    save_survival_charts)
from src.lib.visualize import create_survival_probability_chart
from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels,
                                      run_best_combination_analysis)


def calculate_preference_order(df_survival: pd.DataFrame,
                               target_year: str,
                               threshold: float,
                               dim_cols: List[str],
                               value_col: str) -> List[Any]:
  """
  全グリッドセルにおける出現頻度に基づいて優先順位を自動計算する。
  """
  counts: Dict[Any, int] = {}

  for _, group in df_survival.groupby(dim_cols):
    max_prob = float(group[target_year].max())
    within_threshold = group[group[target_year] >= (max_prob - threshold)][value_col].tolist()
    for val in within_threshold:
      if pd.isna(val):
        continue
      counts[val] = counts.get(val, 0) + 1

  # 頻度が高い順にソート。頻度が同じなら値自体でソートして安定させる
  sorted_items = sorted(counts.items(), key=lambda x: (x[1], str(x[0])), reverse=True)
  return [item[0] for item in sorted_items]


# 設定
IMG_DIR = "docs/imgs/all_60yr"
TEMP_DIR = "temp/all_60yr"
BASE_SPEND_ANNUAL = 540.0
NUM_YEARS = 35


def create_best_strategy_heatmap(df_best: pd.DataFrame,
                                title: str,
                                x_col: str,
                                x_title: str,
                                y_col: str,
                                y_title: str,
                                output_path: str,
                                color_col: str,
                                color_title: str,
                                color_map: Dict[str, str],
                                x_sort: Optional[List] = None,
                                y_sort: Optional[List] = None,
                                width: int = 500,
                                height: int = 450):
  """
  選択された戦略を可視化するヒートマップ。
  """
  plot_df = df_best.copy()
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
      color=alt.Color(f'{color_col}:N',
                      title=color_title,
                      scale=alt.Scale(domain=domain, range=range_)))

  text = base.mark_text(baseline='middle',
                        lineBreak='\n').encode(text=alt.Text('combo_label:N'),
                                               color=alt.value('black'))

  chart = (heatmap + text).properties(title=title, width=width, height=height)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def create_optimal_pension_heatmap(df_best: pd.DataFrame,
                                   title: str,
                                   x_col: str,
                                   x_title: str,
                                   y_col: str,
                                   y_title: str,
                                   output_path: str,
                                   x_sort: Optional[List] = None,
                                   y_sort: Optional[List] = None,
                                   width: int = 500,
                                   height: int = 450):
  """
  最適な年金受給開始年齢を可視化するヒートマップ。
  """
  color_map = {
      "60歳": "#FBD38D",  # Light orange
      "65歳": "#9AE6B4",  # Light green
      "70歳": "#B2F5EA",  # Light teal
      "75歳": "#FEB2B2"  # Light red
  }
  return create_best_strategy_heatmap(df_best, title, x_col, x_title, y_col,
                                      y_title, output_path, "display_age",
                                      "受給開始年齢", color_map, x_sort, y_sort,
                                      width, height)


def create_pension_survival_curve(df: pd.DataFrame,
                                 multiplier: float,
                                 rule: float,
                                 title: str,
                                 output_path: str):
  """
  指定された multiplier と rule における、受給開始年齢別の生存確率推移を描画する。
  """
  # 年度列 (1, 2, ..., NUM_YEARS) を取得
  year_cols = [str(i) for i in range(1, NUM_YEARS + 1) if str(i) in df.columns]

  # 指定された条件でフィルタ
  plot_df = df[(df["spend_multiplier"] == multiplier) &
               (df["spending_rule"] == rule) &
               (df["value_type"] == "survival")].copy()

  if plot_df.empty:
    print(f"Warning: No data for multiplier={multiplier}, rule={rule}")
    return

  # メルトしてロング形式に
  df_long = plot_df.melt(id_vars=["pension_start_age"],
                         value_vars=year_cols,
                         var_name="Year",
                         value_name="Survival Probability (%)")
  df_long["Year"] = df_long["Year"].astype(int)
  df_long["Survival Probability (%)"] *= 100.0

  # 0年目のデータを追加 (開始時は100%)
  ages = plot_df["pension_start_age"].unique()
  start_rows = pd.DataFrame({
      "pension_start_age": ages,
      "Year": 0,
      "Survival Probability (%)": 100.0
  })
  df_long = pd.concat([start_rows, df_long], ignore_index=True)

  df_long["Strategy"] = df_long["pension_start_age"].map(
      lambda x: f"{int(x)}歳受給開始")

  # 共通ライブラリの可視化関数を使用
  _, chart = create_survival_probability_chart(df_plot=df_long,
                                               start_age=60,
                                               height=300)

  chart = chart.properties(title=title)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def run_optimal_pension_analysis(df_all: pd.DataFrame, target_year: str):
  """
  最適な年金受給開始年齢を分析する。
  """
  print(f"\n\n{'='*20} 最適な年金受給開始年齢の分析 {'='*20}")

  # 1. グラフ作成 (m=1, r=4% と m=1, r=5%)
  create_pension_survival_curve(
      df_all,
      multiplier=1.0,
      rule=4.0,
      title="受給開始年齢別 生存確率推移 (支出レベル1.0, 初年度支出率4%)",
      output_path=os.path.join(IMG_DIR, "survival_curve_pension_m1_r4.svg"))

  create_pension_survival_curve(
      df_all,
      multiplier=1.0,
      rule=5.0,
      title="受給開始年齢別 生存確率推移 (支出レベル1.0, 初年度支出率5%)",
      output_path=os.path.join(IMG_DIR, "survival_curve_pension_m1_r5.svg"))

  create_pension_survival_curve(
      df_all,
      multiplier=3.0,
      rule=5.0,
      title="受給開始年齢別 生存確率推移 (支出レベル2.0, 初年度支出率5%)",
      output_path=os.path.join(IMG_DIR, "survival_curve_pension_m3_r5.svg"))

  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    print("Error: Survival data not found.")
    return

  dim_cols = ['spend_multiplier', 'spending_rule']
  threshold = 0.01  # 許容範囲 1%

  # 優先順位を自動計算 (閾値内の出現頻度順)
  pref_order = calculate_preference_order(df_survival, target_year, threshold, dim_cols, "pension_start_age")
  print(f"Computed preference order for pension ages: {pref_order}")

  def get_best_age(group: pd.DataFrame) -> pd.Series:
    max_prob = float(group[target_year].max())

    # 0. 優先順位を数値化 (値が小さいほど高優先)
    pref_map = {age: i for i, age in enumerate(pref_order)}
    temp_group = group.copy()
    temp_group["pref_score"] = temp_group["pension_start_age"].map(pref_map)

    # 1. 生存確率の降順、同じなら優先順位の昇順でソート
    sorted_group = temp_group.sort_values(by=[target_year, "pref_score"],
                                          ascending=[False, True])

    # 2. 閾値内の全年齢を取得
    within_threshold_rows = sorted_group[sorted_group[target_year] >= (max_prob - threshold)]
    within_threshold_ages = within_threshold_rows["pension_start_age"].tolist()

    # 3. 色決定用の代表年齢 (優先順位に従う)
    selected_row = None
    for age in pref_order:
      if age in within_threshold_ages:
        selected_row = group[group["pension_start_age"] == age].iloc[0].copy()
        break

    if selected_row is None:
      selected_row = within_threshold_rows.iloc[0].copy()

    selected_row["display_age"] = f"{int(selected_row['pension_start_age'])}歳"

    # 4. ラベル作成 (1行目: 生存率, 2行目: 1つ目, 2つ目, 3行目: 3つ目, 4つ目)
    label = f"{max_prob*100:.1f}%"

    line2 = f"{int(within_threshold_ages[0])}歳"
    if len(within_threshold_ages) >= 2:
      line2 += f", {int(within_threshold_ages[1])}歳"
    label += f"\n{line2}"

    if len(within_threshold_ages) >= 3:
      line3 = f"{int(within_threshold_ages[2])}歳"
      if len(within_threshold_ages) >= 4:
        line3 += f", {int(within_threshold_ages[3])}歳"
      label += f"\n{line3}"

    selected_row["combo_label"] = label
    return selected_row

  results = []
  for _, group in df_survival.groupby(dim_cols):
    results.append(get_best_age(group))
  df_best = pd.DataFrame(results)

  df_best, m_order, r_order = prepare_heatmap_labels(df_best)

  title = f"最適年金受給開始年齢 ({target_year}年後生存確率, 優先: {'>'.join([f'{int(a)}歳' for a in pref_order])}, 許容差{threshold*100:g}%)"
  output_path = os.path.join(IMG_DIR, "optimal_pension_age_heatmap.svg")
  create_optimal_pension_heatmap(df_best,
                                 title=title,
                                 x_col="rule_label",
                                 x_title="初期支出率 (%ルール)",
                                 y_col="multiplier_label",
                                 y_title="支出レベル",
                                 output_path=output_path,
                                 x_sort=r_order,
                                 y_sort=m_order)


def run_p_d_range_analysis(df_all: pd.DataFrame, target_year: str):
  """
  P-D-RANGE の分析を実行する。
  """
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  run_best_combination_analysis(
      df_survival,
      target_year=target_year,
      img_dir=IMG_DIR,
      temp_dir=TEMP_DIR,
      title_prefix="60歳リタイア",
      threshold=0.02,
      pref_order=["P60_D1", "P65_D1", "P60_D0", "P65_D0"],
      width=500,
      height=450)


def run_p60_d1_analysis(df_all: pd.DataFrame, target_year: str):
  """
  P60-D1 の詳細分析を実行する。
  """
  df_survival = df_all[df_all["value_type"] == "survival"].copy()

  # 1. ヒートマップ
  run_p60_d1_heatmap(df_survival)

  # 2. 予測モデルの評価
  fitting_results = run_fitting_analysis(df_survival, target_year)

  # 3. ステップワイズ特徴量選択
  logit_results = [r for r in fitting_results if r["use_logit"]]
  best_eval = max(logit_results, key=lambda x: x["adj_r2"])

  model_sw, selected_sw, poly_sw = run_stepwise_fitting_analysis(
      df_survival,
      target_year,
      max_adj_r2=float(best_eval["adj_r2"]),
      poly_deg=int(best_eval["poly_deg"]),
      interaction_only=bool(best_eval["interaction_only"]),
      use_logit=True)

  # 4. 生存達成データの生成
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  df_plot_survival, base_cost = run_survival_curve_analysis(
      df_survival,
      model_sw,
      selected_sw,
      poly_sw,
      use_logit=True,
      target_probs=target_probs)

  # 5. グラフ保存
  save_survival_charts(df_plot_survival,
                       base_cost,
                       target_probs,
                       img_dir=IMG_DIR)

  # 6. Rule of Thumb
  run_rule_of_thumb_analysis(df_survival, target_year, target_probs)


def run_p60_d1_heatmap(df_survival: pd.DataFrame):
  """
  P60, D1 のヒートマップを作成する。
  """
  print(f"\n\n{'='*20} P60, D1 ヒートマップ生成 {'='*20}")

  if df_survival.empty:
    return

  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)

  year_target = str(NUM_YEARS)
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


def run_pen70_lifeplan_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-lifeplan の分析を実行する。
  """
  print(f"\n\n{'='*20} pen70-lifeplan 分析 {'='*20}")

  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    print("Error: Survival data not found.")
    return

  # 戦略の略称マッピング
  strategy_map = {
      "SpendAwareDPRebalance (R70-aware)": "R70",
      "DynamicV1Rebalance": "V1",
      "固定最適比率": "固定",
      "No dynamic rebalance": "なし"
  }

  df_survival["strategy_short"] = df_survival["strategy"].map(strategy_map)

  dim_cols = ['spend_multiplier', 'spending_rule']
  threshold = 0.01  # 1%

  # 優先順位を自動計算 (閾値内の出現頻度順)
  pref_order = calculate_preference_order(df_survival, target_year, threshold, dim_cols, "strategy_short")
  print(f"Computed preference order for strategies: {pref_order}")

  def get_best_strategy(group: pd.DataFrame) -> pd.Series:
    max_prob = float(group[target_year].max())

    # 0. 優先順位を数値化 (値が小さいほど高優先)
    pref_map = {name: i for i, name in enumerate(pref_order)}
    temp_group = group.copy()
    temp_group["pref_score"] = temp_group["strategy_short"].map(pref_map)

    # 1. 生存確率の降順、同じなら優先順位の昇順でソート
    sorted_group = temp_group.sort_values(by=[target_year, "pref_score"],
                                          ascending=[False, True])

    # 2. 閾値内の全戦略を取得
    within_threshold_rows = sorted_group[sorted_group[target_year] >= (max_prob - threshold)]
    within_threshold_names = within_threshold_rows["strategy_short"].tolist()

    # 3. 色決定用の代表戦略 (優先順位に従う)
    selected_row = None
    for strat in pref_order:
      if strat in within_threshold_names:
        selected_row = group[group["strategy_short"] == strat].iloc[0].copy()
        break

    if selected_row is None:
      selected_row = within_threshold_rows.iloc[0].copy()

    selected_row["display_strategy"] = selected_row["strategy_short"]

    # 4. ラベル作成
    # 1行目: 生存率 (最大値)
    label = f"{max_prob*100:.1f}%"

    # 2行目: 1つ目 (, 2つ目) -> ソート順 (同率なら優先順)
    line2 = within_threshold_names[0]
    if len(within_threshold_names) >= 2:
      line2 += f", {within_threshold_names[1]}"
    label += f"\n{line2}"

    # 3行目: (3つ目) (, 4つ目) -> ソート順
    if len(within_threshold_names) >= 3:
      line3 = within_threshold_names[2]
      if len(within_threshold_names) >= 4:
        line3 += f", {within_threshold_names[3]}"
      label += f"\n{line3}"

    selected_row["combo_label"] = label
    return selected_row

  results = []
  for _, group in df_survival.groupby(dim_cols):
    results.append(get_best_strategy(group))
  df_best = pd.DataFrame(results)

  df_best, m_order, r_order = prepare_heatmap_labels(df_best)

  # 戦略ごとのカラーマップ
  color_map = {
      "R70": "#B2F5EA",    # Light teal
      "V1": "#FBD38D",      # Light orange
      "固定": "#FEB2B2",    # Light red
      "なし": "#CBD5E0"      # Light gray
  }

  title = f"最適リバランス戦略 ({target_year}年後生存確率, 優先: {'>'.join(pref_order)}, 許容差{threshold*100:g}%)"
  output_path = os.path.join(IMG_DIR, "best_rebalance_strategy_heatmap.svg")

  create_best_strategy_heatmap(df_best,
                               title=title,
                               x_col="rule_label",
                               x_title="初期支出率 (%ルール)",
                               y_col="multiplier_label",
                               y_title="支出レベル",
                               output_path=output_path,
                               color_col="display_strategy",
                               color_title="最適戦略",
                               color_map=color_map,
                               x_sort=r_order,
                               y_sort=m_order)


def main():
  parser = argparse.ArgumentParser(description="60歳リタイア開始・95歳までの分析・可視化スクリプト。")
  parser.add_argument(
      "--exp_type",
      type=str,
      default="optimal-pension",
      help=
      "実験設定 (comma separated: optimal-pension, P-D-RANGE, P60-D1, pen70-lifeplan)"
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
    elif et == "P-D-RANGE":
      run_best_combination_analysis(
          df_all[df_all["value_type"] == "survival"].copy(),
          target_year=target_year,
          img_dir=IMG_DIR,
          temp_dir=TEMP_DIR,
          title_prefix="60歳リタイア",
          threshold=0.02,
          pref_order=["P60_D1", "P65_D1", "P60_D0", "P65_D0"],
          width=500,
          height=450)
    elif et == "P60-D1":
      run_p60_d1_analysis(df_all, target_year)
    elif et == "pen70-lifeplan":
      run_pen70_lifeplan_analysis(df_all, target_year)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
