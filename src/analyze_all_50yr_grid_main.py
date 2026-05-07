"""
data/all_50yr/ の結果を分析・可視化するスクリプト。

内容:
1. 最適な受給開始年齢の分析
2. 支出額パーセンタイル推移の生成
3. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd

from src.lib.visualize import create_survival_probability_chart
from src.lib.visualize_all_yr import (create_heatmap,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels)

# 設定
IMG_DIR = "docs/imgs/all_50yr"
TEMP_DIR = "temp/all_50yr"
# BASE_SPEND_ANNUAL (574.0万円) = 統計データの50歳時平均支出 (552.5万円) + 国民年金保険料 (21.5万円)
# シミュレーションでは、国民年金保険料は固定額、生活費のみを倍率 (spend_mult) でスケーリングしている。
BASE_SPEND_ANNUAL = 574.0
NUM_YEARS = 45
START_AGE = 50


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
    within_threshold = group[group[target_year] >=
                             (max_prob - threshold)][value_col].tolist()
    for val in within_threshold:
      if pd.isna(val):
        continue
      counts[val] = counts.get(val, 0) + 1

  # 頻度が高い順にソート。頻度が同じなら値自体でソートして安定させる
  sorted_items = sorted(counts.items(),
                        key=lambda x: (x[1], str(x[0])),
                        reverse=True)
  return [item[0] for item in sorted_items]


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
                                               start_age=START_AGE,
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

  # 1. グラフ作成
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

  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    print("Error: Survival data not found.")
    return

  dim_cols = ['spend_multiplier', 'spending_rule']
  threshold = 0.01  # 許容範囲 1%

  # 優先順位を自動計算 (閾値内の出現頻度順)
  pref_order = calculate_preference_order(df_survival, target_year, threshold,
                                          dim_cols, "pension_start_age")
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
    within_threshold_rows = sorted_group[sorted_group[target_year] >=
                                         (max_prob - threshold)]
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

    # 4. ラベル作成
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

  title = f"最適年金受給開始年齢 ({target_year}年後生存確率, 許容差{threshold*100:g}%)"
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


def run_percentile_analysis(df_all: pd.DataFrame):
  """
  支出額パーセンタイル推移の生成。
  """
  print(f"\n\n{'='*20} 支出額パーセンタイル推移グラフを生成中... {'='*20}")

  # 代表的なケースを選択
  cases = [
      (1.0, 4.0, 65),
      (1.0, 5.0, 75),
  ]

  for s_mult, rule, p_age in cases:
    mask = (df_all["pension_start_age"] == p_age) & \
           (df_all["spend_multiplier"] == s_mult) & \
           (df_all["spending_rule"] == rule)

    df_plot = df_all[mask]
    if df_plot.empty:
      continue

    init_cost = df_plot["initial_annual_cost"].iloc[0]
    title = f"年間支出額推移: 年金{p_age}歳, 初期{int(round(init_cost))}万円/年, 初期支出率{rule:g}%"
    output_name = f"spend_percentiles_p{p_age}_m{s_mult:g}_r{rule:g}.svg"
    output_path = os.path.join(IMG_DIR, output_name)

    create_spend_percentile_chart(df_plot,
                                  title,
                                  output_path,
                                  start_age=START_AGE,
                                  num_years=NUM_YEARS)


def main():
  parser = argparse.ArgumentParser(
      description="50歳リタイア開始・95歳までの分析・可視化スクリプト。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="optimal-pension",
                      help="実験設定 (optimal-pension)")
  args = parser.parse_args()

  exp_types = args.exp_type.split(",")
  target_year = str(NUM_YEARS)

  for et in exp_types:
    et = et.strip()
    csv_path = f"data/all_50yr/{et}.csv"
    if not os.path.exists(csv_path):
      print(f"Warning: {csv_path} が見つかりません。スキップします。")
      continue

    print(f"\nProcessing experiment type: {et}")
    df_all = pd.read_csv(csv_path)

    if et == "optimal-pension":
      run_optimal_pension_analysis(df_all, target_year)
      run_percentile_analysis(df_all)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
