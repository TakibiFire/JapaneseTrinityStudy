"""
data/all_50yr/ の結果を分析・可視化するスクリプト。

内容:
1. 最適な受給開始年齢の分析
2. 支出額パーセンタイル推移の生成
3. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
4. pen70-lifeplan 分析 (リバランス戦略の比較)
"""

import argparse
import os

import pandas as pd

from src.lib.visualize_all_yr import (calculate_preference_order,
                                      create_best_strategy_heatmap,
                                      create_improvement_heatmap,
                                      create_optimal_pension_heatmap,
                                      create_pension_survival_curve,
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
    within_threshold_rows = sorted_group[sorted_group[target_year] >= (
        max_prob - threshold)]
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
  pref_order = calculate_preference_order(df_survival, target_year, threshold,
                                          dim_cols, "strategy_short")
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
    within_threshold_rows = sorted_group[sorted_group[target_year] >= (
        max_prob - threshold)]
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
    label = f"{max_prob*100:.1f}%"

    line2 = within_threshold_names[0]
    if len(within_threshold_names) >= 2:
      line2 += f", {within_threshold_names[1]}"
    label += f"\n{line2}"

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
      "R70": "#B2F5EA",  # Light teal
      "V1": "#FBD38D",  # Light orange
      "固定": "#FEB2B2",  # Light red
      "なし": "#CBD5E0"  # Light gray
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
                               y_sort=m_order,
                               height=405)

  # 改善幅の計算 (R70 vs V1)
  df_r70 = df_survival[df_survival["strategy_short"] == "R70"].copy()
  df_v1 = df_survival[df_survival["strategy_short"] == "V1"].copy()

  if not df_r70.empty and not df_v1.empty:
    df_imp = pd.merge(df_r70[[
        'spend_multiplier', 'spending_rule', 'initial_annual_cost', target_year
    ]],
                      df_v1[['spend_multiplier', 'spending_rule', target_year]],
                      on=['spend_multiplier', 'spending_rule'],
                      suffixes=('_r70', '_v1'))
    df_imp["improvement"] = df_imp[f"{target_year}_r70"] - df_imp[
        f"{target_year}_v1"]

    df_imp, m_order_imp, r_order_imp = prepare_heatmap_labels(df_imp)

    title_imp = f"R70のV1に対する改善幅 ({target_year}年後生存確率 差分)"
    output_path_imp = os.path.join(IMG_DIR, "improvement_r70_vs_v1_heatmap.svg")

    create_improvement_heatmap(df_imp,
                               target_col="improvement",
                               title=title_imp,
                               x_col="rule_label",
                               x_title="初期支出率 (%ルール)",
                               y_col="multiplier_label",
                               y_title="支出レベル",
                               output_path=output_path_imp,
                               x_sort=r_order_imp,
                               y_sort=m_order_imp,
                               height=405)


def main():
  parser = argparse.ArgumentParser(description="50歳リタイア開始・95歳までの分析・可視化スクリプト。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="optimal-pension",
                      help="実験設定 (optimal-pension, pen70-lifeplan)")
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
    elif et == "pen70-lifeplan":
      run_pen70_lifeplan_analysis(df_all, target_year)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
