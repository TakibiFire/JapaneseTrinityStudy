"""
data/all_40yr/ の結果を分析・可視化するスクリプト。

内容:
1. 最適な受給開始年齢の分析
2. 支出額パーセンタイル推移の生成
3. 2次元ヒートマップによる可視化 (支出レベル vs 支出率)
4. pen70-lifeplan 分析 (リバランス戦略の比較)
"""

import argparse
import json
import os
import shutil

import pandas as pd

from src.lib.survival_contours import (generate_rule_of_thumb,
                                       generate_smooth_contour_data,
                                       get_contour_anchor_points,
                                       save_contour_charts)
from src.lib.survival_formula_analysis import run_survival_formula_analysis
from src.lib.visualize_all_yr import (calculate_preference_order,
                                      create_best_strategy_heatmap,
                                      create_heatmap,
                                      create_improvement_heatmap,
                                      create_optimal_pension_heatmap,
                                      create_pension_survival_curve,
                                      create_spend_percentile_chart,
                                      prepare_heatmap_labels)

# 設定
IMG_DIR = "docs/imgs/all_40yr"
DATA_OUT_DIR = "docs/data/all_40yr"
TEMP_DIR = "temp/all_40yr"
NUM_YEARS = 55
START_AGE = 40


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
                                rule=2.5,
                                title="受給開始年齢別 生存確率推移 (支出レベル1.0, 初年度支出率2.5%)",
                                output_path=os.path.join(
                                    IMG_DIR,
                                    "survival_curve_pension_m1_r2_5.svg"),
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

  # NEW: ドキュメント作成用のデータ出力
  print("\n--- 最適年金受給開始年齢の生存確率 (ドキュメント用) ---")
  doc_cases = [
      (1.0, 4.0),
      (1.0, 5.0),
  ]
  for m, r in doc_cases:
    case_data = df_survival[(df_survival["spend_multiplier"] == m) &
                            (df_survival["spending_rule"] == r)]
    if not case_data.empty:
      print(f"\n支出レベル: {m}, 初期支出率: {r}%")
      print(case_data[["pension_start_age", target_year]])

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
                                 y_sort=m_order,
                                 all_ages=[f"{int(a)}歳" for a in pref_order])


def run_percentile_analysis(df_all: pd.DataFrame):
  """
  支出額パーセンタイル推移の生成。
  """
  print(f"\n\n{'='*20} 支出額パーセンタイル推移グラフを生成中... {'='*20}")

  # 代表的なケースを選択
  cases = [
      (1.0, 4.0, 70),
      (1.0, 5.0, 70),
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

    print_comparison_summary(df_r70, df_v1, target_year)


def print_comparison_summary(df_r70: pd.DataFrame, df_v1: pd.DataFrame,
                             target_year: str):
  """
  R70とV1の生存確率の比較サマリーを表示する。
  """
  print("\n--- R70 vs V1 生存確率の比較サマリー ---")
  df_pivot_r70 = df_r70.pivot(index='spend_multiplier',
                              columns='spending_rule',
                              values=target_year)
  df_pivot_v1 = df_v1.pivot(index='spend_multiplier',
                            columns='spending_rule',
                            values=target_year)

  diff = (df_pivot_r70 - df_pivot_v1) * 100
  print("\n改善幅 (R70 - V1) [パーセンテージポイント]:")
  print(diff)

  print("\nR70 生存確率 (%):")
  print(df_pivot_r70 * 100)

  print("\nV1 生存確率 (%):")
  print(df_pivot_v1 * 100)


def run_pen70_formula_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-formula の詳細分析を実行する。
  """
  print(f"\n\n{'='*20} pen70-formula 分析 {'='*20}")
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    return

  # 1. ヒートマップ
  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)
  title = f"40歳リタイア・年金70歳・{target_year}年後生存確率(%) (R70-aware)"
  output_path = os.path.join(IMG_DIR, "pen70_formula_heatmap.svg")

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

  # 2. 生存達成データの生成
  target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]
  plot_data = []
  for p in target_probs:
    anchors = get_contour_anchor_points(df_survival, p, target_year)
    plot_data.extend(generate_smooth_contour_data(anchors, f"{p*100:g}%"))
  df_plot_survival = pd.DataFrame(plot_data)

  # 3. グラフ保存
  save_contour_charts(df_plot_survival,
                      target_probs,
                      img_dir=IMG_DIR,
                      prefix="pen70_formula_")

  # 4. Rule of Thumb
  generate_rule_of_thumb(df_survival, target_probs, target_year)

  # 5. 詳細な近似モデルの分析
  coeffs = run_survival_formula_analysis(df_survival, target_year)

  # 6. JSON出力
  if coeffs:
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    out_json = {
        "start_age": START_AGE,
        "pension_start": 70,
        "target_age": START_AGE + int(target_year),
        "formula": coeffs
    }
    json_path = os.path.join(DATA_OUT_DIR, "formula.json")
    with open(json_path, "w") as f:
      json.dump(out_json, f, indent=2)
    print(f"✅ {json_path} を保存しました。")


def run_pen70_ds_analysis(df_all: pd.DataFrame, target_year: str):
  """
  pen70-ds の詳細分析を実行する。
  """
  print(f"\n\n{'='*20} pen70-ds 分析 {'='*20}")
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    return

  # 1. ヒートマップ
  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)
  title = f"40歳リタイア・年金70歳・{target_year}年後生存確率(%) (R70 + SpendAwareDS)"
  output_path = os.path.join(IMG_DIR, "pen70_ds_heatmap.svg")

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

  # 2. pen70-formula との比較ヒートマップ
  formula_path = "data/all_40yr/pen70-formula.csv"
  if os.path.exists(formula_path):
    df_formula = pd.read_csv(formula_path)
    df_f_surv = df_formula[df_formula["value_type"] == "survival"].copy()

    df_comp = pd.merge(
        df_survival[[
            'spend_multiplier', 'spending_rule', 'initial_annual_cost',
            target_year
        ]],
        df_f_surv[['spend_multiplier', 'spending_rule', target_year]],
        on=['spend_multiplier', 'spending_rule'],
        suffixes=('_ds', '_formula'))

    df_comp["improvement"] = df_comp[f"{target_year}_ds"] - df_comp[
        f"{target_year}_formula"]

    df_comp, m_order_comp, r_order_comp = prepare_heatmap_labels(df_comp)

    title_imp = f"SpendAwareDSによる改善幅 ({target_year}年後生存確率 差分)"
    output_path_imp = os.path.join(IMG_DIR,
                                   "improvement_ds_vs_formula_heatmap.svg")

    create_improvement_heatmap(df_comp,
                               target_col="improvement",
                               title=title_imp,
                               x_col="rule_label",
                               x_title="初期支出率 (%ルール)",
                               y_col="multiplier_label",
                               y_title="支出レベル",
                               output_path=output_path_imp,
                               x_sort=r_order_comp,
                               y_sort=m_order_comp,
                               height=405)

    # 3. 生存確率曲線の比較 (3つ)
    from src.lib.visualize import create_survival_probability_chart

    def create_comp_curve(m, r, filename):
      ds_row = df_survival[(df_survival["spend_multiplier"] == m) &
                           (df_survival["spending_rule"] == r)]
      fo_row = df_f_surv[(df_f_surv["spend_multiplier"] == m) &
                         (df_f_surv["spending_rule"] == r)]

      if ds_row.empty or fo_row.empty:
        return

      year_cols = [str(i) for i in range(1, NUM_YEARS + 1)]

      # DS
      ds_vals = ds_row[year_cols].values[0]
      # Formula
      fo_vals = fo_row[year_cols].values[0]

      data = []
      # Year 0
      data.append({
          "Year": 0,
          "Survival Probability (%)": 100.0,
          "Strategy": "pen70-ds"
      })
      data.append({
          "Year": 0,
          "Survival Probability (%)": 100.0,
          "Strategy": "pen70-formula"
      })

      for i, yr in enumerate(year_cols):
        data.append({
            "Year": int(yr),
            "Survival Probability (%)": ds_vals[i] * 100,
            "Strategy": "pen70-ds"
        })
        data.append({
            "Year": int(yr),
            "Survival Probability (%)": fo_vals[i] * 100,
            "Strategy": "pen70-formula"
        })

      df_plot = pd.DataFrame(data)
      _, chart = create_survival_probability_chart(df_plot=df_plot,
                                                   start_age=START_AGE,
                                                   height=300)
      chart = chart.properties(title=f"DS vs Formula 生存確率 (m={m}, r={r}%)")
      chart.save(os.path.join(IMG_DIR, filename))
      print(f"✅ {filename} に保存しました。")

    create_comp_curve(1.0, 4.0, "comp_ds_formula_m1_r4.svg")
    create_comp_curve(1.0, 3.0, "comp_ds_formula_m1_r3.svg")
    create_comp_curve(3.0, 3.0, "comp_ds_formula_m3_r3.svg")


def generate_dp_calc_json(df_all: pd.DataFrame):
  """
  生存確率計算機（DP版）のための設定JSONを生成する。
  """
  print(f"\n\n{'='*20} DP計算機用JSONの生成 {'='*20}")

  # 支出倍率ごとの代表的な初期支出額を抽出する
  df_survival = df_all[df_all["value_type"] == "survival"].copy()

  # ユニークな倍率を取得してソート
  multipliers = sorted(df_survival["spend_multiplier"].unique())
  base_spends = {}
  models = {}

  # モデルのコピー元ディレクトリ
  model_src_dir = "data/optimal_strategy_dp"

  for m in multipliers:
    # 0.75 -> 0_75, 1.2 -> 1_2
    m_val = float(m)
    m_key = str(m_val).replace(".", "_") if m_val % 1 != 0 else str(int(m_val))
    if m_key.endswith("_0"):
      m_key = m_key[:-2]

    # モデルファイル名 (re40_pen70_95_m0_75.json など)
    model_name = f"re40_pen70_95_m{m_key}.json"
    src_path = os.path.join(model_src_dir, model_name)
    dst_path = os.path.join(DATA_OUT_DIR, model_name)

    # コピー元が存在する場合のみコピーして登録
    if os.path.exists(src_path):
      shutil.copy(src_path, dst_path)
      print(f"Copied {src_path} -> {dst_path}")

      models[str(m_val)] = model_name

      # 初期支出額の取得
      m_rows = df_survival[df_survival["spend_multiplier"] == m]
      base_spends[str(m_val)] = round(
          float(m_rows.iloc[0]["initial_annual_cost"]))
    else:
      # コピー先が既に存在する場合も登録
      if os.path.exists(dst_path):
        models[str(m_val)] = model_name
        m_rows = df_survival[df_survival["spend_multiplier"] == m]
        base_spends[str(m_val)] = round(
            float(m_rows.iloc[0]["initial_annual_cost"]))
      else:
        print(f"Warning: Model file not found: {src_path}")

  out_json = {
      "start_age": START_AGE,
      "target_age": START_AGE + int(NUM_YEARS),
      "models": models,
      "base_spends": base_spends
  }

  os.makedirs(DATA_OUT_DIR, exist_ok=True)
  json_path = os.path.join(DATA_OUT_DIR, "dp_calc.json")
  with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)
  print(f"✅ {json_path} を保存しました。")


def main():
  parser = argparse.ArgumentParser(description="40歳リタイア開始・95歳までの分析・可視化スクリプト。")
  parser.add_argument(
      "--exp_type",
      type=str,
      default="optimal-pension",
      help="実験設定 (optimal-pension, pen70-lifeplan, pen70-formula, pen70-ds)")
  args = parser.parse_args()

  exp_types = args.exp_type.split(",")
  target_year = str(NUM_YEARS)

  for et in exp_types:
    et = et.strip()
    csv_path = f"data/all_40yr/{et}.csv"
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
    elif et == "pen70-formula":
      run_pen70_formula_analysis(df_all, target_year)
      generate_dp_calc_json(df_all)
    elif et == "pen70-ds":
      run_pen70_ds_analysis(df_all, target_year)
    else:
      print(f"Unknown experiment type: {et}")


if __name__ == "__main__":
  main()
