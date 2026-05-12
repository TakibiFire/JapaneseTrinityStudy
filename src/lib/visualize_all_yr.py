"""
各年数（50年、60年等）のグリッド分析結果を処理・可視化するための共通ライブラリ。
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

from src.lib.survival_contours import (generate_rule_of_thumb,
                                       generate_smooth_contour_data,
                                       get_contour_anchor_points,
                                       save_contour_charts)
from src.lib.survival_formula_analysis import run_survival_formula_analysis
from src.lib.visualize import create_survival_probability_chart


def create_heatmap(df: pd.DataFrame,
                   target_col: str,
                   title: str,
                   x_col: str,
                   x_title: str,
                   y_col: str,
                   y_title: str,
                   output_path: str,
                   x_sort: Optional[List[Any]] = None,
                   y_sort: Optional[List[Any]] = None):
  """
  生存確率のヒートマップを作成して保存する。

  Args:
    df: 分析対象のデータフレーム
    target_col: 生存確率が格納されている列名（例: "45"）
    title: グラフのタイトル
    x_col: X軸に使用する列
    x_title: X軸のタイトル
    y_col: Y軸に使用する列
    y_title: Y軸のタイトル
    output_path: 保存先のフルパス
    x_sort: X軸のソート順
    y_sort: Y軸のソート順
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

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def prepare_heatmap_labels(
    df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
  """
  ヒートマップ表示用のラベル列を追加し、ソート順を計算する。
  元のデータフレームは変更せず、コピーを返す。

  Args:
    df: 分析対象のデータフレーム。
        必須な column: initial_annual_cost, spend_multiplier, spending_rule

  Returns:
    df: ラベル列 (multiplier_label, rule_label) が追加されたデータフレーム
    m_order: 支出レベル (multiplier_label) のソート順
    r_order: 初期支出率 (rule_label) のソート順
  """
  df = df.copy()
  df["multiplier_label"] = df.apply(
      lambda r:
      f"{int(round(r['initial_annual_cost'])):d}万円/年@(x{r['spend_multiplier']:g})",
      axis=1)
  df["rule_label"] = df["spending_rule"].map(
      lambda x: f"{x:g}%@(x{round(100/x, 1):g})")

  actual_multipliers = sorted(df["spend_multiplier"].unique(), reverse=True)
  actual_rules = sorted(df["spending_rule"].unique())

  m_order = []
  for m in actual_multipliers:
    cost = df[df["spend_multiplier"] == m]["initial_annual_cost"].iloc[0]
    m_order.append(f"{int(round(cost)):d}万円/年@(x{m:g})")
  r_order = [f"{x:g}%@(x{round(100/x, 1):g})" for x in actual_rules]

  return df, m_order, r_order


def create_best_combo_heatmap(df_best: pd.DataFrame,
                              title: str,
                              x_col: str,
                              x_title: str,
                              y_col: str,
                              y_title: str,
                              output_path: str,
                              x_sort: Optional[List[Any]] = None,
                              y_sort: Optional[List[Any]] = None,
                              width: int = 500,
                              height: int = 450):
  """
  最適な組み合わせ(Pxx_Dx)を可視化するヒートマップ。

  Args:
    df_best: 最適な戦略が格納されたデータフレーム
    title: グラフのタイトル
    x_col: X軸に使用する列
    x_title: X軸のタイトル
    y_col: Y軸に使用する列
    y_title: Y軸のタイトル
    output_path: 保存先のフルパス
    x_sort: X軸のソート順
    y_sort: Y軸のソート順
    width: グラフの幅
    height: グラフの高さ
  """
  plot_df = df_best.copy()

  # 戦略ごとのカラーマップ
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

  # テキストには戦略、確率、および反転時のギャップを表示
  text = base.mark_text(baseline='middle',
                        lineBreak='\n').encode(text=alt.Text('combo_label:N'),
                                               color=alt.value('black'))

  chart = (heatmap + text).properties(title=title, width=width, height=height)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def create_spend_percentile_chart(df: pd.DataFrame,
                                  title: str,
                                  output_path: str,
                                  start_age: int,
                                  num_years: int,
                                  width: int = 600,
                                  height: int = 400,
                                  show_legend: bool = True,
                                  color_domain: Optional[List[str]] = None,
                                  color_range: Optional[List[str]] = None):
  """
  支出額のパーセンタイル推移(25p, 50p, 75p)を可視化する。
  Dynamic SpendingのON/OFF比較をサポートする。

  注:
  - `use_dynamic_spending` カラムが存在する場合、値を ON/OFF にマッピングして色分けします。
  - `strategy` カラムが存在する場合、その値をそのまま凡例ラベルとして使用します。
  - ラベル内に '@' を含めると、凡例表示時にそこで改行されます。

  Args:
    df: 分析対象のデータフレーム。
      Required columns:
      - value_type: 値の種類 ('spend25p', 'spend50p', 'spend75p')
      - "1" から str(num_years) までの数字の列: 各経過年の支出額
      - group_label (または use_dynamic_spending, strategy): 凡例に表示するラベル。
        ラベル内に '@' を含めると改行されます。
    title: グラフのタイトル
    output_path: 保存先のパス
    start_age: シミュレーション開始時の年齢 (x軸の計算に使用)
    num_years: シミュレーション期間（年数）
    width: グラフの幅
    height: グラフの高さ
    show_legend: 凡例を表示するかどうか
    color_domain: 色を適用する値のリスト（任意）
    color_range: 適用する色のリスト（任意）
  """
  # 1からnum_yearsまでの列を年度列として扱う
  year_cols = [str(i) for i in range(1, num_years + 1) if str(i) in df.columns]

  # 年度列以外のすべての列を識別子(id_vars)として保持
  id_vars = [c for c in df.columns if c not in year_cols]

  # 必要な値の種類のみに絞り込む
  plot_df = df[df["value_type"].isin(["spend25p", "spend50p", "spend75p"])]

  df_long = plot_df.melt(id_vars=id_vars,
                         value_vars=year_cols,
                         var_name="year",
                         value_name="spend")
  df_long["year"] = df_long["year"].astype(int)
  # 年数から年齢に変換
  df_long["age"] = df_long["year"] + start_age

  # use_dynamic_spending が存在しない場合は、strategy をラベルとして使用する
  if "use_dynamic_spending" in df_long.columns:
    df_long["group_label"] = df_long["use_dynamic_spending"].map({
        1: "ON",
        0: "OFF"
    })
    color_scale = alt.Scale(domain=["ON", "OFF"], range=["red", "blue"])
    legend_title = "ダイナミックスペンディング"
  elif "strategy" in df_long.columns:
    df_long["group_label"] = df_long["strategy"]
    color_scale = alt.Scale()
    legend_title = "戦略"
  else:
    df_long["group_label"] = "Total"
    color_scale = alt.Scale()
    legend_title = "グループ"

  # カラー設定の上書き
  if color_domain is not None and color_range is not None:
    color_scale = alt.Scale(domain=color_domain, range=color_range)

  # p25, p50, p75 を列に展開
  pivot_df = df_long.pivot_table(index=["group_label", "age"],
                                 columns="value_type",
                                 values="spend").reset_index()

  # Altairでプロット
  base = alt.Chart(pivot_df).encode(x=alt.X("age:Q", title="年齢"))

  # 凡例の改行(split)対応
  legend_option = alt.Legend(
      orient='top',
      labelExpr="split(datum.label, '@')") if show_legend else None
  area = base.mark_area(opacity=0.3).encode(y=alt.Y("spend25p:Q",
                                                    title="年間取り崩し額 (万円)"),
                                            y2="spend75p:Q",
                                            color=alt.Color(
                                                "group_label:N",
                                                scale=color_scale,
                                                title=legend_title,
                                                legend=legend_option))

  # Line (50p)
  line = base.mark_line().encode(y="spend50p:Q",
                                 color=alt.Color("group_label:N",
                                                 legend=legend_option))

  chart = (area + line).properties(title=title, width=width, height=height)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def calculate_preference_order(df_survival: pd.DataFrame, target_year: str,
                               threshold: float, dim_cols: List[str],
                               value_col: str) -> List[Any]:
  """
  全グリッドセルにおける出現頻度に基づいて優先順位を自動計算する。
  """
  counts: Dict[Any, int] = {}

  for _, group in df_survival.groupby(dim_cols):
    max_prob = float(group[target_year].max())
    within_threshold = group[group[target_year] >= (
        max_prob - threshold)][value_col].tolist()
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
                                 x_sort: Optional[List[Any]] = None,
                                 y_sort: Optional[List[Any]] = None,
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
                                   x_sort: Optional[List[Any]] = None,
                                   y_sort: Optional[List[Any]] = None,
                                   width: int = 500,
                                   height: int = 450,
                                   all_ages: Optional[List[str]] = None):
  """
  最適な年金受給開始年齢を可視化するヒートマップ。
  """
  color_map = {
      "60歳": "#FBD38D",  # Light orange
      "62歳": "#F6AD55",  # Orange
      "65歳": "#9AE6B4",  # Light green
      "68歳": "#68D391",  # Green
      "70歳": "#B2F5EA",  # Light teal
      "71歳": "#81E6D9",  # Teal
      "73歳": "#FC8181",  # Pink/Red
      "75歳": "#FEB2B2"  # Light red
  }
  if all_ages is not None:
    color_map = {k: v for k, v in color_map.items() if k in all_ages}

  return create_best_strategy_heatmap(df_best, title, x_col, x_title, y_col,
                                      y_title, output_path, "display_age",
                                      "受給開始年齢", color_map, x_sort, y_sort,
                                      width, height)


def format_age_range_label(selected_ages: List[float],
                           all_ages: List[float]) -> str:
  """
  選ばれた年齢のリストを、連続している場合は範囲（-）を使って短縮表記する。
  例: [60, 62, 65] で all_ages=[60, 62, 65, 68, 70] の場合 -> "60-65"
  """
  if not selected_ages:
    return ""

  # 数値としてソート
  selected_sorted = sorted([int(a) for a in selected_ages])
  all_sorted = sorted([int(a) for a in all_ages])

  # all_ages におけるインデックスを取得
  try:
    indices = [all_sorted.index(a) for a in selected_sorted]
  except ValueError:
    # 含まれていない年齢がある場合はそのままカンマ区切りで返す
    return ", ".join([f"{int(a)}" for a in selected_sorted])

  ranges = []
  if not indices:
    return ""

  start_idx = indices[0]
  prev_idx = indices[0]

  for i in range(1, len(indices)):
    curr_idx = indices[i]
    if curr_idx == prev_idx + 1:
      prev_idx = curr_idx
    else:
      if start_idx == prev_idx:
        ranges.append(f"{all_sorted[start_idx]}")
      else:
        ranges.append(f"{all_sorted[start_idx]}-{all_sorted[prev_idx]}")
      start_idx = curr_idx
      prev_idx = curr_idx

  if start_idx == prev_idx:
    ranges.append(f"{all_sorted[start_idx]}")
  else:
    ranges.append(f"{all_sorted[start_idx]}-{all_sorted[prev_idx]}")

  return ", ".join(ranges)


def run_optimal_pension_age_analysis(df_all: pd.DataFrame,
                                     target_year: str,
                                     img_dir: str,
                                     start_age: int,
                                     num_years: int,
                                     threshold: float = 0.01,
                                     shorten_labels: bool = False,
                                     output_prefix: str = ""):
  """
  最適な年金受給開始年齢を分析し、ヒートマップと生存曲線を作成する。
  """
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  if df_survival.empty:
    print("Error: Survival data not found.")
    return

  # 優先順位を自動計算
  dim_cols = ['spend_multiplier', 'spending_rule']
  pref_order = calculate_preference_order(df_survival, target_year, threshold,
                                          dim_cols, "pension_start_age")
  print(f"Computed preference order for pension ages: {pref_order}")

  all_ages = sorted(df_survival["pension_start_age"].unique().tolist())

  def get_best_age(group: pd.DataFrame) -> pd.Series:
    max_prob = float(group[target_year].max())
    pref_map = {age: i for i, age in enumerate(pref_order)}
    temp_group = group.copy()
    temp_group["pref_score"] = temp_group["pension_start_age"].map(pref_map)
    sorted_group = temp_group.sort_values(by=[target_year, "pref_score"],
                                          ascending=[False, True])

    within_threshold_rows = sorted_group[sorted_group[target_year] >= (
        max_prob - threshold)]
    within_threshold_ages = sorted(
        within_threshold_rows["pension_start_age"].tolist())

    selected_row = None
    for age in pref_order:
      if age in within_threshold_ages:
        selected_row = group[group["pension_start_age"] == age].iloc[0].copy()
        break
    if selected_row is None:
      selected_row = within_threshold_rows.iloc[0].copy()

    selected_row["display_age"] = f"{int(selected_row['pension_start_age'])}歳"

    # ラベル作成
    label = f"{max_prob*100:.1f}%"
    if shorten_labels:
      age_str = format_age_range_label(within_threshold_ages, all_ages)
      label += f"\n{age_str}"
    else:
      # 旧来の形式
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
  output_path = os.path.join(img_dir,
                             f"{output_prefix}optimal_pension_age_heatmap.svg")
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


def create_improvement_heatmap(df: pd.DataFrame,
                               target_col: str,
                               title: str,
                               x_col: str,
                               x_title: str,
                               y_col: str,
                               y_title: str,
                               output_path: str,
                               x_sort: Optional[List[Any]] = None,
                               y_sort: Optional[List[Any]] = None,
                               width: int = 500,
                               height: int = 450):
  """
  改善幅を可視化するヒートマップ。
  """
  plot_df = df.copy()
  plot_df["val"] = plot_df[target_col]
  plot_df["val_pct"] = plot_df["val"] * 100

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
      color=alt.Color('val:Q', title='改善幅 (%)', scale=alt.Scale(
          scheme='blues')))

  text = base.mark_text(baseline='middle').encode(text=alt.Text('val_pct:Q',
                                                                format='.1f'),
                                                  color=alt.condition(
                                                      alt.datum.val > 0.1,
                                                      alt.value('white'),
                                                      alt.value('black')))

  chart = (heatmap + text).properties(title=title, width=width, height=height)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def create_pension_survival_curve(df: pd.DataFrame, multiplier: float,
                                  rule: float, title: str, output_path: str,
                                  start_age: int, num_years: int):
  """
  指定された multiplier と rule における、受給開始年齢別の生存確率推移を描画する。
  """
  # 年度列 (1, 2, ..., num_years) を取得
  year_cols = [str(i) for i in range(1, num_years + 1) if str(i) in df.columns]

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
                                               start_age=start_age,
                                               height=300)

  chart = chart.properties(title=title)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def run_lifeplan_analysis(df_all: pd.DataFrame,
                          target_year: str,
                          img_dir: str,
                          threshold: float = 0.01):
  """
  リバランス戦略（R70, V1, 固定, なし）の比較分析を実行する。

  Args:
    df_all: 実験結果のデータフレーム
    target_year: ターゲット年 (str)
    img_dir: 画像保存ディレクトリ
    threshold: 許容差 (デフォルト 0.01)
  """
  print(f"\n\n{'='*20} リバランス戦略の分析 (lifeplan) {'='*20}")

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
  output_path = os.path.join(img_dir, "best_rebalance_strategy_heatmap.svg")

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
    output_path_imp = os.path.join(img_dir, "improvement_r70_vs_v1_heatmap.svg")

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

  Args:
    df_r70: R70の結果データフレーム (spend_multiplier, spending_rule, target_yearが必要)
    df_v1: V1の結果データフレーム (spend_multiplier, spending_rule, target_yearが必要)
    target_year: ターゲット年 (str)
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


def run_common_formula_analysis(df_survival: pd.DataFrame,
                                target_year: str,
                                img_dir: str,
                                data_out_dir: str,
                                start_age: int,
                                pension_start: int,
                                title: str,
                                prefix: str,
                                target_probs: Optional[List[float]] = None,
                                output_json: Optional[str] = None,
                                generate_heatmap: bool = True):
  """
  生存確率グリッドに対して、ヒートマップ作成、コンター作成、モデルフィッティング、JSON保存を行う。

  Args:
    df_survival: 生存確率のデータフレーム (spend_multiplier, spending_rule, target_yearが必要)
    target_year: ターゲット年 (str)
    img_dir: 画像保存ディレクトリ
    data_out_dir: データ保存ディレクトリ
    start_age: リタイア開始年齢
    pension_start: 年金受給開始年齢
    title: ヒートマップのタイトル
    prefix: 保存ファイル名の接頭辞
    target_probs: 可視化対象の生存確率リスト (デフォルト: [0.97, 0.95, 0.90, 0.80, 0.70])
    output_json: 保存するJSONファイル名 (Noneの場合は保存しない)
    generate_heatmap: ヒートマップを生成するかどうか
  """
  print(f"\n\n{'='*20} 生存確率式の分析 ({prefix}) {'='*20}")

  if target_probs is None:
    target_probs = [0.97, 0.95, 0.90, 0.80, 0.70]

  # 1. ヒートマップ
  if generate_heatmap:
    df_h, m_order, r_order = prepare_heatmap_labels(df_survival)
    output_path = os.path.join(img_dir, f"{prefix}heatmap.svg")

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
  plot_data = []
  for p in target_probs:
    anchors = get_contour_anchor_points(df_survival, p, target_year)
    plot_data.extend(generate_smooth_contour_data(anchors, f"{p*100:g}%"))
  df_plot_survival = pd.DataFrame(plot_data)

  # 3. グラフ保存
  save_contour_charts(df_plot_survival,
                      target_probs,
                      img_dir=img_dir,
                      prefix=prefix)

  # 4. Rule of Thumb
  generate_rule_of_thumb(df_survival, target_probs, target_year)

  # 5. 詳細な近似モデルの分析
  coeffs = run_survival_formula_analysis(df_survival, target_year)

  # 6. JSON出力
  if coeffs and output_json:
    os.makedirs(data_out_dir, exist_ok=True)
    out_json = {
        "start_age": start_age,
        "target_age": start_age + int(target_year),
        "formula": coeffs
    }
    if pension_start > 0:
      out_json["pension_start"] = pension_start

    # household 情報がある場合は追加
    if "household" in df_survival.columns:
      out_json["household"] = df_survival["household"].iloc[0]

    json_path = os.path.join(data_out_dir, output_json)
    with open(json_path, "w") as f:
      json.dump(out_json, f, indent=2)
    print(f"✅ {json_path} を保存しました。")


def run_ds_comparison_analysis(df_survival: pd.DataFrame,
                               df_formula_survival: pd.DataFrame,
                               target_year: str, img_dir: str, num_years: int,
                               start_age: int, title_main: str,
                               output_prefix: str,
                               comp_cases: Optional[List[tuple[float, float]]] = None):
  """
  SpendAwareDSとBase Formulaの比較分析を実行する。

  Args:
    df_survival: DSの結果データフレーム (spend_multiplier, spending_rule, target_yearが必要)
    df_formula_survival: Base Formulaの結果データフレーム (spend_multiplier, spending_rule, target_yearが必要)
    target_year: ターゲット年 (str)
    img_dir: 画像保存ディレクトリ
    num_years: シミュレーション期間
    start_age: リタイア開始年齢
    title_main: ヒートマップのメインタイトル
    output_prefix: 接頭辞
    comp_cases: 比較するケースのリスト [(multiplier, rule), ...]。Noneの場合は生成しない。
  """
  print(f"\n\n{'='*20} SpendAwareDS 比較分析 {'='*20}")

  # 1. ヒートマップ
  df_h, m_order, r_order = prepare_heatmap_labels(df_survival)
  output_path = os.path.join(img_dir, f"{output_prefix}heatmap.svg")

  create_heatmap(df_h,
                 target_col=target_year,
                 title=title_main,
                 x_col="rule_label",
                 x_title="初期支出率 (%ルール)",
                 y_col="multiplier_label",
                 y_title="支出レベル",
                 output_path=output_path,
                 x_sort=r_order,
                 y_sort=m_order)

  # 2. 比較ヒートマップ
  df_comp = pd.merge(
      df_survival[[
          'spend_multiplier', 'spending_rule', 'initial_annual_cost',
          target_year
      ]],
      df_formula_survival[['spend_multiplier', 'spending_rule', target_year]],
      on=['spend_multiplier', 'spending_rule'],
      suffixes=('_ds', '_formula'))

  df_comp["improvement"] = df_comp[f"{target_year}_ds"] - df_comp[
      f"{target_year}_formula"]

  df_comp, m_order_comp, r_order_comp = prepare_heatmap_labels(df_comp)

  title_imp = f"SpendAwareDSによる改善幅 ({target_year}年後生存確率 差分)"
  output_path_imp = os.path.join(img_dir,
                                 f"{output_prefix}improvement_ds_vs_formula_heatmap.svg")

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

  # 3. 生存確率曲線の比較
  if comp_cases:
    for m, r in comp_cases:
      ds_row = df_survival[(df_survival["spend_multiplier"] == m) &
                           (df_survival["spending_rule"] == r)]
      fo_row = df_formula_survival[(df_formula_survival["spend_multiplier"] == m)
                                   & (df_formula_survival["spending_rule"] == r)]

      if ds_row.empty or fo_row.empty:
        continue

      year_cols = [str(i) for i in range(1, num_years + 1)]
      ds_vals = ds_row[year_cols].values[0]
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
                                                   start_age=start_age,
                                                   height=300)
      chart = chart.properties(title=f"DS vs Formula 生存確率 (m={m}, r={r}%)")
      filename = f"{output_prefix}comp_ds_formula_m{str(m).replace('.', '_')}_r{str(r).replace('.', '_')}.svg"
      chart.save(os.path.join(img_dir, filename))
      print(f"✅ {filename} に保存しました。")


def generate_dp_calc_json_common(df_all: pd.DataFrame,
                                 data_out_dir: str,
                                 start_age: int,
                                 num_years: int,
                                 model_prefix: str,
                                 model_src_dir: str = "data/optimal_strategy_dp"):
  """
  生存確率計算機（DP版）のための設定JSONを生成する共通関数。

  Args:
    df_all: 実験結果のデータフレーム (value_type='survival'が必要)
    data_out_dir: データ保存ディレクトリ
    start_age: リタイア開始年齢
    num_years: シミュレーション期間
    model_prefix: モデルファイルの接頭辞 (例: 're60_pen70_95')
    model_src_dir: モデルファイルのソースディレクトリ
  """
  print(f"\n\n{'='*20} DP計算機用JSONの生成 {'='*20}")

  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  multipliers = sorted(df_survival["spend_multiplier"].unique())
  base_spends = {}
  models = {}

  for m in multipliers:
    m_val = float(m)
    m_key = str(m_val).replace(".", "_") if m_val % 1 != 0 else str(int(m_val))
    if m_key.endswith("_0"):
      m_key = m_key[:-2]

    # モデルファイル名 (例: re60_pen70_95_m0_75.json)
    model_name = f"{model_prefix}_m{m_key}.json"
    src_path = os.path.join(model_src_dir, model_name)
    dst_path = os.path.join(data_out_dir, model_name)

    if os.path.exists(src_path):
      shutil.copy(src_path, dst_path)
      print(f"Copied {src_path} -> {dst_path}")
      models[str(m_val)] = model_name
      m_rows = df_survival[df_survival["spend_multiplier"] == m]
      base_spends[str(m_val)] = round(
          float(m_rows.iloc[0]["initial_annual_cost"]))
    elif os.path.exists(dst_path):
      models[str(m_val)] = model_name
      m_rows = df_survival[df_survival["spend_multiplier"] == m]
      base_spends[str(m_val)] = round(
          float(m_rows.iloc[0]["initial_annual_cost"]))
    else:
      print(f"Warning: Model file not found: {src_path}")

  out_json = {
      "start_age": start_age,
      "target_age": start_age + int(num_years),
      "models": models,
      "base_spends": base_spends
  }

  os.makedirs(data_out_dir, exist_ok=True)
  json_path = os.path.join(data_out_dir, "dp_calc.json")
  with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)
  print(f"✅ {json_path} を保存しました。")
