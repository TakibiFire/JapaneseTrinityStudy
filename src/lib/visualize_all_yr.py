"""
各年数（50年、60年等）のグリッド分析結果を処理・可視化するための共通ライブラリ。
"""

import os
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd


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


def run_best_combination_analysis(
    df_survival: pd.DataFrame,
    target_year: str,
    img_dir: str,
    temp_dir: str,
    title_prefix: str = "",
    output_name: str = "best_strategy.svg",
    dim_cols: List[str] = ['spend_multiplier', 'spending_rule'],
    pref_order: List[str] = ["P60_D1", "P65_D1", "P60_D0", "P65_D0"],
    threshold: float = 0.01,
    width: int = 500,
    height: int = 450):
  """
  (受給開始年齢 × Dynamic Spending) の最適な組み合わせを分析する。

  Args:
    df_survival: 生存確率データ（value_type == "survival"）
      必要なカラム:
        - pension_start_age: 年金受給開始年齢
        - use_dynamic_spending: ダイナミックスペンディングの使用有無 (0 or 1)
        - spend_multiplier: 支出倍率
        - spending_rule: 初期支出率 (%)
        - initial_annual_cost: 初期年間支出額
        - target_year: 指定された年数（例: "45"）の生存確率
    target_year: 分析対象 of 年数（例: "45"）
    img_dir: 画像の保存先ディレクトリ
    temp_dir: 一時ファイルの保存先ディレクトリ
    title_prefix: グラフタイトルの接頭辞
    output_name: 画像のファイル名
    dim_cols: 分析対象のディメンション列
    pref_order: 戦略の優先順位 (Pxx_Dx 形式)
    threshold: 最適戦略を選択する際の許容差 (デフォルト 0.01 = 1%)
    width: グラフの幅
    height: グラフの高さ
  """
  # 必要なカラムの確認
  required_cols = [
      'pension_start_age', 'use_dynamic_spending', 'initial_annual_cost'
  ] + dim_cols + [target_year]
  for col in required_cols:
    if col not in df_survival.columns:
      raise ValueError(f"Required column '{col}' not found in df_survival")

  # 重複の確認
  check_cols = dim_cols + ['pension_start_age', 'use_dynamic_spending']
  dupes = df_survival.duplicated(subset=check_cols)
  if dupes.any():
    raise ValueError(
        f"Duplicate entries found for combinations of {check_cols}. "
        "Each combination must have only one survival probability.")

  print(
      f"\n\n{'='*20} 最適な組み合わせ (受給開始年齢 × Dynamic Spending) の分析: {title_prefix} {'='*20}"
  )

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
    for pref in pref_order:
      match = sorted_group[sorted_group["combo"] == pref]
      if not match.empty:
        row = match.iloc[0]
        # 許容範囲: top_prob - threshold 以上なら採用
        if float(row[target_year]) >= (top_prob - threshold):
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

  results = []
  for _, group in df_survival.groupby(dim_cols):
    results.append(get_selected_strategy(group))
  df_best = pd.DataFrame(results)

  # ヒートマップ用のテキストラベル (3行目: 反転時のギャップ)
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
  output_path = os.path.join(img_dir, output_name)
  title = f"戦略選択: {title_prefix} (優先順: 60歳あり > 65歳あり > 60歳なし > 65歳なし, 許容差{threshold*100:g}%)"
  create_best_combo_heatmap(df_best,
                            title=title,
                            x_col="rule_label",
                            x_title="初期支出率 (%ルール)",
                            y_col="multiplier_label",
                            y_title="支出レベル",
                            output_path=output_path,
                            x_sort=r_order,
                            y_sort=m_order,
                            width=width,
                            height=height)

  # CSV出力
  csv_filename = output_name.replace(".svg", ".csv")
  csv_output = os.path.join(temp_dir, csv_filename)
  os.makedirs(temp_dir, exist_ok=True)
  df_best.to_csv(csv_output, index=False)
  print(f"✅ {csv_output} にCSVを保存しました。")

  print(f"\n--- {title_prefix} {target_year}年後生存確率を最大化する組み合わせの分布 ---")
  counts = df_best["display_combo"].value_counts().sort_index()
  print(counts.to_string())


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
