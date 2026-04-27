"""
data/dynamic_rebalance/{exp_name}.csv の結果を可視化し、数式フィッティングを行うスクリプト。
指定された年数（30年、50年など）と実験名に応じた生存確率を
ヒートマップとして表示し、さらに数式近似を出力する。
"""

import argparse
import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fitting_model(X, a, b, c, d, e, f, g):
  """
  フィットさせる数式モデル。
  L: lower_limit, U: upper_limit
  """
  L, U = X
  term1 = a * np.power(np.abs(L - b), c)
  term2 = d * np.power(np.abs(U - e), f)
  return term1 + term2 + g


def run_fitting(df: pd.DataFrame, target_years: List[str], exp_name: str):
  """
  数式フィッティングを実行し、結果を表示する。
  """
  L = np.asarray(df["lower_limit"], dtype=float)
  U = np.asarray(df["upper_limit"], dtype=float)

  print("\n" + "=" * 50)
  print(f"数式フィッティング結果 (実験: {exp_name})")
  print("=" * 50)

  print("\n| 期間 | 近似式 | 決定係数 ($R^2$) |")
  print("| :--- | :--- | :--- |")

  for year in target_years:
    if year not in df.columns:
      continue
    y = np.asarray(df[year], dtype=float)
    # 初期推測値
    p0 = [-5.0, -0.05, 1.0, -2.0, -0.05, 1.0, 1.0]
    # 探索範囲
    bounds = ([-100, -1, 0.1, -100, -1, 0.1, -10], [0, 1, 5, 0, 1, 5, 10])

    try:
      popt, _ = curve_fit(fitting_model, (L, U),
                          y,
                          p0=p0,
                          bounds=bounds,
                          maxfev=50000)
      a, b, c, d, e, f, g = popt
      y_pred = fitting_model((L, U), *popt).astype(float)
      r2 = 1.0 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

      # 式の構築 (符号の調整)
      def fmt_term(val):
        if val < 0:
          return f"+ {-val:.4f}"
        else:
          return f"- {val:.4f}"

      l_term = fmt_term(b)
      u_term = fmt_term(e)

      formula = (f"$\\text{{Rate}} = {a:.2f} \\times (L {l_term})^{{{c:.2f}}} "
                 f"{d:+.2f} \\times (U {u_term})^{{{f:.2f}}} + {g:.4f}$")

      print(f"| **{year}年** | {formula} | {r2:.4f} |")
    except Exception as err:
      print(f"| **{year}年** | Error: {err} | - |")


def print_representative_stats(df: pd.DataFrame, exp_name: str):
  """
  代表的な戦略の生存確率を表示する。
  """
  print("\n" + "=" * 50)
  print(f"代表的な戦略の生存確率 (実験: {exp_name})")
  print("=" * 50)

  conditions = [(0.00, 0.00, "何もしない"), (0.03, 0.00, "支出は落とさない<br>上限3%, 下限0%"),
                (0.05, -0.015, "ヴァンガード社の手法<br>上限5%, 下限-1.5%"),
                (0.00, -0.03, "一番辛そう<br>上限0%, 下限-3%")]

  stats_data = []
  l_vals = np.asarray(df["lower_limit"], dtype=float)
  u_vals = np.asarray(df["upper_limit"], dtype=float)

  # 表形式での出力準備
  table_rows = ["30年生存確率", "50年生存確率"]
  table_cols = []
  col_names = []

  for up, low, label in conditions:
    res = df[(np.isclose(u_vals, up)) & (np.isclose(l_vals, low))]
    if not res.empty:
      s30 = float(res["30"].values[0])
      s50 = float(res["50"].values[0])
      stats_data.append({
          "ラベル": label,
          "30年": f"{s30:.1%}",
          "50年": f"{s50:.1%}"
      })

      col_label = label.replace("\n", "<br>")
      col_names.append(col_label)
      s30_str = f"{s30:.1%}"
      s50_str = f"{s50:.1%}"
      # ヴァンガード社の手法を強調
      if "ヴァンガード" in label:
        s30_str = f"=={s30_str}=="
        s50_str = f"=={s50_str}=="
      table_cols.append([s30_str, s50_str])

  if stats_data:
    print(pd.DataFrame(stats_data))

    # Markdownテーブルの出力
    print("\n[Markdown Table]")
    header = "| 手法 | " + " | ".join(col_names) + " |"
    sep = "|---| " + " | ".join(["--:"] * len(col_names)) + " |"
    print(header)
    print(sep)
    for i, row_name in enumerate(table_rows):
      row_str = f"| {row_name} | " + " | ".join([col[i] for col in table_cols
                                                ]) + " |"
      print(row_str)
  else:
    print("条件に一致するデータが見つかりませんでした。")


def create_spend_distribution_charts(df: pd.DataFrame, exp_name: str):
  """
  実質支出額の分布ヒストグラムを作成する。
  """
  print(f"\n実質支出額の分布を生成中... (実験: {exp_name})")

  target_years = [30, 40, 50]
  # 2.7%ルールの初期支出
  initial_spend = 270.0

  upper = df["upper_limit"].iloc[0]
  lower = df["lower_limit"].iloc[0]

  os.makedirs("docs/imgs/dynamic_spending", exist_ok=True)

  # 指定年のパーセンタイル計算 (生存パスのみ)
  percentiles_to_print = [1, 5, 10, 25, 50, 75, 90, 95, 99]
  print(f"\n--- Real Spending Percentiles (実質・万円) - Experiment: {exp_name} ---")

  for y in target_years:
    col_name = str(y)
    if col_name not in df.columns:
      continue

    vals_real = np.asarray(df[col_name], dtype=float)
    # 破産ケース(0.0)を除外
    survivors_real = vals_real[vals_real > 0]
    bankrupt_rate = float(np.mean(vals_real == 0))

    print(f"\nYear {y} ({bankrupt_rate:.1%} bankrupt paths excluded):")
    if len(survivors_real) > 0:
      for p in percentiles_to_print:
        val = np.percentile(survivors_real, p)
        print(f"  {p}th: {val:.2f}")
    else:
      print("  No survivors.")

  for y in target_years:
    col_name = str(y)
    if col_name not in df.columns:
      continue

    vals_real = np.asarray(df[col_name], dtype=float)
    survivors_real = vals_real[vals_real > 0]
    bankrupt_rate = float(np.mean(vals_real == 0))

    if len(survivors_real) == 0:
      continue

    valid_data = pd.DataFrame({'real_spend': survivors_real})

    # パーセンタイル計算 (生存パス)
    p25 = float(np.percentile(survivors_real, 25))
    p50 = float(np.percentile(survivors_real, 50))
    p75 = float(np.percentile(survivors_real, 75))

    # ヒストグラム
    hist = alt.Chart(valid_data).mark_bar(opacity=0.6).encode(
        alt.X("real_spend:Q", bin=alt.Bin(maxbins=100), title="年間実質支出額 (万円)"),
        alt.Y("count()", title="試行回数"))

    # 垂直線のデータ
    rules_data = pd.DataFrame([{
        'x': initial_spend,
        'label': '初期の生活水準 (100%)'
    }, {
        'x': p25,
        'label': '25パーセンタイル'
    }, {
        'x': p50,
        'label': '50パーセンタイル (中央値)'
    }, {
        'x': p75,
        'label': '75パーセンタイル'
    }])

    rules = alt.Chart(rules_data).mark_rule(size=2).encode(
        x='x:Q',
        color=alt.Color(
            'label:N',
            title='',
            scale=alt.Scale(domain=[
                '初期の生活水準 (100%)', '25パーセンタイル', '50パーセンタイル (中央値)', '75パーセンタイル'
            ],
                            range=['red', 'blue', 'green', 'purple'])),
        strokeDash=alt.condition(alt.datum.label == '初期の生活水準 (100%)',
                                 alt.value([5, 5]), alt.value([0])))

    final_chart = (hist + rules).properties(
        title=
        f"実質支出額（購買力）の分布: {y}年目 (上限{upper:.1%}, 下限{lower:.1%}) - 破産ケース {bankrupt_rate:.1%} を除外",
        width=600,
        height=300).configure_legend(orient='top', titleOrient='left')

    svg_path = f"docs/imgs/dynamic_spending/real_spend_hist_year_{y}.svg"
    try:
      import vl_convert as vlc
      svg_str = vlc.vegalite_to_svg(final_chart.to_json())
      with open(svg_path, "w") as f:
        f.write(svg_str)
      print(f"✅ {svg_path} に保存しました。")
    except (ImportError, Exception) as e:
      print(f"Could not save as SVG via vl-convert: {e}. Saving as HTML.")
      final_chart.save(svg_path.replace(".svg", ".html"))


def process_experiment(exp_name: str):
  """
  個別の実験結果ファイルを処理する。
  """
  csv_path = f"data/dynamic_rebalance/{exp_name}.csv"
  old_csv_path = "data/dynamic_spending_grid_comp.csv"

  if os.path.exists(csv_path):
    path_to_load = csv_path
  elif exp_name == "4p" and os.path.exists(old_csv_path):
    path_to_load = old_csv_path
  else:
    print(f"Error: {csv_path} not found (and no fallback for {exp_name}).")
    return

  df = pd.read_csv(path_to_load)
  print(f"\nProcessing: {exp_name} (from {path_to_load})")

  # 特殊な実験 (支出分布のダンプ) の場合
  if exp_name == "1p_1.5p_spend":
    create_spend_distribution_charts(df, exp_name)
    return

  # 代表的な統計を表示
  print_representative_stats(df, exp_name)

  # ヒートマップを生成する対象年数
  target_years = ["30", "50"]

  # フィッティングの実行 (2.7p または 2.7p_dp の場合)
  if exp_name in ["2.7p", "2.7p_dp"]:
    run_fitting(df, ["20", "30", "40", "50"], exp_name)

  # ヒートマップの生成
  print(f"\nヒートマップを生成中... (実験: {exp_name})")
  for target_year in target_years:
    if target_year not in df.columns:
      continue

    # データの抽出
    plot_df = df[["upper_limit", "lower_limit", target_year]].copy()
    plot_df.columns = ["upper_limit", "lower_limit",
                       "survival_rate"]  # type: ignore

    # 表示用に値を調整 (0.0〜1.0 -> 0%〜100%)
    plot_df["survival_rate_pct"] = plot_df["survival_rate"] * 100

    # Altair ヒートマップの作成
    base = alt.Chart(plot_df).encode(
        x=alt.X('lower_limit:O',
                title='下限 (lower_limit)',
                axis=alt.Axis(format='.1%')),
        y=alt.Y('upper_limit:O',
                title='上限 (upper_limit)',
                sort='descending',
                axis=alt.Axis(format='.1%')),
    )

    # ヒートマップ部分
    heatmap = base.mark_rect().encode(color=alt.Color(
        'survival_rate:Q',
        title='生存確率',
        scale=alt.Scale(scheme='redyellowgreen',
                        domain=[plot_df["survival_rate"].min(), 1.0])))

    # テキスト部分 (背景色に応じて文字色を変更)
    text_color_threshold = plot_df["survival_rate"].quantile(0.3)

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('survival_rate_pct:Q', format='.1f'),
        color=alt.condition(alt.datum.survival_rate > text_color_threshold,
                            alt.value('black'), alt.value('white')))

    is_dyn = df["is_dynamic_rebalance"].iloc[0]
    dyn_label = "あり" if is_dyn == 1 else "なし"
    chart = (heatmap + text).properties(
        title=f'{target_year}年後の生存確率 (%) (実験: {exp_name}, DR: {dyn_label})',
        width=400,
        height=300)

    # 保存
    output_dir = "docs/imgs/dynamic_spending"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"grid_heatmap_{exp_name}_{target_year}yr_survival.svg")
    chart.save(output_path)
    print(f"✅ {output_path} に保存しました。")


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(description="グリッドシミュレーションの結果を解析・可視化する")
  parser.add_argument(
      "--exp_name",
      type=str,
      default="4p,2.7p,2.7p_dp,1p_1.5p_spend",
      help="解析対象の実験名。カンマ区切りで複数指定可能 (例: 4p,2.7p,2.7p_dp,1p_1.5p_spend)")
  args = parser.parse_args()

  exp_names = [name.strip() for name in args.exp_name.split(",")]

  for exp_name in exp_names:
    process_experiment(exp_name)


if __name__ == "__main__":
  main()
