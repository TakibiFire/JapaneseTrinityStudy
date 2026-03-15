"""
シミュレーション結果の可視化とHTML出力を行うユーティリティ。
"""

import os
import webbrowser
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd

from core import SimulationResult


def create_styled_summary(
    results: Dict[str, SimulationResult],
    quantiles: List[float] = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
    bankruptcy_years: List[int] = [20, 30, 40, 50]
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
  """
  シミュレーション結果の辞書からサマリー統計を計算し、
  データフレームとフォーマットされた Styler オブジェクトを返す。
  
  分位点や破産確率など、複数の指標を算出して視覚的に整えたテーブルを作成する。
  
  Args:
    results: 戦略名をキー、SimulationResult インスタンスを値とする辞書。
    quantiles: 算出するパーセンタイルのリスト (0.0 〜 1.0)
    bankruptcy_years: 破産確率を算出する年数のリスト
  
  Returns:
    生データの DataFrame と、表示用にフォーマット・スタイリングされた pandas Styler オブジェクトのタプル。
  """
  summary_data = {}

  # パーセンタイルのラベル名マッピング
  quantile_labels = {
      0.01: "下位1% (だいぶ運が悪い)",
      0.10: "下位10% (運が悪い)",
      0.25: "下位25% (やや不運)",
      0.50: "中央値 (普通)",
      0.75: "上位25% (やや幸運)",
      0.90: "上位10% (運が良い)"
  }

  for name, res in results.items():
    net_values = res.net_values
    sustained_months = res.sustained_months

    data = {}
    for q in quantiles:
      label = quantile_labels.get(q, f"{q*100:.0f}%パーセンタイル")
      data[label] = np.quantile(net_values, q)

    for y in bankruptcy_years:
      data[f"{y}年破産確率 (%)"] = np.mean(sustained_months < y * 12) * 100.0

    summary_data[name] = data

  summary_df = pd.DataFrame(summary_data).T

  def format_oku(x: float) -> str:
    return f"{x / 10000:.1f}億円"

  def format_pct(x: float) -> str:
    return f"{x:.1f}%"

  format_dict = {}
  formatted_df = summary_df.copy()

  for q in quantiles:
    label = quantile_labels.get(q, f"{q*100:.0f}%パーセンタイル")
    format_dict[label] = format_oku
    formatted_df[label] = formatted_df[label].map(format_oku)

  for y in bankruptcy_years:
    format_dict[f"{y}年破産確率 (%)"] = format_pct
    formatted_df[f"{y}年破産確率 (%)"] = formatted_df[f"{y}年破産確率 (%)"].map(
        format_pct)

  styled_summary = summary_df.style.format(format_dict)
  styled_summary.index.name = "戦略"

  return formatted_df, styled_summary


def create_survival_probability_chart(
    results: Dict[str, SimulationResult],
    max_years: int = 50) -> Tuple[pd.DataFrame, alt.Chart]:
  """
  各戦略の生存確率 (1 - 破産確率) の推移を年単位で計算し、
  データフレームと Altairの折れ線グラフを返す。

  Args:
    results: 戦略名をキー、SimulationResult を値とする辞書。
    max_years: 何年後までを計算するか。デフォルトは50年。

  Returns:
    生データの DataFrame と Altair チャートのタプル。
  """
  plot_data = []
  years = list(range(max_years + 1))

  for name, res in results.items():
    sustained = res.sustained_months

    for y in years:
      # y年の時点で生存している = sustained_months >= y * 12
      survival_rate = np.mean(sustained >= y * 12) * 100.0
      plot_data.append({
          'Year': y,
          'Strategy': name,
          'Survival Probability (%)': survival_rate
      })

  df_plot = pd.DataFrame(plot_data)

  chart = alt.Chart(df_plot).mark_line(point=True).encode(
      x=alt.X('Year:Q', title='経過年数 (年)'),
      y=alt.Y('Survival Probability (%):Q',
              title='生存確率 (%)',
              scale=alt.Scale(domain=[0, 100])),
      color=alt.Color('Strategy:N', legend=alt.Legend(title="戦略")),
      tooltip=[
          'Year', 'Strategy',
          alt.Tooltip('Survival Probability (%):Q', format='.1f')
      ]
  ).properties(title='経過年数と生存確率の推移', width=600, height=250).interactive()

  return df_plot, chart


def visualize_and_save(results: Dict[str, SimulationResult],
                       html_file: str,
                       image_file: Optional[str] = None,
                       survival_image_file: Optional[str] = None,
                       title: str = '50年後の最終評価額のパーセンタイル分布',
                       summary_title: str = '最終評価額サマリー（1,000回試行）',
                       bankruptcy_years: List[int] = [20, 30, 40, 50]) -> None:
  """
  シミュレーション結果を可視化し、HTMLファイルに保存してブラウザで開く。
  オプションで画像ファイル（PNG/SVG等）としても保存する。
  マークダウンのサマリーテーブルを標準出力にプリントする。

  Args:
    results: 戦略名をキーとする SimulationResult の辞書
    html_file: 保存先のHTMLファイルパス
    image_file: オプションの保存先画像ファイルパス (例: .png, .svg) (最終評価額分布)
    survival_image_file: オプションの保存先画像ファイルパス (生存確率推移)
    title: グラフのタイトル
    summary_title: サマリー表のタイトル
    bankruptcy_years: サマリーに含める破産確率の年数リスト
  """
  # 可視化用に最終純資産額のみの DataFrame を作成
  df_results_net_values = pd.DataFrame({
      name: res.net_values for name, res in results.items()
  })

  # Altair用 Quantile データ作成
  quantiles = np.linspace(0, 1, 101)
  plot_data = []

  for q in quantiles:
    for col in df_results_net_values.columns:
      val = max(df_results_net_values[col].quantile(q), 1.0)  # 対数表示のため0以下は1に
      plot_data.append({
          'Quantile (%)': q * 100,
          'Strategy': col,
          'Final Value (万円)': val
      })

  df_plot = pd.DataFrame(plot_data)

  # Altair チャート描画（領域グラフ＋線グラフ）
  area_chart = alt.Chart(df_plot).mark_area(opacity=0.3).encode(
      x=alt.X('Quantile (%):Q', title='運の良さ (パーセンタイル %)'),
      y=alt.Y('Final Value (万円):Q',
              title='最終評価額(万円), 対数スケール',
              scale=alt.Scale(type='log'),
              stack=None),
      y2=alt.Y2(datum=1),
      color=alt.Color('Strategy:N', legend=alt.Legend(title="戦略")),
      tooltip=[
          'Quantile (%)', 'Strategy',
          alt.Tooltip('Final Value (万円):Q', format=',.0f')
      ])

  line_chart = alt.Chart(df_plot).mark_line(point=False).encode(
      x=alt.X('Quantile (%):Q'),
      y=alt.Y('Final Value (万円):Q', scale=alt.Scale(type='log')),
      color='Strategy:N')

  final_chart = (area_chart + line_chart).properties(title=title,
                                                     width=600,
                                                     height=300).interactive()

  # 生存確率のチャートを作成
  _, survival_chart = create_survival_probability_chart(results, max_years=50)

  # HTML表示用に垂直結合し、各グラフに凡例を独立して表示させる
  combined_chart = (final_chart &
                    survival_chart).resolve_legend(color='independent')

  # サマリーとHTMLの出力
  formatted_df, styled_summary = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=bankruptcy_years)

  # STDOUT にマークダウンを出力
  print(f"\n## {summary_title}")
  print(
      formatted_df.to_markdown(colalign=("left",) +
                               ("right",) * len(formatted_df.columns)))
  print("\n")

  # ensure temp directory exists
  os.makedirs(os.path.dirname(html_file), exist_ok=True)

  # 1. AltairのチャートをHTML文字列として取得
  chart_html = combined_chart.to_html()

  # 2. DataFrame(Styler)をHTMLのテーブル文字列として取得
  table_html = styled_summary.to_html()

  # 3. チャートのHTMLの<body>の先頭にテーブルのHTMLを挿入する
  style_tag = """
<style>
.summary-table table {border-collapse: collapse; font-family: sans-serif; margin-bottom: 30px;}
.summary-table th, .summary-table td {border: 1px solid #ddd; padding: 8px; text-align: right;}
.summary-table th {background-color: #f2f2f2; text-align: center;}
</style>
"""
  insert_html = f"""
<body>
<h2>{summary_title}</h2>
{style_tag}
<div class='summary-table'>
{table_html}
</div>
<hr>
"""
  full_html = chart_html.replace('<body>', insert_html)

  # 4. ファイルに保存してブラウザで開く
  with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

  print(f"✅ 結果を {html_file} に保存しました。")

  if image_file:
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    final_chart.save(image_file)
    print(f"✅ グラフを {image_file} に保存しました。")

  if survival_image_file:
    os.makedirs(os.path.dirname(survival_image_file), exist_ok=True)
    survival_chart.save(survival_image_file)
    print(f"✅ グラフを {survival_image_file} に保存しました。")

  print("🌐 ブラウザで開いています...")

  # file:// URIを作成してブラウザで開く
  abs_path = os.path.abspath(html_file)
  webbrowser.open('file://' + abs_path)
