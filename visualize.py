"""
シミュレーション結果の可視化とHTML出力を行うユーティリティ。
"""

import os
import webbrowser
from typing import Dict, Optional

import altair as alt
import numpy as np
import pandas as pd

from core import SimulationResult, create_styled_summary


def visualize_and_save(results: Dict[str, SimulationResult],
                       html_file: str,
                       image_file: Optional[str] = None,
                       title: str = '50年後の最終評価額のパーセンタイル分布',
                       summary_title: str = '最終評価額サマリー（1,000回試行）') -> None:
  """
  シミュレーション結果を可視化し、HTMLファイルに保存してブラウザで開く。
  オプションで画像ファイル（PNG/SVG等）としても保存する。

  Args:
    results: 戦略名をキーとする SimulationResult の辞書
    html_file: 保存先のHTMLファイルパス
    image_file: オプションの保存先画像ファイルパス (例: .png, .svg)
    title: グラフのタイトル
    summary_title: サマリー表のタイトル
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

  final_chart = (area_chart + line_chart).properties(
      title=title, width=600, height=300).interactive()

  # サマリーとHTMLの出力
  styled_summary = create_styled_summary(results)

  # ensure temp directory exists
  os.makedirs(os.path.dirname(html_file), exist_ok=True)

  # 1. AltairのチャートをHTML文字列として取得
  chart_html = final_chart.to_html()

  # 2. DataFrame(Styler)をHTMLのテーブル文字列として取得
  table_html = styled_summary.to_html()

  # 3. チャートのHTMLの<body>の先頭にテーブルのHTMLを挿入する
  style_tag = """
<style>
table {border-collapse: collapse; font-family: sans-serif; margin-bottom: 30px;}
th, td {border: 1px solid #ddd; padding: 8px; text-align: right;}
th {background-color: #f2f2f2; text-align: center;}
</style>
"""
  insert_html = f"<body>\n<h2>{summary_title}</h2>\n{style_tag}\n{table_html}\n<hr>\n"
  full_html = chart_html.replace('<body>', insert_html)

  # 4. ファイルに保存してブラウザで開く
  with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

  print(f"✅ 結果を {html_file} に保存しました。")
  
  if image_file:
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    final_chart.save(image_file)
    print(f"✅ グラフを {image_file} に保存しました。")

  print("🌐 ブラウザで開いています...")

  # file:// URIを作成してブラウザで開く
  abs_path = os.path.abspath(html_file)
  webbrowser.open('file://' + abs_path)
