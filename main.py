"""
日本版トリニティ・スタディのシミュレーションを実行し、HTMLレポートを生成するスクリプト。

`core.py` の機能を利用して複数の投資戦略（オルカン100%やレバレッジ活用など）の
シミュレーションを比較し、最終純資産の分布や破産確率などを視覚化して `new_result.html` に出力する。
"""

import os
import webbrowser

import altair as alt
import numpy as np
import pandas as pd

from core import (Asset, Strategy, create_styled_summary,
                  generate_monthly_asset_prices, simulate_strategy)


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  assets = [
      Asset(name="オルカン", yearly_cost=0.05775 / 100, leverage=1),
      Asset(name="レバカン", yearly_cost=0.422 / 100, leverage=2)
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  plan_zero = Strategy(name="ZERO",
                       initial_money=10000,
                       initial_loan=0,
                       yearly_loan_interest=2.125 / 100,
                       initial_asset_ratio={},
                       annual_cost=400,
                       annual_cost_inflation=0.015,
                       selling_priority=[])

  plan_a = Strategy(name="A: オルカン100%",
                    initial_money=10000,
                    initial_loan=0,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={"オルカン": 1.0},
                    annual_cost=0,
                    annual_cost_inflation=0,
                    selling_priority=["オルカン"])

  plan_a_cost_4p = Strategy(name="オルカン100%, cost4%, inf1.5%",
                            initial_money=10000,
                            initial_loan=0,
                            yearly_loan_interest=2.125 / 100,
                            initial_asset_ratio={"オルカン": 1.0},
                            annual_cost=400,
                            annual_cost_inflation=0.015,
                            selling_priority=["オルカン"])

  plan_a_80p_cost_4p = Strategy(name="オルカン80%, cost4%, inf1.5%",
                                initial_money=10000,
                                initial_loan=0,
                                yearly_loan_interest=2.125 / 100,
                                initial_asset_ratio={"オルカン": 0.8},
                                annual_cost=400,
                                annual_cost_inflation=0.015,
                                selling_priority=["オルカン"])

  plan_a_50p_cost_4p = Strategy(name="オルカン50%, cost4%, inf1.5%",
                                initial_money=10000,
                                initial_loan=0,
                                yearly_loan_interest=2.125 / 100,
                                initial_asset_ratio={"オルカン": 0.5},
                                annual_cost=400,
                                annual_cost_inflation=0.015,
                                selling_priority=["オルカン"])

  plan_opt = Strategy(name="Opt",
                      initial_money=10000,
                      initial_loan=3000,
                      yearly_loan_interest=2.125 / 100,
                      initial_asset_ratio={
                          "オルカン": 0.9,
                          "レバカン": 0.0
                      },
                      annual_cost=400,
                      annual_cost_inflation=0.015,
                      selling_priority=["オルカン", "レバカン"])

  plan_opt_reb1 = Strategy(name="Opt Rebalance 1",
                           initial_money=10000,
                           initial_loan=3000,
                           yearly_loan_interest=2.125 / 100,
                           initial_asset_ratio={
                               "オルカン": 0.9,
                               "レバカン": 0.0
                           },
                           annual_cost=400,
                           annual_cost_inflation=0.015,
                           selling_priority=["オルカン", "レバカン"],
                           rebalance_interval=1)

  plan_opt_reb12 = Strategy(name="Opt Rebalance 12",
                            initial_money=10000,
                            initial_loan=3000,
                            yearly_loan_interest=2.125 / 100,
                            initial_asset_ratio={
                                "オルカン": 0.9,
                                "レバカン": 0.0
                            },
                            annual_cost=400,
                            annual_cost_inflation=0.015,
                            selling_priority=["オルカン", "レバカン"],
                            rebalance_interval=12)

  plan_opt_reb180 = Strategy(name="Opt Rebalance 180",
                             initial_money=10000,
                             initial_loan=3000,
                             yearly_loan_interest=2.125 / 100,
                             initial_asset_ratio={
                                 "オルカン": 0.9,
                                 "レバカン": 0.0
                             },
                             annual_cost=400,
                             annual_cost_inflation=0.015,
                             selling_priority=["オルカン", "レバカン"],
                             rebalance_interval=180)

  plan_b = Strategy(name="B: オルカン50%+レバカン50%, cost4%, inf1.5%",
                    initial_money=10000,
                    initial_loan=0,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={
                        "オルカン": 0.5,
                        "レバカン": 0.5
                    },
                    annual_cost=400,
                    annual_cost_inflation=0.015,
                    selling_priority=["オルカン", "レバカン"])

  plan_b_4_4 = Strategy(name="B: オルカン40%+レバカン40%, cost4%, inf1.5%",
                        initial_money=10000,
                        initial_loan=0,
                        yearly_loan_interest=2.125 / 100,
                        initial_asset_ratio={
                            "オルカン": 0.4,
                            "レバカン": 0.4
                        },
                        annual_cost=400,
                        annual_cost_inflation=0.015,
                        selling_priority=["オルカン", "レバカン"])

  plan_c = Strategy(name="C: 証券担保ローン1.5倍, cost4%, inf1.5%",
                    initial_money=10000,
                    initial_loan=5000,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={"オルカン": 1.0},
                    annual_cost=400,
                    annual_cost_inflation=0.015,
                    selling_priority=["オルカン", "レバカン"])

  opt50 = Strategy(name="Opt50",
                   initial_money=10000,
                   initial_loan=5000,
                   yearly_loan_interest=2.125 / 100,
                   initial_asset_ratio={
                       "オルカン": 0.0,
                       "レバカン": 0.9
                   },
                   annual_cost=400,
                   annual_cost_inflation=0.015,
                   selling_priority=["レバカン", "オルカン"],
                   rebalance_interval=0)

  strategies = [
      #plan_zero,
      #plan_a,
      plan_a_cost_4p,
      plan_a_80p_cost_4p,
      plan_a_50p_cost_4p,
      plan_opt,
      plan_opt_reb1,
      plan_opt_reb12,
      plan_opt_reb180,
      opt50,
      #plan_b,
      #plan_b_4_4,
      #plan_c,
  ]

  # ---------------------------------------------------------------------------
  # 3. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    net_values = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = net_values

  # DataFrame化
  df_results = pd.DataFrame(results)

  # ---------------------------------------------------------------------------
  # 4. 可視化 (Altair)
  # ---------------------------------------------------------------------------
  # Altair用 Quantile データ作成
  quantiles = np.linspace(0, 1, 101)
  plot_data = []

  for q in quantiles:
    for col in df_results.columns:
      val = max(df_results[col].quantile(q), 1.0)  # 対数表示のため0以下は1に
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
      title='30年後の最終評価額のパーセンタイル分布', width=600, height=300).interactive()

  # ---------------------------------------------------------------------------
  # 5. サマリーとHTMLの出力
  # ---------------------------------------------------------------------------
  styled_summary = create_styled_summary(df_results)

  html_file = 'temp/new_result.html'

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
  insert_html = f"<body>\n<h2>30年後の最終評価額サマリー（1,000回試行）</h2>\n{style_tag}\n{table_html}\n<hr>\n"
  full_html = chart_html.replace('<body>', insert_html)

  # 4. ファイルに保存してブラウザで開く
  with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

  print(f"✅ 結果を {html_file} に保存しました。")
  print("🌐 ブラウザで開いています...")

  # file:// URIを作成してブラウザで開く
  abs_path = os.path.abspath(html_file)
  webbrowser.open('file://' + abs_path)


if __name__ == "__main__":
  main()
