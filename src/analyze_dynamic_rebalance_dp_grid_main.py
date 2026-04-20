"""
data/dynamic_rebalance_dp/dp_comp.csv の結果を分析・可視化するスクリプト。

内容:
1. 支出額のパーセンタイル推移可視化 (25p, 50p, 75p)
2. 各支出率 (Spending Rule) における生存確率の比較 (Fixed Ratio, V1 Dynamic, V2 DP)

Example:
  python src/analyze_dynamic_rebalance_dp_grid_main.py
"""

import os
from typing import Dict, List

import altair as alt
import pandas as pd

from src.lib.visualize import create_survival_probability_chart
from src.lib.visualize_all_yr import create_spend_percentile_chart

# 設定
DATA_PATH = "data/dynamic_rebalance_dp/dp_comp.csv"
IMG_DIR = "docs/imgs/dynamic_rebalance_dp"


def run_survival_analysis(df_survival: pd.DataFrame):
  """
  生存確率グラフの生成。
  各支出率 (spending_rule) ごとに、3つの戦略を比較するグラフを作成する。
  """
  print(f"\n\n{'='*20} 生存確率グラフを生成中... {'='*20}")

  rules = sorted(df_survival["spending_rule"].unique())

  for rule in rules:
    mask = (df_survival["spending_rule"] == rule)
    df_plot = df_survival[mask]

    if df_plot.empty:
      continue

    # create_survival_probability_chart は Dict[str, SimulationResult] を想定しているが、
    # 内部で res.sustained_months を参照しているだけなので、
    # 既に集計済みのデータ（csvの1, 2, 3...列）から DataFrame を再構成して Altair で直接描画するか、
    # あるいは簡易的な描画関数をここで作成する。
    # ここでは Altair で直接描画する。

    year_cols = [str(i) for i in range(1, 56) if str(i) in df_plot.columns]
    id_vars = ["strategy", "spending_rule"]
    
    df_long = df_plot.melt(id_vars=id_vars,
                           value_vars=year_cols,
                           var_name="Year",
                           value_name="Survival Probability (%)")
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Survival Probability (%)"] *= 100.0

    # 戦略名の順序を固定
    strategy_order = ["固定最適比率", "ダイナミック最適比率 (V1)", "Dynamic Rebalance DP (V2)"]

    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X('Year:Q', title='経過年数 (年)'),
        y=alt.Y('Survival Probability (%):Q',
                title='生存確率 (%)',
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('strategy:N',
                        sort=strategy_order,
                        legend=alt.Legend(title="戦略", orient='top')),
        tooltip=['Year', 'strategy', alt.Tooltip('Survival Probability (%):Q', format='.1f')]
    ).properties(
        title=f"生存確率の比較: 初期支出率 {rule:g}%",
        width=600,
        height=300
    )

    output_name = f"survival_comp_rule_{rule:g}.svg"
    output_path = os.path.join(IMG_DIR, output_name)
    os.makedirs(IMG_DIR, exist_ok=True)
    chart.save(output_path)
    print(f"✅ {output_path} に保存しました。")


def run_percentile_analysis(df_all: pd.DataFrame):
  """
  支出額パーセンタイル推移の生成。
  """
  print(f"\n\n{'='*20} 支出額パーセンタイル推移グラフを生成中... {'='*20}")

  # data/dynamic_rebalance_dp/dp_comp.csv には rule=4.0 & V2 DP のみ支出データが含まれている
  mask = (df_all["value_type"].isin(["spend25p", "spend50p", "spend75p"]))
  df_plot = df_all[mask]

  if df_plot.empty:
    print("支出データが見つかりませんでした。")
    return

  # 戦略ごとにグラフを作成（現状は V2 DP のみ）
  strategies = df_plot["strategy"].unique()
  for strat in strategies:
    strat_df = df_plot[df_plot["strategy"] == strat]
    rule = strat_df["spending_rule"].iloc[0]
    
    title = f"年間支出額推移: {strat}, 初期支出率 {rule:g}%"
    output_name = f"spend_percentiles_{strat}_rule_{rule:g}.svg"
    # ファイル名に使用できない文字を置換
    output_name = output_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_path = os.path.join(IMG_DIR, output_name)

    # create_spend_percentile_chart を呼び出す
    # start_age は src/dynamic_rebalance_dp_grid_main.py に合わせ 40 としている
    create_spend_percentile_chart(strat_df,
                                  title,
                                  output_path,
                                  start_age=40,
                                  num_years=55)


def main():
  if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} が見つかりません。")
    return

  df_all = pd.read_csv(DATA_PATH)
  
  # 1. 生存確率分析
  df_survival = df_all[df_all["value_type"] == "survival"].copy()
  run_survival_analysis(df_survival)

  # 2. 支出額パーセンタイル分析
  run_percentile_analysis(df_all)


if __name__ == "__main__":
  main()
