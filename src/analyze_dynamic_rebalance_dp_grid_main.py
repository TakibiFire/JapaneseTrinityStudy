"""
data/dynamic_rebalance_dp/dp_comp.csv の結果を分析・可視化するスクリプト。

内容:
1. 取り崩し額のパーセンタイル推移可視化 (25p, 50p, 75p)
2. 各支出率 (Spending Rule) における生存確率の比較 (オルカン100%, 固定最適, 一般的, 支出に合わせた)

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
WITHDRAW_DATA_PATH = "data/dynamic_rebalance_dp/dump_withdraw.csv"
IMG_DIR = "docs/imgs/dynamic_rebalance_dp"
START_AGE = 40
NUM_YEARS = 55


def run_survival_analysis(df_survival: pd.DataFrame):
  """
  生存確率グラフの生成。
  各支出率 (spending_rule) ごとに、複数の戦略を比較するグラフを作成する。
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

    year_cols = [str(i) for i in range(1, NUM_YEARS + 1) if str(i) in df_plot.columns]
    id_vars = ["strategy", "spending_rule"]

    df_long = df_plot.melt(id_vars=id_vars,
                           value_vars=year_cols,
                           var_name="Year",
                           value_name="Survival Probability (%)")
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Age"] = df_long["Year"] + START_AGE
    df_long["Survival Probability (%)"] *= 100.0

    # 戦略名の順序を固定
    strategy_order = [
        "オルカン100%", "無リスク100%", "固定最適比率", "一般的な最適リバランス", "支出に合わせた最適リバランス"
    ]

    # y軸の下限を、最小値の10の倍数（切り捨て）に設定
    min_val = df_long['Survival Probability (%)'].min()
    y_min = (min_val // 10) * 10
    y_max = 100

    title = f"生存確率の比較: 初期支出率 {rule:g}%"
    if y_min > 0:
      title += f"（生存確率 {y_min:.0f}%以下は描画を省略）"

    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X('Age:Q', title='年齢'),
        y=alt.Y('Survival Probability (%):Q',
                title='生存確率 (%)',
                scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color('strategy:N',
                        sort=strategy_order,
                        legend=alt.Legend(title="戦略", orient='top')),
        tooltip=[
            'Age', 'strategy',
            alt.Tooltip('Survival Probability (%):Q', format='.1f')
        ]).properties(title=title, width=600, height=300)

    output_name = f"survival_comp_rule_{rule:g}.svg"
    output_path = os.path.join(IMG_DIR, output_name)
    os.makedirs(IMG_DIR, exist_ok=True)
    chart.save(output_path)
    print(f"✅ {output_path} に保存しました。")


def run_percentile_analysis(df_all: pd.DataFrame):
  """
  取り崩し額パーセンタイル推移の生成。
  """
  print(f"\n\n{'='*20} 取り崩し額パーセンタイル推移グラフを生成中... {'='*20}")

  # 支出データ (25p, 50p, 75p) を抽出
  mask = (df_all["value_type"].isin(["spend25p", "spend50p", "spend75p"]))
  df_plot = df_all[mask]

  if df_plot.empty:
    print("支出データが見つかりませんでした。")
    return

  title = "年間取り崩し額推移"
  output_name = "spend_percentiles_dump_withdraw.svg"
  output_path = os.path.join(IMG_DIR, output_name)

  # create_spend_percentile_chart を呼び出す
  create_spend_percentile_chart(df_plot,
                                title,
                                output_path,
                                start_age=START_AGE,
                                num_years=NUM_YEARS,
                                show_legend=False)


def main():
  # 1. dp_comp.csv の分析 (生存確率など)
  if os.path.exists(DATA_PATH):
    df_all = pd.read_csv(DATA_PATH)
    df_survival = df_all[df_all["value_type"] == "survival"].copy()
    #run_survival_analysis(df_survival)
    # 元のファイルに支出データが含まれている場合の互換性維持
    #run_percentile_analysis(df_all)

  # 2. dump_withdraw.csv の分析
  if os.path.exists(WITHDRAW_DATA_PATH):
    df_dump = pd.read_csv(WITHDRAW_DATA_PATH)
    run_percentile_analysis(df_dump)


if __name__ == "__main__":
  main()
