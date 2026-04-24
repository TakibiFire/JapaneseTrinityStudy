"""
年齢階級別の消費支出および非消費支出データからスプライン補間を用いて各年齢の支出を推計する。
"""
import os

import altair as alt
import numpy as np
import pandas as pd

from src.lib.retired_spending import (BASE_AGES, CONSUMPTION_DATA,
                                      NON_CONSUMPTION_DATA,
                                      SINGLE_2019_BASE_AGES,
                                      SINGLE_2019_CONSUMPTION_DATA,
                                      SpendingType,
                                      get_retired_spending_values)


def analyze_single_spending():
  """
  単身世帯の2019年データに基づく支出推移を可視化する。
  """
  target_ages = np.arange(20, 101)
  target_con = get_retired_spending_values([SpendingType.SINGLE_2019_CONSUMPTION],
                                           target_ages)

  df_actual = pd.DataFrame({
      "Age": SINGLE_2019_BASE_AGES,
      "Value": SINGLE_2019_CONSUMPTION_DATA,
      "支出種別": "消費支出 (2019年単身世帯)",
      "データ": "実績"
  })

  df_spline = pd.DataFrame({
      "Age": target_ages,
      "Value": target_con,
      "支出種別": "消費支出 (2019年単身世帯)",
      "データ": "推計"
  })

  chart_actual = alt.Chart(df_actual).mark_point(size=60, filled=True).encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[20, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=alt.value("#1f77b4"))

  chart_spline = alt.Chart(df_spline).mark_line().encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[20, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=alt.value("#1f77b4"))

  chart = (chart_actual + chart_spline).properties(
      width=600, height=350, title="単身世帯の年齢別消費支出推移（2019年全国家計構造調査）")

  output_path = "docs/imgs/retired_spending/cost_by_age_single.svg"
  chart.save(output_path)
  print(f"単身世帯のグラフを保存しました: {output_path}")


def main():
  # 推計対象の年齢 (30歳〜100歳)
  target_ages = np.arange(30, 101)

  # 各支出種別の推計値を計算
  target_con = get_retired_spending_values([SpendingType.CONSUMPTION],
                                           target_ages)
  target_non_con = get_retired_spending_values([SpendingType.NON_CONSUMPTION],
                                               target_ages)
  target_non_con_ex = get_retired_spending_values(
      [SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION], target_ages)

  # 結果をSTDOUTに出力
  print("年齢, 推計消費支出(円), 推計非消費支出(円), 推計非消費支出(年金除)(円), 合計支出(円)")
  for i, age in enumerate(target_ages):
    print(f"{age}, {int(target_con[i])}, {int(target_non_con[i])}, "
          f"{int(target_non_con_ex[i])}, "
          f"{int(target_con[i] + target_non_con[i])}")

  # グラフ描画用データフレーム作成
  df_actual = pd.concat([
      pd.DataFrame({
          "Age": BASE_AGES,
          "Value": CONSUMPTION_DATA,
          "支出種別": "消費支出 (生活費)",
          "データ": "実績"
      }),
      pd.DataFrame({
          "Age": BASE_AGES,
          "Value": NON_CONSUMPTION_DATA,
          "支出種別": "非消費支出 (税・保険料)",
          "データ": "実績"
      })
  ])

  df_spline = pd.concat([
      pd.DataFrame({
          "Age": target_ages,
          "Value": target_con,
          "支出種別": "消費支出 (生活費)",
          "データ": "推計"
      }),
      pd.DataFrame({
          "Age": target_ages,
          "Value": target_non_con,
          "支出種別": "非消費支出 (税・保険料)",
          "データ": "推計"
      }),
      pd.DataFrame({
          "Age": target_ages,
          "Value": target_non_con_ex,
          "支出種別": "非消費支出 (年金を除く)",
          "データ": "推計"
      }),
      pd.DataFrame({
          "Age": target_ages,
          "Value": target_con + target_non_con,
          "支出種別": "合計支出",
          "データ": "推計"
      })
  ])

  # グラフ描画
  color_scale = alt.Color("支出種別:N",
                          scale=alt.Scale(domain=[
                              "消費支出 (生活費)", "非消費支出 (税・保険料)", "合計支出", "非消費支出 (年金を除く)"
                          ],
                                          range=[
                                              "#1f77b4", "#ff7f0e", "#2ca02c",
                                              "#ff7f0e"
                                          ]),
                          legend=alt.Legend(orient='top'))

  chart_actual = alt.Chart(df_actual).mark_point(size=60, filled=True).encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=color_scale)

  chart_spline = alt.Chart(df_spline).mark_line().encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=color_scale,
      strokeDash=alt.condition(
          (alt.datum.支出種別 == "合計支出") |
          (alt.datum.支出種別 == "非消費支出 (年金を除く)"), alt.value([5, 5]),
          alt.value([0, 0])))

  chart = (chart_actual + chart_spline).properties(width=600,
                                                   height=400,
                                                   title="年齢別支出（消費・非消費）の推移")

  output_path = "docs/imgs/retired_spending/cost_by_age.svg"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")

  # 単身世帯の分析を追加
  analyze_single_spending()


if __name__ == '__main__':
  main()
