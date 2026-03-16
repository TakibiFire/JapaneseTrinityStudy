import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def main():
  """
  年齢階級別の消費支出データからスプライン補間を用いて各年齢の消費支出を推計する。
  """
  # 家計調査報告のデータ
  # 世帯主の年齢階級, 世帯主の年齢, 消費支出(円)
  # 40歳未満 34.4 280,451
  # 40～49歳 44.8 331,134
  # 50～59歳 54.2 356,946
  # 60～69歳 64.6 311,392
  # 70歳以上 77.6 252,781

  # 70歳以上のデータ(平均77.6歳、252,781円)について、70歳時点の近似値(284,506円)と
  # バランスを取るため、77.6歳を中間点とした対称な位置(85.2歳)に
  # 対称な生活費(221,056円)の仮想的なデータポイントを追加して不自然な下降を防ぐ。
  ages = np.array([34.4, 44.8, 54.2, 64.6, 77.6, 85.2])
  costs = np.array([280451, 331134, 356946, 311392, 252781, 221056])

  # 3次スプライン補間 (自然スプライン)
  cs = CubicSpline(ages, costs, bc_type='natural')

  # 推計対象の年齢 (30歳〜100歳)
  target_ages = np.arange(30, 101)
  target_costs = cs(target_ages)

  # 85.2歳付近の221,056円で保守的に下限クリップする
  target_costs = np.maximum(target_costs, 221056)

  # 結果をSTDOUTに出力
  print("年齢, 推計消費支出(円)")
  for age, cost in zip(target_ages, target_costs):
    print(f"{age}, {int(cost)}")

  # グラフ描画用データフレーム作成
  df_actual = pd.DataFrame({
      "Age": ages[:-1], # 最後の仮想データポイント(85.2歳)はオリジナルデータではないのでプロットから除外
      "Cost": costs[:-1],
      "Type": ["Original Data"] * (len(ages) - 1)
  })

  df_spline = pd.DataFrame({
      "Age": target_ages,
      "Cost": target_costs,
      "Type": ["Cubic Spline"] * len(target_ages)
  })

  # オリジナルデータはポイントで、スプライン補間はラインで描画
  chart_actual = alt.Chart(df_actual).mark_point(size=100, filled=True).encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Cost:Q", title="消費支出(円)"),
      color=alt.Color("Type:N", title="データ種類")
  )

  chart_spline = alt.Chart(df_spline).mark_line().encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Cost:Q", title="消費支出(円)"),
      color=alt.Color("Type:N", title="データ種類")
  )

  chart = (chart_actual + chart_spline).properties(
      width=500,
      height=300,
      title="年齢別消費支出の推移"
  )

  output_path = "imgs/cost_by_age.svg"
  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")


if __name__ == '__main__':
  main()
