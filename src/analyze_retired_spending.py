"""
年齢階級別の消費支出および非消費支出データからスプライン補間を用いて各年齢の支出を推計する。
"""
import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES


def calculate_average_age_75plus(male_rates: List[float], female_rates: List[float]) -> float:
  """
  生命表データを用いて75歳以上の平均年齢を推計する。
  """
  m_surv = [1.0]
  f_surv = [1.0]
  for m in male_rates:
    m_surv.append(m_surv[-1] * (1 - m))
  for f in female_rates:
    f_surv.append(f_surv[-1] * (1 - f))

  pop_sum = 0.0
  age_sum = 0.0
  # 75歳から105歳（生命表の終端）まで
  for x in range(75, len(male_rates)):
    # 男女平均の生存数に比例すると仮定
    pop = (m_surv[x] + f_surv[x]) / 2.0
    pop_sum += pop
    age_sum += pop * (x + 0.5)

  return age_sum / pop_sum


def main():
  # 75歳以上の平均年齢を計算
  age_75plus = calculate_average_age_75plus(MALE_MORTALITY_RATES,
                                            FEMALE_MORTALITY_RATES)
  print(f"推計された75歳以上の平均年齢: {age_75plus:.1f}")

  # 家計調査報告のデータ (2024年平均結果の概要)
  # 世帯主の年齢, 非消費支出, 消費支出
  # 勤労世帯 (<65歳想定)
  # 34.4, 90018, 280544
  # 44.8, 129607, 331526
  # 54.1, 141647, 359951
  # 無職世帯 (>=65歳想定)
  # 67.5 (65-69歳), 41405, 311281
  # 72.5 (70-74歳), 34824, 269015
  # age_75plus, 30558, 242840

  base_ages = np.array([34.4, 44.8, 54.1, 67.5, 72.5, age_75plus])
  non_consumption_data = np.array([90018, 129607, 141647, 41405, 34824, 30558])
  consumption_data = np.array([280544, 331526, 359951, 311281, 269015, 242840])

  # 仮想データポイントの追加 (スプラインの端部を安定させるため)
  # 最後のデータポイントを軸に対称な位置に仮想点を追加
  last_age = base_ages[-1]
  prev_age = base_ages[-2]
  virtual_age = last_age + (last_age - prev_age)
  
  # 仮想点での支出は、最後の値から少し下がる程度でクリップされるように設定
  # ここでは最後の値と同じにすることで、端部での急激な変化を抑制する
  virtual_non_con = non_consumption_data[-1] * 0.9
  virtual_con = consumption_data[-1] * 0.9

  ages = np.append(base_ages, virtual_age)
  non_consumption = np.append(non_consumption_data, virtual_non_con)
  consumption = np.append(consumption_data, virtual_con)

  # 3次スプライン補間 (自然スプライン)
  cs_non_con = CubicSpline(ages, non_consumption, bc_type='natural')
  cs_con = CubicSpline(ages, consumption, bc_type='natural')

  # 推計対象の年齢 (30歳〜100歳)
  target_ages = np.arange(30, 101)
  target_non_con = cs_non_con(target_ages)
  target_con = cs_con(target_ages)

  # 保守的な下限クリップ (最後のデータポイントの90%を下限とする)
  target_non_con = np.maximum(target_non_con, virtual_non_con)
  target_con = np.maximum(target_con, virtual_con)

  # 結果をSTDOUTに出力
  print("年齢, 推計消費支出(円), 推計非消費支出(円), 合計支出(円)")
  for i, age in enumerate(target_ages):
    print(f"{age}, {int(target_con[i])}, {int(target_non_con[i])}, "
          f"{int(target_con[i] + target_non_con[i])}")

  # グラフ描画用データフレーム作成
  df_actual = pd.concat([
      pd.DataFrame({
          "Age": base_ages,
          "Value": consumption_data,
          "支出種別": "消費支出 (生活費)",
          "データ": "実績"
      }),
      pd.DataFrame({
          "Age": base_ages,
          "Value": non_consumption_data,
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
          "Value": target_con + target_non_con,
          "支出種別": "合計支出",
          "データ": "推計"
      })
  ])

  # グラフ描画
  color_scale = alt.Color("支出種別:N",
                          scale=alt.Scale(domain=[
                              "消費支出 (生活費)", "非消費支出 (税・保険料)", "合計支出"
                          ],
                                          range=[
                                              "#1f77b4", "#ff7f0e", "#2ca02c"
                                          ]))

  chart_actual = alt.Chart(df_actual).mark_point(size=60, filled=True).encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=color_scale)

  chart_spline = alt.Chart(df_spline).mark_line().encode(
      x=alt.X("Age:Q", title="年齢", scale=alt.Scale(domain=[30, 100])),
      y=alt.Y("Value:Q", title="支出(円)"),
      color=color_scale,
      strokeDash=alt.condition(alt.datum.支出種別 == "合計支出",
                               alt.value([5, 5]), alt.value([0, 0])))

  chart = (chart_actual + chart_spline).properties(width=600,
                                                   height=400,
                                                   title="年齢別支出（消費・非消費）の推移")

  output_path = "docs/imgs/retired_spending/cost_by_age.svg"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")


if __name__ == '__main__':
  main()
