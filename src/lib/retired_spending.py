"""
年齢階級別の支出データに基づき、各年齢の支出倍率を計算するライブラリ。
"""
from enum import Enum, auto
from typing import List

import numpy as np
from scipy.interpolate import CubicSpline

from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES

# 生命表に基づき推計された75歳以上の平均年齢 (calculate_average_age_75plus() の結果)
AVERAGE_AGE_75PLUS = 83.88316897574647


class SpendingType(Enum):
  """支出の種類を表す列挙型。"""
  CONSUMPTION = auto()  # 消費支出 (生活費)
  NON_CONSUMPTION = auto()  # 非消費支出 (税・保険料)
  BOTH = auto()  # 合計 (消費支出 + 非消費支出)


def calculate_average_age_75plus() -> float:
  """
  生命表データを用いて75歳以上の平均年齢を推計する。
  """
  m_surv = [1.0]
  f_surv = [1.0]
  for m in MALE_MORTALITY_RATES:
    m_surv.append(m_surv[-1] * (1 - m))
  for f in FEMALE_MORTALITY_RATES:
    f_surv.append(f_surv[-1] * (1 - f))

  pop_sum = 0.0
  age_sum = 0.0
  # 75歳から105歳まで
  for x in range(75, len(MALE_MORTALITY_RATES)):
    pop = (m_surv[x] + f_surv[x]) / 2.0
    pop_sum += pop
    age_sum += pop * (x + 0.5)

  return age_sum / pop_sum


def get_retired_spending_multipliers(spending_type: SpendingType,
                                     start_age: int,
                                     num_years: int = 50) -> List[float]:
  """
  指定された開始年齢からの支出の倍率（開始年齢時を1.0とする）を返す。

  Args:
    spending_type: 支出の種類 (CONSUMPTION, NON_CONSUMPTION, BOTH)
    start_age: 開始年齢
    num_years: 取得する年数 (デフォルト 50)

  Returns:
    開始年齢時を1.0とした相対的な支出のリスト。
  """
  # 家計調査報告のデータポイント
  # https://www.stat.go.jp/data/kakei/sokuhou/tsuki/pdf/fies_gaikyo2024.pdf
  base_ages = np.array([34.4, 44.8, 54.1, 67.5, 72.5, AVERAGE_AGE_75PLUS])
  non_consumption_data = np.array([90018, 129607, 141647, 41405, 34824, 30558])
  consumption_data = np.array([280544, 331526, 359951, 311281, 269015, 242840])

  # 仮想データポイントの追加 (端部の安定化)
  last_age = base_ages[-1]
  virtual_age = last_age + (last_age - base_ages[-2])
  virtual_non_con = non_consumption_data[-1] * 0.9
  virtual_con = consumption_data[-1] * 0.9

  ages = np.append(base_ages, virtual_age)
  non_con_full = np.append(non_consumption_data, virtual_non_con)
  con_full = np.append(consumption_data, virtual_con)

  # スプライン補間
  cs_non_con = CubicSpline(ages, non_con_full, bc_type='natural')
  cs_con = CubicSpline(ages, con_full, bc_type='natural')

  target_ages = np.arange(start_age, start_age + num_years)

  if spending_type == SpendingType.CONSUMPTION:
    values = np.maximum(cs_con(target_ages), virtual_con)
  elif spending_type == SpendingType.NON_CONSUMPTION:
    values = np.maximum(cs_non_con(target_ages), virtual_non_con)
  else:  # BOTH
    v_con = np.maximum(cs_con(target_ages), virtual_con)
    v_non_con = np.maximum(cs_non_con(target_ages), virtual_non_con)
    values = v_con + v_non_con

  # 開始年齢の値を 1.0 とした倍率を返す
  return (values / values[0]).tolist()
