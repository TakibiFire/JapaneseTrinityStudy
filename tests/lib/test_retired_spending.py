import numpy as np
import pytest

from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES
from src.lib.retired_spending import (AVERAGE_AGE_75PLUS, SpendingType,
                                      get_annual_retired_spending_multipliers,
                                      get_annual_retired_spending_values)


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


def test_average_age_consistency():
  calculated = calculate_average_age_75plus()
  assert AVERAGE_AGE_75PLUS == pytest.approx(calculated)


def test_multipliers():
  # CONSUMPTION
  m_con = get_annual_retired_spending_multipliers([SpendingType.CONSUMPTION],
                                                  start_age=30,
                                                  num_years=5)
  assert len(m_con) == 5
  assert m_con[0] == pytest.approx(1.0)

  # NON_CONSUMPTION
  m_non = get_annual_retired_spending_multipliers(
      [SpendingType.NON_CONSUMPTION], start_age=40, num_years=10)
  assert len(m_non) == 10
  assert m_non[0] == pytest.approx(1.0)

  # NON_CONSUMPTION_EXCLUDE_PENSION
  m_ex = get_annual_retired_spending_multipliers(
      [SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION], start_age=30, num_years=5)
  assert len(m_ex) == 5
  assert m_ex[0] == pytest.approx(1.0)
  # Verify difference between standard and excluded (at age 30)
  m_non_30 = get_annual_retired_spending_multipliers(
      [SpendingType.NON_CONSUMPTION], start_age=30, num_years=5)
  assert not np.array_equal(m_ex, m_non_30)

  # ALL_HOUSEHOLDS_2019_CONSUMPTION
  m_all = get_annual_retired_spending_multipliers(
      [SpendingType.ALL_HOUSEHOLDS_2019_CONSUMPTION], start_age=30, num_years=5)
  assert len(m_all) == 5
  assert m_all[0] == pytest.approx(1.0)

  # Multiple types
  m_both = get_annual_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION],
      start_age=50,
      num_years=20)
  assert len(m_both) == 20
  assert m_both[0] == pytest.approx(1.0)


def test_annual_values():
  # 開始年齢 34.4歳付近の年額支出を確認。
  # 月額: 消費支出 280,544, 非消費支出(年金除) 90,018 - 38,125 = 51,893
  # 月額合計: 332,437
  # 年額合計 (万円): 332,437 * 12 / 10000 = 398.9244
  vals = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=34,
      num_years=1)
  assert len(vals) == 1
  # 390万円 ~ 410万円 程度であることを確認
  assert 390 < vals[0] < 410

  # 開始年齢 35歳
  vals_35 = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=35,
      num_years=1)
  # 400万円 ~ 450万円 程度であることを確認
  assert 400 < vals_35[0] < 450


def test_all_households_2019_values():
  # 2019年総世帯データポイントの検証 (万円/年)
  # 年齢 25歳: 168,552 * 12 / 10000 = 202.2624
  # 年齢 55歳: 283,725 * 12 / 10000 = 340.47
  # 年齢 35歳: 222,432 * 12 / 10000 = 266.9184
  vals = get_annual_retired_spending_values(
      [SpendingType.ALL_HOUSEHOLDS_2019_CONSUMPTION],
      start_age=25,
      num_years=31)  # 25から55まで

  assert vals[0] == pytest.approx(202.2624)
  assert vals[30] == pytest.approx(340.47)

  vals_35 = get_annual_retired_spending_values(
      [SpendingType.ALL_HOUSEHOLDS_2019_CONSUMPTION], start_age=35, num_years=1)
  assert vals_35[0] == pytest.approx(266.9184)


def test_single_2025_values():
  # 2025年単身世帯データポイントの検証 (万円/年)
  # 年齢 30.0: 177,542 * 12 / 10000 = 213.0504
  # 年齢 47.5: 198,488 * 12 / 10000 = 238.1856
  # 年齢 62.5: 179,933 * 12 / 10000 = 215.9196
  # 年齢 75.0: 155,782 * 12 / 10000 = 186.9384

  # 整数年齢でチェック
  vals = get_annual_retired_spending_values(
      [SpendingType.SINGLE_2025_CONSUMPTION], start_age=30, num_years=46)

  assert vals[0] == pytest.approx(213.0504)
  # 75歳の値を確認
  assert vals[45] == pytest.approx(186.9384)

  # 非消費支出 (高齢単身無職世帯)
  vals_non = get_annual_retired_spending_values(
      [SpendingType.UNEMPLOYED_SINGLE_2025_NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=65,
      num_years=1)
  assert vals_non[0] == pytest.approx(12930.0 * 12.0 / 10000.0)
