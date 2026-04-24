import math

import pytest

from src.lib.retired_spending import (AVERAGE_AGE_75PLUS, SpendingType,
                                      calculate_average_age_75plus,
                                      get_retired_spending_multipliers)


def test_average_age_consistency():
  calculated = calculate_average_age_75plus()
  print(f"Calculated: {calculated}")
  assert AVERAGE_AGE_75PLUS == pytest.approx(calculated)


def test_multipliers():
  # CONSUMPTION
  m_con = get_retired_spending_multipliers([SpendingType.CONSUMPTION],
                                           start_age=30,
                                           num_years=5)
  assert len(m_con) == 5
  assert m_con[0] == 1.0

  # NON_CONSUMPTION
  m_non = get_retired_spending_multipliers([SpendingType.NON_CONSUMPTION],
                                           start_age=40,
                                           num_years=10)
  assert len(m_non) == 10
  assert m_non[0] == 1.0

  # NON_CONSUMPTION_EXCLUDE_PENSION
  m_ex = get_retired_spending_multipliers(
      [SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION], start_age=30, num_years=5)
  assert len(m_ex) == 5
  assert m_ex[0] == 1.0
  # Verify difference between standard and excluded (at age 30)
  m_non_30 = get_retired_spending_multipliers([SpendingType.NON_CONSUMPTION],
                                              start_age=30,
                                              num_years=5)
  assert m_ex != m_non_30

  # SINGLE_2019_CONSUMPTION
  m_single = get_retired_spending_multipliers([SpendingType.SINGLE_2019_CONSUMPTION],
                                              start_age=30,
                                              num_years=5)
  assert len(m_single) == 5
  assert m_single[0] == 1.0

  # Multiple types (replacing BOTH)
  m_both = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION],
      start_age=50,
      num_years=20)
  assert len(m_both) == 20
  assert m_both[0] == 1.0


def test_non_normalized_values():
  # 開始年齢 34.4歳付近の月額支出を確認。BASE_AGES[0] = 34.4
  # 消費支出 280,544, 非消費支出(年金除) 90,018 - 38,125 = 51,893
  # 合計 332,437 付近になるはず。
  vals = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=34,
      num_years=1,
      normalize=False)
  assert len(vals) == 1
  assert 320000 < vals[0] < 340000

  # 開始年齢 35歳
  vals_35 = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=35,
      num_years=1,
      normalize=False)
  # 375,000 / month = 4,500,000 / year くらい
  # 実際には 330,000 ~ 380,000 程度
  assert 330000 < vals_35[0] < 380000

def test_single_2019_values():
  # 2019年単身世帯データポイントの検証
  # 年齢 25歳: 168,552
  # 年齢 55歳: 283,725
  vals = get_retired_spending_multipliers(
      [SpendingType.SINGLE_2019_CONSUMPTION],
      start_age=25,
      num_years=31,  # 25から55まで
      normalize=False)
  
  assert vals[0] == pytest.approx(168552)
  assert vals[30] == pytest.approx(283725)
  
  # 30代 (35歳) の値
  vals_35 = get_retired_spending_multipliers(
      [SpendingType.SINGLE_2019_CONSUMPTION],
      start_age=35,
      num_years=1,
      normalize=False)
  assert vals_35[0] == pytest.approx(222432)
