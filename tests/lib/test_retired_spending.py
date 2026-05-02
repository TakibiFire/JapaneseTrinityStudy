import numpy as np
import pytest

from src.lib.retired_spending import (AVERAGE_AGE_75PLUS, SpendingType,
                                      calculate_average_age_75plus,
                                      get_annual_retired_spending_multipliers,
                                      get_annual_retired_spending_values)


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
  m_non = get_annual_retired_spending_multipliers([SpendingType.NON_CONSUMPTION],
                                                  start_age=40,
                                                  num_years=10)
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

  # SINGLE_2019_CONSUMPTION
  m_single = get_annual_retired_spending_multipliers(
      [SpendingType.SINGLE_2019_CONSUMPTION], start_age=30, num_years=5)
  assert len(m_single) == 5
  assert m_single[0] == pytest.approx(1.0)

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


def test_single_2019_values():
  # 2019年単身世帯データポイントの検証 (万円/年)
  # 年齢 25歳: 168,552 * 12 / 10000 = 202.2624
  # 年齢 55歳: 283,725 * 12 / 10000 = 340.47
  # 年齢 35歳: 222,432 * 12 / 10000 = 266.9184
  vals = get_annual_retired_spending_values(
      [SpendingType.SINGLE_2019_CONSUMPTION],
      start_age=25,
      num_years=31)  # 25から55まで

  assert vals[0] == pytest.approx(202.2624)
  assert vals[30] == pytest.approx(340.47)

  vals_35 = get_annual_retired_spending_values(
      [SpendingType.SINGLE_2019_CONSUMPTION], start_age=35, num_years=1)
  assert vals_35[0] == pytest.approx(266.9184)
