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

  # Multiple types (replacing BOTH)
  m_both = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION],
      start_age=50,
      num_years=20)
  assert len(m_both) == 20
  assert m_both[0] == 1.0
