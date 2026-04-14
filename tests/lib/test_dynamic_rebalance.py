import numpy as np
import pytest

from src.lib.dynamic_rebalance import calculate_safe_target_ratio


def test_calculate_safe_target_ratio_default():
  # rem_years=35.0 で約 0.036332 になることを確認
  # (0.04 * (1 - 0.20315) - log(1 + 0.0177)) / (1 - exp(-delta * 35))
  # delta = 0.031874 - 0.017545 = 0.014329
  # s_safe = 0.014329 / (1 - exp(-0.014329 * 35)) = 0.014329 / 0.39439 = 0.036332
  assert calculate_safe_target_ratio(35.0) == pytest.approx(0.036332, abs=1e-5)


def test_calculate_safe_target_ratio_custom():
  # 10年、利回り5%、税金なし、インフレなし
  # delta = 0.05
  # s_safe = 0.05 / (1 - exp(-0.05 * 10)) = 0.05 / (1 - 0.60653) = 0.05 / 0.39347 = 0.12707
  ratio = calculate_safe_target_ratio(10.0,
                                      base_yield=0.05,
                                      tax_rate=0.0,
                                      inflation_rate=0.0)
  assert ratio == pytest.approx(0.12707, abs=1e-5)


def test_calculate_safe_target_ratio_long_term():
  # 非常に長い期間では実質利回りに近づくはず
  # delta = 0.014329
  ratio = calculate_safe_target_ratio(10000.0)
  # s_safe = delta / (1 - exp(-delta * 10000)) approx delta
  assert ratio == pytest.approx(0.014329, abs=1e-6)
