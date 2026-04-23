"""
SpendAwareDynamicSpending の単体テスト（pytest 使用）。
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.lib.spend_aware_dynamic_spending import SpendAwareDynamicSpending


@pytest.fixture
def base_strategy():
  # テスト用の共通セットアップ
  annual_cost_real = [400.0] * 100
  mock_predictor = MagicMock()
  strategy = SpendAwareDynamicSpending(initial_age=60,
                                       p_low=0.8,
                                       p_high=0.9,
                                       lower_mult=0.9,
                                       upper_mult=1.2,
                                       annual_cost_real=annual_cost_real,
                                       dp_predictor=mock_predictor)
  return strategy, mock_predictor


def test_no_adjustment(base_strategy):
  # ガードレール内（p_tgt が [0.8, 0.9] の範囲内）の場合、支出調整が行われないことを確認
  strategy, mock_predictor = base_strategy

  n_sim = 5
  m = 0
  active_paths = np.ones(n_sim, dtype=bool)

  # 入力が配列の場合、配列を返すようにする
  mock_predictor.predict_p_surv.side_effect = lambda age, s: np.full_like(
      s, 0.85)

  net_worth = np.full(n_sim, 10000.0)
  prev_base_spend_y = np.full(n_sim, 400.0)
  cpi_m = np.ones(n_sim)
  cpi_m_minus_12 = np.ones(n_sim)
  other_net_m = np.zeros(n_sim)

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  # ターゲット支出（400.0）のまま維持されるはず
  np.testing.assert_allclose(new_spend, 400.0)


def test_hard_floor(base_strategy):
  # 最大限削減（lower_mult = 0.9）しても生存確率が p_low を下回る場合、下限値が採用されることを確認
  strategy, mock_predictor = base_strategy
  mock_predictor.predict_p_surv.side_effect = lambda age, s: np.full_like(
      s, 0.7)

  n_sim = 3
  m = 12
  active_paths = np.ones(n_sim, dtype=bool)

  net_worth = np.full(n_sim, 10000.0)
  prev_base_spend_y = np.full(n_sim, 400.0)
  cpi_m = np.ones(n_sim)
  cpi_m_minus_12 = np.ones(n_sim)
  other_net_m = np.zeros(n_sim)

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  # ターゲット * lower_mult = 400 * 0.9 = 360.0 になるはず
  np.testing.assert_allclose(new_spend, 360.0)


def test_bisection_to_p_low(base_strategy):
  # 二分探索によって生存確率が p_low (0.8) になる支出額が正しく計算されることを確認
  strategy, mock_predictor = base_strategy

  def predict_p(age, s_rate):
    # s_rate=0.04 (400) -> p=0.7 (低すぎ)
    # s_rate=0.036 (360) -> p=0.9 (安全)
    # p = 0.7 + (0.04 - s_rate) * 50 とすると、s_rate=0.038 (380) で p=0.8 となる
    return 0.7 + (0.04 - s_rate) * 50.0

  mock_predictor.predict_p_surv.side_effect = predict_p

  n_sim = 1
  m = 12
  net_worth = np.array([10000.0])
  prev_base_spend_y = np.array([400.0])
  cpi_m = np.array([1.0])
  cpi_m_minus_12 = np.array([1.0])
  other_net_m = np.array([0.0])
  active_paths = np.array([True])

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  # 約 380.0 になるはず
  assert new_spend[0] == pytest.approx(380.0, abs=0.1)


def test_hard_ceiling(base_strategy):
  # 最大限増額（upper_mult = 1.2）しても生存確率が p_high を上回る場合、上限値が採用されることを確認
  strategy, mock_predictor = base_strategy
  mock_predictor.predict_p_surv.side_effect = lambda age, s: np.full_like(
      s, 0.95)

  n_sim = 2
  m = 12
  active_paths = np.ones(n_sim, dtype=bool)

  net_worth = np.full(n_sim, 10000.0)
  prev_base_spend_y = np.full(n_sim, 400.0)
  cpi_m = np.ones(n_sim)
  cpi_m_minus_12 = np.ones(n_sim)
  other_net_m = np.zeros(n_sim)

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  # ターゲット * upper_mult = 400 * 1.2 = 480.0 になるはず
  np.testing.assert_allclose(new_spend, 480.0)


def test_bisection_to_p_high(base_strategy):
  # 二分探索によって生存確率が p_high (0.9) になる支出額が正しく計算されることを確認
  strategy, mock_predictor = base_strategy

  def predict_p(age, s_rate):
    # s_rate=0.04 (400) -> p=0.95 (高すぎ)
    # s_rate=0.048 (480) -> p=0.85 (安全)
    # p = 0.95 - (s_rate - 0.04) * 12.5 とすると、s_rate=0.044 (440) で p=0.9 となる
    return 0.95 - (s_rate - 0.04) * 12.5

  mock_predictor.predict_p_surv.side_effect = predict_p

  n_sim = 1
  m = 12
  net_worth = np.array([10000.0])
  prev_base_spend_y = np.array([400.0])
  cpi_m = np.array([1.0])
  cpi_m_minus_12 = np.array([1.0])
  other_net_m = np.array([0.0])
  active_paths = np.array([True])

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  # 約 440.0 になるはず
  assert new_spend[0] == pytest.approx(440.0, abs=0.1)


def test_mixed_paths(base_strategy):
  # 複数のパスが混在（維持、削減、増額）する場合のベクトル化処理を確認
  strategy, mock_predictor = base_strategy

  # パス 0: p=0.85 (調整なし)
  # パス 1: p=0.7 (下限まで削減 -> 360)
  # パス 2: p=0.95 (上限まで増額 -> 480)

  def predict_p(age, s_rate):
    # s_rate は配列として渡される

    # calculate_nominal_spend 内で predict_p_surv は複数回呼び出される:
    # 1. p_tgt = predict_p_surv(age, s_rate_tgt)
    # 2. p_min = predict_p_surv(age, s_rate_min) (削減が必要なパスのみ)
    # 3. p_max = predict_p_surv(age, s_rate_max) (増額が可能なパスのみ)
    # 4. p_mid = predict_p_surv(age, get_s_rate(mid)) (二分探索中)

    # ターゲット s_rate = 400 / 10000 = 0.04
    if np.allclose(s_rate, 0.04):
      return np.array([0.85, 0.7, 0.95])

    # パス 1 (削減) の下限 s_rate = 360 / 10000 = 0.036
    # 削減しても p_low (0.8) に届かないようにし、ハードフロアを誘発する
    if np.allclose(s_rate, 0.036):
      return np.array([0.6])  # low_mask がかかったパス 1 に対してのみ呼ばれる

    # パス 2 (増額) の上限 s_rate = 480 / 10000 = 0.048
    # 増額しても p_high (0.9) を超えるようにし、ハードシーリングを誘発する
    if np.allclose(s_rate, 0.048):
      return np.array([0.98])  # high_mask がかかったパス 2 に対してのみ呼ばれる

    return np.full_like(s_rate, 0.85)

  mock_predictor.predict_p_surv.side_effect = predict_p

  n_sim = 3
  m = 12
  net_worth = np.full(n_sim, 10000.0)
  prev_base_spend_y = np.full(n_sim, 400.0)
  cpi_m = np.ones(n_sim)
  cpi_m_minus_12 = np.ones(n_sim)
  other_net_m = np.zeros(n_sim)
  active_paths = np.ones(n_sim, dtype=bool)

  new_spend = strategy.calculate_nominal_spend(m, net_worth, prev_base_spend_y,
                                               cpi_m, cpi_m_minus_12,
                                               other_net_m, active_paths)

  assert new_spend[0] == 400.0
  assert new_spend[1] == 360.0
  assert new_spend[2] == 480.0
