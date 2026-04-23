"""
dynamic_rebalance_dp.py のテスト。
"""

import json

import numpy as np
import pytest

from src.lib.dp_predictor import DPOptimalStrategyPredictor
from src.lib.dynamic_rebalance_dp import calculate_optimal_strategy_dp


@pytest.fixture
def mock_models_json(tmp_path):
  """
  テスト用のモックモデルJSONファイルを作成するフィクスチャ。
  """
  # 最適資産配分モデル (A_opt): R=0.02 で 1.0, R=0.10 で 0.5 の直線的（PCHIP）な関係
  a_opt_model = {
      "r_points": [0.02, 0.10],
      "a_points": [1.0, 0.5],
      "r_min_a": 0.02,
      "r_max_a": 0.10
  }

  # 生存確率モデル (P_surv): R=0.02 で 0.9, R=0.10 で 0.1 の直線的（PCHIP）な関係
  p_survival_model = {
      "r_points": [0.02, 0.10],
      "p_points": [0.9, 0.1],
      "r_min_p": 0.02,
      "r_max_p": 0.10
  }

  models = {
      "cpi_annual_mu": 0.01,
      "cpi_annual_sigma": 0.04,
      "35": {
          "avg_y_withdraw": 100.0,
          "m_winning_multiplier": 10.0,
          "a_opt_model": a_opt_model,
          "p_survival_model": p_survival_model,
          "p_min": 0.1,
          "p_max": 0.9
      }
  }

  path = tmp_path / "test_models_dr.json"
  with open(path, "w") as f:
    json.dump(models, f)
  return str(path)


def test_calculate_optimal_strategy_dp(mock_models_json):
  """calculate_optimal_strategy_dp の基本的なテスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  total_net = np.array([2000.0, 1000.0])
  cur_ann_spend = np.array([100.0, 100.0])
  # initial_age=35, rem_years=50 -> current_age = 35 + (50 - 50) = 35
  rem_years = 50.0
  post_tax_net = total_net.copy()

  res = calculate_optimal_strategy_dp(
      total_net,
      cur_ann_spend,
      rem_years,
      post_tax_net,
      dp_predictor=predictor,
      initial_age=35,
      use_winning_threshold=True,
      z_score_for_winning=2.0)

  assert isinstance(res, np.ndarray)
  assert res.shape == (2,)
  # threshold = m_n * last_y * growth * jump
  # threshold for 35: 10 * 100 * 1 * 1.07920792 = 1079.20792
  # A = (2000 - 1079.20792) / 2000 = 0.4603960396
  assert pytest.approx(res[0]) == 0.4603960396
  # Path 1: 勝利条件未達成 -> 通常DP (s_rate = 100/1000 = 0.1 -> A=1.0)
  assert res[1] == 1.0
