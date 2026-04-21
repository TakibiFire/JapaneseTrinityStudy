"""
DPOptimalStrategyPredictor のテストスイート。
"""

import json

import numpy as np
import pytest

from src.lib.dynamic_rebalance_dp import DPOptimalStrategyPredictor


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
      },
      "36": {
          "avg_y_withdraw": 110.0,
          "m_winning_multiplier": 9.0,
          "a_opt_model": a_opt_model,
          "p_survival_model": p_survival_model,
          "p_min": 0.1,
          "p_max": 0.9
      }
  }

  path = tmp_path / "test_models.json"
  with open(path, "w") as f:
    json.dump(models, f)
  return str(path)


def test_predictor_initialization(mock_models_json):
  """初期化のテスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)
  # 内部モデルが正しくロードされているか
  assert len(predictor.get_a_opt_model(35).r_points) == 2
  assert predictor.get_a_opt_model(35).r_min_a == 0.02
  assert predictor.get_p_surv_model(35).r_min_p == 0.02
  assert predictor.get_p_surv_model(35).p_max == 0.9


def test_predict_a_opt_scalar(mock_models_json):
  """predict_a_opt のスカラー入力テスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # R <= r_min_a (0.02) -> 1.0
  assert predictor.predict_a_opt(35, 0.01) == 1.0
  assert predictor.predict_a_opt(35, 0.02) == 1.0

  # R >= r_max_a (0.10) -> 1.0 (仕様: 範囲外は 1.0)
  assert predictor.predict_a_opt(35, 0.10) == 1.0
  assert predictor.predict_a_opt(35, 0.11) == 1.0

  # 中間値: R=0.06 -> (1.0 + 0.5) / 2 = 0.75 ?
  # PCHIP だが 2点間は線形
  val = predictor.predict_a_opt(35, 0.06)
  assert isinstance(val, float)
  assert pytest.approx(val) == 0.75


def test_predict_p_surv_scalar_boundaries(mock_models_json):
  """predict_p_surv の境界条件テスト（スカラー）。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # R <= r_min_p (0.02) -> p_max (0.9)
  assert predictor.predict_p_surv(35, 0.01) == 0.9
  assert predictor.predict_p_surv(35, 0.02) == 0.9

  # R >= r_max_p (0.10) -> p_min (0.1)
  assert predictor.predict_p_surv(35, 0.10) == 0.1
  assert predictor.predict_p_surv(35, 0.11) == 0.1

  # 負の支出率 -> p_max (0.9)
  assert predictor.predict_p_surv(35, -0.05) == 0.9


def test_predict_p_surv_scalar_middle(mock_models_json):
  """predict_p_surv の中間値テスト（スカラー）。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # R=0.06 は 0.02 と 0.10 の中間 -> (0.9 + 0.1) / 2 = 0.5
  val = predictor.predict_p_surv(35, 0.06)
  assert isinstance(val, float)
  assert pytest.approx(val) == 0.5


def test_vectorized_predictions(mock_models_json):
  """ベクトル化された予測のテスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)
  r_vals = np.array([0.01, 0.06, 0.11])

  # 最適 A の予測
  a_vals = predictor.predict_a_opt(35, r_vals)
  assert isinstance(a_vals, np.ndarray)
  assert a_vals.shape == (3,)
  assert a_vals[0] == 1.0  # R=0.01 <= 0.02
  assert pytest.approx(a_vals[1]) == 0.75  # R=0.06
  assert a_vals[2] == 1.0  # R=0.11 >= 0.10

  # 生存確率の予測
  p_vals = predictor.predict_p_surv(35, r_vals)
  assert isinstance(p_vals, np.ndarray)
  assert p_vals.shape == (3,)
  assert p_vals[0] == 0.9  # 境界以下
  assert pytest.approx(p_vals[1]) == 0.5  # 中間
  assert p_vals[2] == 0.1  # 境界以上


def test_error_handling(mock_models_json):
  """エラーハンドリングのテスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # 未サポートの年齢
  with pytest.raises(ValueError,
                     match="Optimal A model for age 40 is not found"):
    predictor.predict_a_opt(40, 0.05)

  with pytest.raises(
      ValueError, match="Survival probability model for age 40 is not found"):
    predictor.predict_p_surv(40, 0.05)


def test_winning_threshold_calculation(mock_models_json):
  """勝利しきい値の計算テスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # age 36 をテスト。
  # last_y_withdraw = 100.0 (age 35 の実績)
  # expected_growth = avg_y_withdraw[36] / avg_y_withdraw[35] = 110.0 / 100.0 = 1.1
  # mu = 0.01, sigma = 0.04
  # z_score = 2.0 (テスト用)
  # unexpected_cpi_jump = (1 + 0.01 + 2.0 * 0.04) / (1 + 0.01) = 1.09 / 1.01 = 1.07920792
  # m_n = 9.0 (age 36)
  # worst_case_y_n = 100.0 * 1.1 * 1.07920792 = 118.712871
  # threshold = 9.0 * 118.712871 = 1068.4158

  threshold = predictor.calculate_winning_threshold(36, 100.0, z_score=2.0)
  assert pytest.approx(threshold) == 1068.41583


def test_get_a_opt_with_winning_threshold(mock_models_json):
  """勝利しきい値を考慮した A の予測テスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # z_score=2.0 のとき、しきい値 = 1068.41583
  # ケース1: 資産がしきい値を超える場合 (initial_wealth = 2000.0)
  # A = (2000 - 1068.41583) / 2000 = 931.58417 / 2000 = 0.465792
  a1 = predictor.get_a_opt_with_winning_threshold(36,
                                                  2000.0,
                                                  100.0,
                                                  z_score_for_winning=2.0)
  assert pytest.approx(a1) == 0.465792

  # ケース2: 資産がしきい値を下回る場合 (initial_wealth = 1000.0)
  # 通常の DP モデルを使用。
  # z_score_for_next_spend=0.0 (デフォルト)
  # expected_y_n = 100.0 * 1.1 = 110.0
  # s_rate = 110 / 1000 = 0.11
  # R=0.11 は境界 (0.10) 以上なので predict_a_opt は 1.0
  a2 = predictor.get_a_opt_with_winning_threshold(36,
                                                  1000.0,
                                                  100.0,
                                                  z_score_for_winning=2.0)
  assert a2 == 1.0

  # ケース3: z_score_for_next_spend の効果を確認
  # wealth = 2000.0, last_y = 100.0 (w_n = 1068.41583 なので勝利条件は満たすが、
  # 比較のために内部の DP ロジックを直接テストしたい)
  # 勝利条件を回避するために w_n を大きくする (z_score_for_winning を大きくする)
  # s_rate (z=0) = 0.055 -> A = 0.78125
  a3 = predictor.get_a_opt_with_winning_threshold(36,
                                                  2000.0,
                                                  100.0,
                                                  z_score_for_winning=100.0,
                                                  z_score_for_next_spend=0.0)
  assert pytest.approx(a3) == 0.78125

  # s_rate (z=2.0) = 0.059356 -> A = 0.754022...
  a4 = predictor.get_a_opt_with_winning_threshold(36,
                                                  2000.0,
                                                  100.0,
                                                  z_score_for_winning=100.0,
                                                  z_score_for_next_spend=2.0)
  assert pytest.approx(a4) == 0.754022277


def test_get_a_opt_with_winning_threshold_vectorized(mock_models_json):
  """勝利しきい値を考慮した A の予測テスト（ベクトル化）。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # z_score=2.0 のとき、しきい値 = 1068.41583
  wealth = np.array([2000.0, 1000.0])
  last_y = np.array([100.0, 100.0])

  res = predictor.get_a_opt_with_winning_threshold(36,
                                                   wealth,
                                                   last_y,
                                                   z_score_for_winning=2.0)
  assert isinstance(res, np.ndarray)
  assert res.shape == (2,)
  assert pytest.approx(res[0]) == 0.465792
  assert res[1] == 1.0
