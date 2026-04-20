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
      "35": {
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
