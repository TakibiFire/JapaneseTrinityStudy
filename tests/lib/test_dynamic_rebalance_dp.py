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
  # 20個の係数（sklearnのPolynomialFeatures(degree=3)の順序に合わせる）
  # [1, R, invR, logR, R^2, R*invR, R*logR, invR^2, invR*logR, logR^2, ...]

  # 最適資産配分モデル: A = 0.5 + 0.1*R
  coef_a = [0.0] * 20
  coef_a[0] = 0.5  # Bias
  coef_a[1] = 0.1  # R の係数

  # 生存確率モデル: logit_p = 0 (常に 0.5 の確率)
  coef_p = [0.0] * 20

  models = {
      "35": {
          "a_opt_model": {
              "coef": coef_a,
              "r2": 0.9
          },
          "p_survival_model": {
              "coef": coef_p
          },
          "r_min": 0.02,
          "r_max": 0.10,
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
  assert predictor.get_a_opt_model(35).coef.shape == (20,)
  assert predictor.get_a_opt_model(35).r2 == 0.9
  assert predictor.get_p_surv_model(35).r_min == 0.02


def test_predict_a_opt_scalar(mock_models_json):
  """predict_a_opt のスカラー入力テスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # float 入力: A = 0.5 + 0.1 * 0.04 = 0.504
  val = predictor.predict_a_opt(35, 0.04)
  assert isinstance(val, float)
  assert pytest.approx(val) == 0.504

  # int 入力: A = 0.5 + 0.1 * 1 = 0.6
  val_int = predictor.predict_a_opt(35, 1)
  assert isinstance(val_int, float)
  assert pytest.approx(val_int) == 0.6

  # クリップのテスト: A = 0.5 + 0.1 * 10 = 1.5 -> 1.0
  val_clip = predictor.predict_a_opt(35, 10.0)
  assert val_clip == 1.0


def test_predict_p_surv_scalar_boundaries(mock_models_json):
  """predict_p_surv の境界条件テスト（スカラー）。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # R <= r_min (0.02) -> p_max (0.9)
  assert predictor.predict_p_surv(35, 0.01) == 0.9
  assert predictor.predict_p_surv(35, 0.02) == 0.9

  # R >= r_max (0.10) -> p_min (0.1)
  assert predictor.predict_p_surv(35, 0.10) == 0.1
  assert predictor.predict_p_surv(35, 0.11) == 0.1

  # 負の支出率 -> p_max (0.9)
  assert predictor.predict_p_surv(35, -0.05) == 0.9


def test_predict_p_surv_scalar_middle(mock_models_json):
  """predict_p_surv の中間値テスト（スカラー）。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)

  # logit_p = 0 -> sigmoid(0) = 0.5
  # p = p_min + 0.5 * (p_max - p_min) = 0.1 + 0.5 * 0.8 = 0.5
  val = predictor.predict_p_surv(35, 0.05)
  assert isinstance(val, float)
  assert pytest.approx(val) == 0.5


def test_vectorized_predictions(mock_models_json):
  """ベクトル化された予測のテスト。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)
  r_vals = np.array([0.01, 0.05, 0.11])

  # 最適 A の予測
  a_vals = predictor.predict_a_opt(35, r_vals)
  assert isinstance(a_vals, np.ndarray)
  assert a_vals.shape == (3,)
  assert pytest.approx(a_vals[0]) == 0.5 + 0.1 * 0.01
  assert pytest.approx(a_vals[1]) == 0.5 + 0.1 * 0.05
  assert pytest.approx(a_vals[2]) == 0.5 + 0.1 * 0.11

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


def test_feature_generation_order(mock_models_json):
  """多項式特徴量の生成順序が正しいか確認。"""
  predictor = DPOptimalStrategyPredictor(mock_models_json)
  # R=1.0 の場合: invR=1.0, logR=0.0
  # [1, R, invR, logR, R^2, R*invR, R*logR, invR^2, invR*logR, logR^2, ...]
  # = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, ...]
  feats = predictor._get_features(1.0)
  assert feats[0] == 1.0  # Bias
  assert feats[1] == 1.0  # R
  assert feats[2] == 1.0  # invR
  assert feats[3] == 0.0  # logR
  assert feats[5] == 1.0  # R * invR = 1.0
  assert feats[6] == 0.0  # R * logR = 0.0
