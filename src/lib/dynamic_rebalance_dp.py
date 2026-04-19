"""
動的計画法（DP）に基づく最適戦略の予測モジュール。
モデルパラメータを読み込み、最適な資産配分比率と生存確率を予測します。
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Union, cast

import numpy as np


@dataclass
class AOptModel:
  """
  最適資産配分モデルのパラメータを保持するデータクラス。
  """
  coef: np.ndarray
  r2: Optional[float] = None


@dataclass
class PSurvModel:
  """
  生存確率モデルのパラメータを保持するデータクラス。
  """
  coef: np.ndarray
  r_min: float
  r_max: float
  p_max: float
  p_min: float


class DPOptimalStrategyPredictor:
  """
  DPベースの最適資産配分および生存確率の予測クラス。

  Attributes:
    _a_opt_models (Dict[int, AOptModel]): 年齢ごとの最適資産配分モデル。
    _p_surv_models (Dict[int, PSurvModel]): 年齢ごとの生存確率モデル。
  """

  def __init__(self, models_path: str):
    """
    JSONファイルからモデルパラメータを読み込み、予測器を初期化します。

    Args:
      models_path: モデルパラメータが格納されたJSONファイルのパス。
    """
    with open(models_path, "r") as f:
      raw_models = json.load(f)

    self._a_opt_models: Dict[int, AOptModel] = {}
    self._p_surv_models: Dict[int, PSurvModel] = {}

    for age_str, data in raw_models.items():
      age = int(age_str)
      if "a_opt_model" in data:
        a_data = data["a_opt_model"]
        self._a_opt_models[age] = AOptModel(coef=np.array(a_data["coef"]),
                                            r2=a_data.get("r2"))
      if "p_survival_model" in data:
        p_data = data["p_survival_model"]
        self._p_surv_models[age] = PSurvModel(coef=np.array(p_data["coef"]),
                                              r_min=data.get("r_min", 0.0),
                                              r_max=data.get("r_max", 1.0),
                                              p_max=data.get("p_max", 1.0),
                                              p_min=data.get("p_min", 0.0))

  def _get_features(self, s_rate: Union[float, np.ndarray]) -> np.ndarray:
    """
    与えられた支出率に対して多項式特徴量を生成します。
    s_rate がスカラーの場合は形状 (20,)、配列の場合は形状 (N, 20) を返します。

    Args:
      s_rate: 支出率（R）。スカラーまたは numpy 配列。

    Returns:
      np.ndarray: [R, 1/R, log(R)] の3次までの多項式特徴量（20個）。
    """
    # 0 以下の場合は 1e-5 でクリップ（log計算およびゼロ除算回避のため）
    if isinstance(s_rate, (float, int)):
      r = max(float(s_rate), 1e-5)
      inv_r = 1.0 / r
      log_r = np.log(r)
      # x = [x0, x1, x2] = [R, 1/R, log(R)]
      x = [r, inv_r, log_r]

      # sklearn.preprocessing.PolynomialFeatures(degree=3) の出力順序を再現
      feats = [1.0]
      # 次数 1
      feats.extend(x)
      # 次数 2
      for i in range(3):
        for j in range(i, 3):
          feats.append(x[i] * x[j])
      # 次数 3
      for i in range(3):
        for j in range(i, 3):
          for k in range(j, 3):
            feats.append(x[i] * x[j] * x[k])
      return np.array(feats, dtype=np.float64)
    else:
      # ベクトル化された特徴量生成
      r_arr = np.maximum(np.asarray(s_rate, dtype=np.float64), 1e-5)
      inv_r_arr = 1.0 / r_arr
      log_r_arr = np.log(r_arr)
      x_mat = np.column_stack([r_arr, inv_r_arr, log_r_arr])  # (N, 3)
      n = x_mat.shape[0]
      feats_list = [np.ones(n)]
      # 次数 1
      for i in range(3):
        feats_list.append(x_mat[:, i])
      # 次数 2
      for i in range(3):
        for j in range(i, 3):
          feats_list.append(x_mat[:, i] * x_mat[:, j])
      # 次数 3
      for i in range(3):
        for j in range(i, 3):
          for k in range(j, 3):
            feats_list.append(x_mat[:, i] * x_mat[:, j] * x_mat[:, k])
      return np.column_stack(feats_list)  # (N, 20)

  def get_a_opt_model(self, age: int) -> AOptModel:
    """
    指定された年齢の最適資産配分モデルを取得します。
    """
    if age not in self._a_opt_models:
      raise ValueError(f"Optimal A model for age {age} is not found.")
    return self._a_opt_models[age]

  def get_p_surv_model(self, age: int) -> PSurvModel:
    """
    指定された年齢の生存確率モデルを取得します。
    """
    if age not in self._p_surv_models:
      raise ValueError(
          f"Survival probability model for age {age} is not found.")
    return self._p_surv_models[age]

  def predict_a_opt(
      self, age: int, s_rate: Union[float,
                                    np.ndarray]) -> Union[float, np.ndarray]:
    """
    指定された年齢と支出率に対する最適な株式比率を予測します。

    Args:
      age: 現在の年齢。
      s_rate: 支出率。スカラーまたは numpy 配列。

    Returns:
      Union[float, np.ndarray]: 最適な株式比率 [0.0, 1.0]。

    Raises:
      ValueError: 指定された年齢のモデルが存在しない場合。
    """
    if age not in self._a_opt_models:
      raise ValueError(f"Optimal A model for age {age} is not found.")

    model = self._a_opt_models[age]
    feats = self._get_features(s_rate)
    predicted_a = np.dot(feats, model.coef)
    res = np.clip(predicted_a, 0.0, 1.0)
    if isinstance(s_rate, (float, int)):
      return float(res)
    else:
      return cast(np.ndarray, res)

  def predict_p_surv(
      self, age: int, s_rate: Union[float,
                                    np.ndarray]) -> Union[float, np.ndarray]:
    """
    指定された年齢と支出率に対する生存確率を予測します。

    Args:
      age: 現在の年齢。
      s_rate: 支出率。スカラーまたは numpy 配列。

    Returns:
      Union[float, np.ndarray]: 生存確率 [0.0, 1.0]。

    Raises:
      ValueError: 指定された年齢のモデルが存在しない場合。
    """
    if age not in self._p_surv_models:
      raise ValueError(
          f"Survival probability model for age {age} is not found.")

    model = self._p_surv_models[age]

    # スカラーの場合は境界条件の判定を早期に行う
    if isinstance(s_rate, (float, int)):
      rv = float(s_rate)
      if rv <= model.r_min:
        return float(model.p_max)
      if rv >= model.r_max:
        return float(model.p_min)

    feats = self._get_features(s_rate)
    logit_p = np.dot(feats, model.coef)

    # シグモイド逆変換ロジック
    s = 1.0 / (1.0 + np.exp(-np.clip(logit_p, -100, 100)))
    p_surv = model.p_min + s * (model.p_max - model.p_min)

    # 配列の場合は境界条件を適用
    if isinstance(s_rate, np.ndarray):
      p_surv_arr = cast(np.ndarray, p_surv)
      p_surv_arr[s_rate <= model.r_min] = model.p_max
      p_surv_arr[s_rate >= model.r_max] = model.p_min
      return p_surv_arr
    else:
      return float(p_surv)
