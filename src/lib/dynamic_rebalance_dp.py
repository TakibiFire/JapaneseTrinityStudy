"""
動的計画法（DP）に基づく最適戦略の予測モジュール。
モデルパラメータを読み込み、最適な資産配分比率と生存確率を予測します。
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Union, cast

import numpy as np
from scipy.interpolate import pchip_interpolate


@dataclass
class AOptModel:
  """
  最適資産配分モデルのパラメータを保持するデータクラス。
  """
  r_points: np.ndarray
  a_points: np.ndarray
  r_min_a: float
  r_max_a: float


@dataclass
class PSurvModel:
  """
  生存確率モデルのパラメータを保持するデータクラス。
  """
  r_points: np.ndarray
  p_points: np.ndarray
  r_min_p: float
  r_max_p: float
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
        self._a_opt_models[age] = AOptModel(
            r_points=np.array(a_data["r_points"]),
            a_points=np.array(a_data["a_points"]),
            r_min_a=a_data["r_min_a"],
            r_max_a=a_data["r_max_a"])
      if "p_survival_model" in data:
        p_data = data["p_survival_model"]
        self._p_surv_models[age] = PSurvModel(
            r_points=np.array(p_data["r_points"]),
            p_points=np.array(p_data["p_points"]),
            r_min_p=p_data["r_min_p"],
            r_max_p=p_data["r_max_p"],
            p_max=data.get("p_max", 1.0),
            p_min=data.get("p_min", 0.0))

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

    # スカラーの場合は境界条件の判定を早期に行う
    if isinstance(s_rate, (float, int)):
      rv = float(s_rate)
      if rv <= model.r_min_a or rv >= model.r_max_a:
        return 1.0
      return float(
          pchip_interpolate(model.r_points, model.a_points, np.array([rv]))[0])
    else:
      # 配列の場合
      r_arr = np.asarray(s_rate, dtype=np.float64)
      res = np.ones_like(r_arr)
      in_range = (r_arr > model.r_min_a) & (r_arr < model.r_max_a)
      if np.any(in_range):
        res[in_range] = pchip_interpolate(model.r_points, model.a_points,
                                          r_arr[in_range])
      return res

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
      if rv <= model.r_min_p:
        return float(model.p_max)
      if rv >= model.r_max_p:
        return float(model.p_min)
      return float(
          pchip_interpolate(model.r_points, model.p_points, np.array([rv]))[0])
    else:
      # 配列の場合
      r_arr = np.asarray(s_rate, dtype=np.float64)
      res = np.zeros_like(r_arr)
      res[r_arr <= model.r_min_p] = model.p_max
      res[r_arr >= model.r_max_p] = model.p_min
      in_range = (r_arr > model.r_min_p) & (r_arr < model.r_max_p)
      if np.any(in_range):
        res[in_range] = pchip_interpolate(model.r_points, model.p_points,
                                          r_arr[in_range])
      return res
