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
  PCHIP Spline 用のアンカーポイントを保持する。
  """
  r_points: np.ndarray
  a_points: np.ndarray
  r_min_a: float
  r_max_a: float


@dataclass
class PSurvModel:
  """
  生存確率モデルのパラメータを保持するデータクラス。
  PCHIP Spline 用のアンカーポイントを保持する。
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
    self._avg_y_withdraws: Dict[int, float] = {}
    self._winning_multipliers: Dict[int, float] = {}
    self._cpi_annual_mu: float = raw_models.get("cpi_annual_mu", 0.0)
    self._cpi_annual_sigma: float = raw_models.get("cpi_annual_sigma", 0.0)

    for age_str, data in raw_models.items():
      if not age_str.isdigit():
        continue
      age = int(age_str)
      if "avg_y_withdraw" in data:
        self._avg_y_withdraws[age] = float(data["avg_y_withdraw"])
      if "m_winning_multiplier" in data:
        self._winning_multipliers[age] = float(data["m_winning_multiplier"])
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

  def get_unexpected_cpi_jump(self, z_score: float = 2.326) -> float:
    """
    CPI の想定外のジャンプ倍率（バッファ）を取得します。
    unexpected_cpi_jump = (1 + mu + z_score * sigma) / (1 + mu)

    Args:
      z_score: 想定外のジャンプを計算するための Z スコア（デフォルト 2.326 は 99%ile）。

    Returns:
      float: 想定外のジャンプ倍率。
    """
    denom = 1.0 + self._cpi_annual_mu
    if denom <= 0:
      return 1.0
    return (1.0 + self._cpi_annual_mu +
            z_score * self._cpi_annual_sigma) / denom

  def get_winning_multiplier(self, age: int) -> float:
    """
    指定された年齢の勝利しきい値マルチプライヤー M_N を取得します。
    """
    return self._winning_multipliers.get(age, 0.0)

  def calculate_winning_threshold(
      self,
      age: int,
      last_y_withdraw: Union[float, np.ndarray],
      z_score: float = 2.326) -> Union[float, np.ndarray]:
    """
    現在の年齢と前年の支出額から、パス依存の勝利しきい値 W_N を計算します。
    W_N = M_N * Y_{N-1} * (Avg_Y_N / Avg_Y_{N-1}) * unexpected_cpi_jump

    Args:
      age: 現在の年齢。
      last_y_withdraw: 前年の実際の支出額（正味）。
      z_score: 勝利しきい値の保守性を決める Z スコア（デフォルト 2.326 は 99%ile）。

    Returns:
      Union[float, np.ndarray]: パス依存の勝利しきい値（万円）。
    """
    m_n = self.get_winning_multiplier(age)
    if m_n <= 0:
      if isinstance(last_y_withdraw, np.ndarray):
        return np.full_like(last_y_withdraw, float('inf'))
      return float('inf')

    # Y_{N-1} から Y_N (最悪ケース) を推定
    expected_growth = self.get_spend_multiplier(age - 1, age)
    worst_case_y_n = last_y_withdraw * expected_growth * self.get_unexpected_cpi_jump(
        z_score)

    return m_n * worst_case_y_n

  def get_a_opt_with_winning_threshold(
      self,
      age: int,
      initial_wealth: Union[float, np.ndarray],
      last_y_withdraw: Union[float, np.ndarray],
      z_score_for_winning: float = 2.326,
      z_score_for_next_spend: float = 2.326) -> Union[float, np.ndarray]:
    """
    勝利しきい値を考慮して、最適な資産配分 A を決定します。
    もし X_N > W_N であれば、W_N を安全資産に割り当て、残りをオルカンに割り当てます。
    そうでなければ、通常の DP モデルに従います。

    Args:
      age: 現在の年齢。
      initial_wealth: 年始の総資産（税引き前、あるいは税引き後の保守的見積もり）。
      last_y_withdraw: 前年の実際の支出額（正味）。
      z_score_for_winning: 勝利しきい値の保守性を決める Z スコア
        （デフォルト 2.326 は 99%ile）。
      z_score_for_next_spend: 来年の支出の保守性を決める Z スコア
        （デフォルト 2.326 は 99%ile）。
    Returns:
      Union[float, np.ndarray]: 最適な株式比率 [0.0, 1.0]。
    """
    w_n = self.calculate_winning_threshold(age, last_y_withdraw,
                                           z_score_for_winning)

    # スカラーの場合
    if isinstance(initial_wealth, (float, int)):
      if initial_wealth > w_n:
        return (initial_wealth - w_n) / initial_wealth

      expected_growth = self.get_spend_multiplier(age - 1, age)
      expected_y_n = last_y_withdraw * expected_growth
      s_rate = expected_y_n / initial_wealth
      return cast(float, self.predict_a_opt(age, s_rate))

    # 配列の場合
    wealth_arr = np.asarray(initial_wealth, dtype=np.float64)
    last_y_arr = np.asarray(last_y_withdraw, dtype=np.float64)
    w_n_arr = np.asarray(w_n, dtype=np.float64)

    # 勝利判定
    won_mask = wealth_arr > w_n_arr
    res = np.zeros_like(wealth_arr)

    # 勝利した場合: A = (X_N - W_N) / X_N
    res[won_mask] = (wealth_arr[won_mask] -
                     w_n_arr[won_mask]) / wealth_arr[won_mask]

    # 勝利していない場合: 通常の DP
    not_won_mask = ~won_mask
    if np.any(not_won_mask):
      expected_growth = self.get_spend_multiplier(
          age - 1, age) * self.get_unexpected_cpi_jump(z_score_for_next_spend)
      expected_y_n = last_y_arr[not_won_mask] * expected_growth
      s_rate = expected_y_n / wealth_arr[not_won_mask]
      res[not_won_mask] = self.predict_a_opt(age, s_rate)

    return res

  def get_p_surv_model(self, age: int) -> PSurvModel:
    """
    指定された年齢の生存確率モデルを取得します。
    """
    if age not in self._p_surv_models:
      raise ValueError(
          f"Survival probability model for age {age} is not found.")
    return self._p_surv_models[age]

  def get_spend_multiplier(self, age_from: int, age_to: int) -> float:
    """
    指定された年齢間の平均支出（Withdrawal）の比率（倍率）を取得します。
    投影に使用されます。
    """
    if age_from not in self._avg_y_withdraws or age_to not in self._avg_y_withdraws:
      return 1.0

    y_from = self._avg_y_withdraws[age_from]
    y_to = self._avg_y_withdraws[age_to]

    if y_from <= 0 or y_to <= 0:
      raise ValueError(
          f"Average withdrawal must be positive for multiplier calculation. Age {age_from}: {y_from}, Age {age_to}: {y_to}"
      )

    return y_to / y_from

  def project_s_rate(self, age_from: int, s_rate_from: Union[float, np.ndarray],
                     age_to: int) -> Union[float, np.ndarray]:
    """
    age_from における支出率 s_rate_from を、age_to における支出率に投影します。
    S_to = S_from * (Avg_Y_to / Avg_Y_from) として計算されます。
    """
    multiplier = self.get_spend_multiplier(age_from, age_to)
    return s_rate_from * multiplier

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
