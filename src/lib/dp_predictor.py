"""
動的計画法（DP）に基づく最適戦略の予測モジュール。
モデルパラメータを読み込み、最適な資産配分比率と生存確率を予測します。
"""

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union, cast

import numpy as np
from scipy.interpolate import pchip_interpolate


class WinThresholdType(Enum):
  """勝利しきい値の計算方法。"""
  DISABLED = auto()  # 無効
  V1 = auto()  # 従来方式 (Net Withdrawal ベース)
  V2_50 = auto()  # 堅牢方式 (Gross Spend ベース, 50%ile)
  V2_60 = auto()  # 堅牢方式 (Gross Spend ベース, 60%ile)
  V2_70 = auto()  # 堅牢方式 (Gross Spend ベース, 70%ile)
  V2_80 = auto()  # 堅牢方式 (Gross Spend ベース, 80%ile)
  V2_85 = auto()  # 堅牢方式 (Gross Spend ベース, 85%ile)
  V2_90 = auto()  # 堅牢方式 (Gross Spend ベース, 90%ile)
  V2_95 = auto()  # 堅牢方式 (Gross Spend ベース, 95%ile)
  V2_97 = auto()  # 堅牢方式 (Gross Spend ベース, 97%ile)
  V2_98 = auto()  # 堅牢方式 (Gross Spend ベース, 98%ile)
  V2_99 = auto()  # 堅牢方式 (Gross Spend ベース, 99%ile)
  V2_MAX = auto()  # 堅牢方式 (Gross Spend ベース, Max)


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

  def __init__(self,
               models_path: str,
               win_threshold_type: Union[WinThresholdType,
                                         bool] = WinThresholdType.V1):
    """
    JSONファイルからモデルパラメータを読み込み、予測器を初期化します。

    Args:
      models_path: モデルパラメータが格納されたJSONファイルのパス。
      win_threshold_type: 勝利しきい値の種類、または無効化するかどうかの真偽値。
    """
    with open(models_path, "r") as f:
      raw_models = json.load(f)

    self._a_opt_models: Dict[int, AOptModel] = {}
    self._p_surv_models: Dict[int, PSurvModel] = {}
    self._avg_y_withdraws: Dict[int, float] = {}
    self._winning_multipliers: Dict[int, float] = {}
    self._m_winning_multiplier_v2: Dict[int, Dict[str, float]] = {}
    self._cpi_prev_coef: Dict[int, float] = {}
    self._cpi_curr_coef: Dict[int, float] = {}
    self._intercept: Dict[int, float] = {}
    self._ar1_resid_points: Dict[int, List[float]] = {}
    self._cpi_annual_mu: float = raw_models.get("cpi_annual_mu", 0.0)
    self._cpi_annual_sigma: float = raw_models.get("cpi_annual_sigma", 0.0)
    self._net_prediction: str = raw_models.get("net_prediction", "legacy")

    # 勝利しきい値の設定
    if isinstance(win_threshold_type, bool):
      self._win_threshold_type = WinThresholdType.DISABLED if win_threshold_type else WinThresholdType.V1
    else:
      self._win_threshold_type = win_threshold_type

    for age_str, data in raw_models.items():
      if not age_str.isdigit():
        continue
      age = int(age_str)
      if "avg_y_withdraw" in data:
        self._avg_y_withdraws[age] = float(data["avg_y_withdraw"])
      if "m_winning_multiplier" in data:
        self._winning_multipliers[age] = float(data["m_winning_multiplier"])
      if "m_winning_multiplier_v2" in data:
        self._m_winning_multiplier_v2[age] = {
            k: float(v) for k, v in data["m_winning_multiplier_v2"].items()
        }
      if "cpi_prev_coef" in data and "cpi_curr_coef" in data and "intercept" in data:
        self._cpi_prev_coef[age] = float(data["cpi_prev_coef"])
        self._cpi_curr_coef[age] = float(data["cpi_curr_coef"])
        self._intercept[age] = float(data["intercept"])
      if "resid_points" in data:
        self._ar1_resid_points[age] = [float(v) for v in data["resid_points"]]
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

  @property
  def net_prediction(self) -> str:
    """来年の支出予測手法を返します。"""
    return self._net_prediction

  def predict_r_from_ar1(
      self, age: int, current_money: Union[float, np.ndarray],
      cpi_prev: Union[float, np.ndarray],
      cpi_curr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    現在の年齢、総資産、前年CPI、当期首CPIから来年の支出率 R を予測します。
    Expected Resid (50%ile) を使用します。
    """
    if age not in self._intercept:
      # モデルがない場合は 0 を返す
      if isinstance(current_money, np.ndarray):
        return np.zeros_like(current_money)
      return 0.0

    a = self._cpi_prev_coef[age]
    b = self._cpi_curr_coef[age]
    c = self._intercept[age]
    expected_resid = self._ar1_resid_points[age][3]  # 50%ile (Median)

    predicted_net = np.maximum(0.0,
                               a * cpi_prev + b * cpi_curr + c + expected_resid)
    return predicted_net / np.maximum(current_money, 1e-7)

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
      last_gross_withdraw: Optional[Union[float, np.ndarray]] = None,
      z_score: float = 2.326) -> Union[float, np.ndarray]:
    """
    現在の年齢と前年の支出額から、パス依存の勝利しきい値 W_N を計算します。

    Args:
      age: 現在の年齢。
      last_y_withdraw: 前年の実際の支出額（正味）。
      last_gross_withdraw: 前年の実際の総支出額（Gross）。V2系で使用。
      z_score: 勝利しきい値の保守性を決める Z スコア（デフォルト 2.326 は 99%ile）。
        注: V2系ではモデルに Z スコアが内包されているため、この値は V1 方式でのみ使用される。

    Returns:
      Union[float, np.ndarray]: パス依存の勝利しきい値（万円）。
    """
    if self._win_threshold_type == WinThresholdType.DISABLED:
      if isinstance(last_y_withdraw, np.ndarray):
        return np.full_like(last_y_withdraw, float('inf'))
      return float('inf')

    if self._win_threshold_type == WinThresholdType.V1:
      m_n = self._winning_multipliers.get(age, 0.0)
      if m_n <= 0:
        if isinstance(last_y_withdraw, np.ndarray):
          return np.full_like(last_y_withdraw, float('inf'))
        return float('inf')

      # Y_{N-1} から Y_N (最悪ケース) を推定
      expected_growth = self.get_spend_multiplier(age - 1, age)
      worst_case_y_n = last_y_withdraw * expected_growth * self.get_unexpected_cpi_jump(
          z_score)
      return m_n * worst_case_y_n

    # V2 方式 (Robust)
    m_dict = self._m_winning_multiplier_v2.get(age, {})
    if self._win_threshold_type == WinThresholdType.V2_50:
      m_n = m_dict.get("50", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_60:
      m_n = m_dict.get("60", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_70:
      m_n = m_dict.get("70", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_80:
      m_n = m_dict.get("80", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_85:
      m_n = m_dict.get("85", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_90:
      m_n = m_dict.get("90", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_95:
      m_n = m_dict.get("95", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_97:
      m_n = m_dict.get("97", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_98:
      m_n = m_dict.get("98", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_99:
      m_n = m_dict.get("99", 0.0)
    elif self._win_threshold_type == WinThresholdType.V2_MAX:
      m_n = m_dict.get("MAX", 0.0)
    else:
      m_n = 0.0

    if m_n <= 0:
      if isinstance(last_y_withdraw, np.ndarray):
        return np.full_like(last_y_withdraw, float('inf'))
      return float('inf')

    # V2 では分母に Gross Spend を使用する
    if last_gross_withdraw is None:
      # Gross が提供されない場合は Net を代用（非推奨）
      denominator = last_y_withdraw
    else:
      denominator = last_gross_withdraw

    return m_n * denominator

  def get_a_opt_with_winning_threshold(
      self,
      age: int,
      initial_wealth: Union[float, np.ndarray],
      last_y_withdraw: Union[float, np.ndarray],
      last_gross_withdraw: Optional[Union[float, np.ndarray]] = None,
      z_score_for_winning: float = 2.326,
      z_score_for_next_spend: float = 0.0,
      precomputed_r: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    勝利しきい値を考慮して、最適な資産配分 A を決定します。

    アルゴリズム：
    1. 現在の資産 X_N が勝利しきい値 W_N を超えているか判定します。
       W_N = M_V2 * last_gross_withdraw (V2方式) または M_N * WorstCaseY (V1方式)。
    2. もし X_N > W_N であれば、勝利とみなし、W_N を安全資産に割り当て、残りをオルカンに割り当てます。
       A = (X_N - W_N) / X_N
    3. そうでなければ、通常の DP モデルに従って A を決定します。
       この際、支出率 R (withdrawal rate) を計算する必要があります。
       - ar1_residual モデルの場合: すでに計算済みのモーメンタムを考慮した R 
         (precomputed_r) を使用することを強く推奨します。
       - それ以外、または R が提供されない場合: 従来の比率ベースの投影
         (last_y * growth / initial_wealth) を用いて R を算出します。

    Args:
      age: 現在の年齢。
      initial_wealth: 年始の総資産（税引き前、あるいは税引き後の保守的見積もり）。
      last_y_withdraw: 前年の実際の支出額（正味）。
      last_gross_withdraw: 前年の実際の総支出額（Gross）。V2系で使用。
      z_score_for_winning: 勝利しきい値の保守性を決める Z スコア
        （デフォルト 2.326 は 99%ile）。
      z_score_for_next_spend: 来年の支出の保守性を決める Z スコア
        （デフォルト 0.0 は期待値）。比率ベースの計算でのみ使用。
      precomputed_r: 計算済みの支出率 R。ar1_residual 等でモーメンタムを
        考慮した R を再利用する場合に指定します。

    Returns:
      Union[float, np.ndarray]: 最適な株式比率 [0.0, 1.0]。
    """
    w_n = self.calculate_winning_threshold(
        age,
        last_y_withdraw,
        last_gross_withdraw=last_gross_withdraw,
        z_score=z_score_for_winning)

    # スカラーの場合
    if isinstance(initial_wealth, (float, int)):
      if initial_wealth > w_n:
        return (initial_wealth - w_n) / initial_wealth

      if precomputed_r is not None:
        r = float(precomputed_r)
      else:
        expected_growth = self.get_spend_multiplier(age - 1, age)
        if z_score_for_next_spend != 0:
          expected_growth *= self.get_unexpected_cpi_jump(z_score_for_next_spend)
        expected_y_n = last_y_withdraw * expected_growth
        r = expected_y_n / initial_wealth
      return cast(float, self.predict_a_opt(age, r))

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
      if precomputed_r is not None:
        if isinstance(precomputed_r, np.ndarray):
          r = precomputed_r[not_won_mask]
        else:
          r = precomputed_r
      else:
        expected_growth = self.get_spend_multiplier(age - 1, age)
        if z_score_for_next_spend != 0:
          expected_growth *= self.get_unexpected_cpi_jump(z_score_for_next_spend)
        expected_y_n = last_y_arr[not_won_mask] * expected_growth
        r = expected_y_n / wealth_arr[not_won_mask]
      res[not_won_mask] = self.predict_a_opt(age, r)

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

    if y_from <= 1e-6:
      # 前年の取り崩しが0の場合、倍率は定義できないが、安全に 1.0 または y_to をそのまま使うような値を返す
      # ここでは 1.0 を返し、project_s_rate 側で s_rate_from=0 なら 0 になるようにする
      return 1.0

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
