"""
生存確率に基づき基本支出を動的に調整する SpendAwareDynamicSpending クラスの実装。
"""

import dataclasses
from typing import Dict, List, Optional, Union

import numpy as np

from src.lib.cashflow_generator import CashflowDynamicHandler
from src.lib.dp_predictor import DPOptimalStrategyPredictor


@dataclasses.dataclass
class SpendAwareDynamicSpending(CashflowDynamicHandler):
  """
  生存確率（P）ベースのガードレール戦略。
  
  ライフプランに合わせた支出曲線を維持しつつ、資産状況と支出額から
  計算される生存確率が一定範囲（p_low, p_high）に収まるように、
  基本支出額を動的に増減させる。
  """
  initial_age: int
  p_low: float
  p_high: float
  lower_mult: float
  upper_mult: float
  annual_cost_real: List[float]
  dp_predictor: DPOptimalStrategyPredictor

  def evaluate(self, m: int, active_paths: np.ndarray,
               current_net_worth: np.ndarray, tax_cost_m: np.ndarray,
               prev_actual_amount: np.ndarray, other_net_m: np.ndarray,
               precomputed_cf_m: np.ndarray,
               precomputed_cf_prev_m: np.ndarray) -> np.ndarray:
    """
    動的なキャッシュフロー（支出額）を算出する。

    Args:
      m: シミュレーション開始からの経過月数
      active_paths: 現在生存しているパスのマスク (n_sim,)
      current_net_worth: 現在の純資産 (n_sim,)
      tax_cost_m: 今月支払う税金 (n_sim,)
      prev_actual_amount: 前年のこのルールによる実際の年間支出実績 (n_sim,)
      other_net_m: 他の REGULAR キャッシュフローによる正味収支（正は支出、負は収入） (n_sim,)
      precomputed_cf_m: 事前計算された今月のキャッシュフロー額（正の絶対値） (n_sim,)
      precomputed_cf_prev_m: 事前計算された12ヶ月前のキャッシュフロー額（正の絶対値） (n_sim,)

    Returns:
      np.ndarray: 新しい年間名目支出額 (n_sim,)。正の値が支出を表す。
    """
    n_sim = len(active_paths)
    new_base_nom = np.zeros(n_sim, dtype=np.float64)

    age = self.initial_age + m // 12
    year_idx = m // 12

    # Step 1: 目標名目基本支出額の算出
    if m == 0:
      target_base_nom = np.full(n_sim, float(self.annual_cost_real[0]))
    else:
      # precomputed_cf_m / precomputed_cf_prev_m が cpi_ratio * lifeplan_ratio に相当する
      cf_ratio = np.ones(n_sim)
      mask_nonzero = precomputed_cf_prev_m > 0
      cf_ratio[mask_nonzero] = precomputed_cf_m[mask_nonzero] / precomputed_cf_prev_m[mask_nonzero]
      target_base_nom = prev_actual_amount * cf_ratio

    # 有効なパスのみ計算対象とする
    # Effective NW = current_net_worth - tax_cost_m
    nw_active = current_net_worth[active_paths] - tax_cost_m[active_paths]
    other_net_active = other_net_m[active_paths]
    target_active = target_base_nom[active_paths]

    # Step 3: S_Rate 関数 (annual withdrawal rate based on nominal base spend Y)
    def get_s_rate(y: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
      nw = nw_active[mask] if mask is not None else nw_active
      other_net = other_net_active[mask] if mask is not None else other_net_active
      safe_nw = np.maximum(nw, 1.0)
      return ((y / 12.0 + other_net) * 12.0) / safe_nw

    # Step 4: ガードレール判定
    s_rate_tgt = get_s_rate(target_active)
    p_tgt = self.dp_predictor.predict_p_surv(age, s_rate_tgt)

    y_min = target_active * self.lower_mult
    y_max = target_active * self.upper_mult

    res_active = target_active.copy()

    # 1. Low Probability Handling (支出削減が必要な場合)
    low_mask = p_tgt < self.p_low
    if np.any(low_mask):
      s_rate_min = get_s_rate(y_min[low_mask], low_mask)
      p_min = self.dp_predictor.predict_p_surv(age, s_rate_min)

      # 最大限削減しても p_low に届かない場合
      fully_cut_mask = p_min < self.p_low
      res_active[np.where(low_mask)[0]
                 [fully_cut_mask]] = y_min[low_mask][fully_cut_mask]

      # 二分探索で p_low になる支出額を探す
      bisect_mask = ~fully_cut_mask
      if np.any(bisect_mask):
        idx = np.where(low_mask)[0][bisect_mask]
        y_low = y_min[low_mask][bisect_mask]
        y_high = target_active[idx]

        # bisection 用の NW と other_net を抽出
        nw_bisect = nw_active[idx]
        other_net_bisect = other_net_active[idx]

        for _ in range(15):
          mid = (y_low + y_high) / 2.0
          # get_s_rate を使わず直接計算（効率のため）
          s_mid = ((mid / 12.0 + other_net_bisect) * 12.0) / np.maximum(
              nw_bisect, 1.0)
          p_mid = self.dp_predictor.predict_p_surv(age, s_mid)
          # p_surv は支出 Y に対して広義単調減少
          go_lower = p_mid < self.p_low
          y_high[go_lower] = mid[go_lower]
          y_low[~go_lower] = mid[~go_lower]

        res_active[idx] = y_low  # より安全側（支出が少ない側）を選択

    # 2. High Probability Handling (支出増加が可能な場合)
    high_mask = p_tgt > self.p_high
    if np.any(high_mask):
      s_rate_max = get_s_rate(y_max[high_mask], high_mask)
      p_max = self.dp_predictor.predict_p_surv(age, s_rate_max)

      # 最大限増やしても p_high を超える場合
      fully_boost_mask = p_max > self.p_high
      res_active[np.where(high_mask)[0]
                 [fully_boost_mask]] = y_max[high_mask][fully_boost_mask]

      # 二分探索で p_high になる支出額を探す
      bisect_mask = ~fully_boost_mask
      if np.any(bisect_mask):
        idx = np.where(high_mask)[0][bisect_mask]
        y_low = target_active[idx]
        y_high = y_max[high_mask][bisect_mask]

        # bisection 用の NW と other_net を抽出
        nw_bisect = nw_active[idx]
        other_net_bisect = other_net_active[idx]

        for _ in range(15):
          mid = (y_low + y_high) / 2.0
          s_mid = ((mid / 12.0 + other_net_bisect) * 12.0) / np.maximum(
              nw_bisect, 1.0)
          p_mid = self.dp_predictor.predict_p_surv(age, s_mid)
          go_higher = p_mid > self.p_high
          y_low[go_higher] = mid[go_higher]
          y_high[~go_higher] = mid[~go_higher]

        res_active[idx] = y_low  # より安全側（支出が少ない側）を選択

    new_base_nom[active_paths] = res_active
    return new_base_nom

  def calculate_nominal_spend(self, m: int, net_worth: np.ndarray,
                              prev_base_spend_y: np.ndarray, cpi_m: np.ndarray,
                              cpi_m_minus_12: np.ndarray,
                              other_net_m: np.ndarray,
                              active_paths: np.ndarray) -> np.ndarray:
    """
    (Deprecated) 旧エンジン用の計算メソッド。

    
    TODO: この関数は Phase 1 の `core.py` への移行が完了し、全ての既存スクリプトが
    `evaluate` を使用するように修正された後に削除可能です。
    削除前には、`evaluate` が `calculate_nominal_spend` と（数学的に）同一の
    結果を返すことを `tests/lib/test_core.py` で検証する必要があります。
    """
    return self.evaluate(
        m=m,
        active_paths=active_paths,
        current_net_worth=net_worth,
        tax_cost_m=np.zeros_like(net_worth),
        prev_actual_amount=prev_base_spend_y,
        other_net_m=other_net_m,
        precomputed_cf_m=cpi_m * self.annual_cost_real[m//12] if m//12 < len(self.annual_cost_real) else cpi_m * self.annual_cost_real[-1],
        precomputed_cf_prev_m=cpi_m_minus_12 * self.annual_cost_real[max(0, m//12-1)] if max(0, m//12-1) < len(self.annual_cost_real) else cpi_m_minus_12 * self.annual_cost_real[-1]
    )
