"""
動的計画法（DP）に基づく最適戦略を用いたリバランス戦略。
"""

from typing import Dict, Optional

import numpy as np

from .dp_predictor import DPOptimalStrategyPredictor
from .dynamic_rebalance_type import DPDebugOutput, DRResult


def calculate_optimal_strategy_dp(
    total_net: np.ndarray,
    cur_ann_spend: np.ndarray,
    rem_years: float,
    post_tax_net: np.ndarray,
    dp_predictor: DPOptimalStrategyPredictor,
    initial_age: int,
    total_years: float,
    current_prices: Optional[Dict[str, np.ndarray]] = None,
    prev_prices: Optional[Dict[str, np.ndarray]] = None,
    prev_gross_ann_spend: Optional[np.ndarray] = None,
    use_winning_threshold: bool = True,
    z_score_for_winning: float = 2.326,
    z_score_for_next_spend: float = 0.0,
    min_a: float = 0.0,
    max_a: float = 1.0,
    need_debug: Optional[np.ndarray] = None) -> DRResult:
  """
  DPベースの予測器を用いて、現在の年齢と支出率に対する最適な株式比率を算出します。
  
  Args:
    total_net: 現在の総資産（名目）。
    cur_ann_spend: 現在の年間支出額（名目）。
    rem_years: 残り年数。
    post_tax_net: 税引き後の純資産見積もり。
    dp_predictor: DPモデルに基づく予測器。
    initial_age: シミュレーション開始時の年齢。
    total_years: シミュレーションの全期間（年）。
    current_prices: 現在のアセット価格の辞書（CPIを含む）。
    prev_prices: 前年のアセット価格の辞書。
    prev_gross_ann_spend: 前年の年間総支出額（名目）。
    use_winning_threshold: 勝利しきい値ロジックを使用するかどうか。
    z_score_for_winning: 勝利しきい値の保守性を決める Z スコア。
    z_score_for_next_spend: 支出率計算の保守性を決める Z スコア。
    min_a: 株式比率の下限。
    max_a: 株式比率の上限。
    need_debug: 各パスについてデバッグ情報を返すかどうかのマスク。

  Returns:
    DRResult: 株式（オルカン）の配分比率と、オプションのデバッグ情報。
  """
  n_sim = len(total_net)
  # 現在の年齢を計算
  current_age = int(
      round(initial_age + (max(0, total_years - rem_years + 0.25))))

  if dp_predictor.net_prediction == "ar1_residual":
    if current_prices is None or prev_prices is None:
      raise ValueError(
          "current_prices and prev_prices must be provided when net_prediction is 'ar1_residual'"
      )
    current_cpi = current_prices.get("Japan_CPI", np.ones(n_sim))
    prev_cpi = prev_prices.get("Japan_CPI", np.ones(n_sim))
    s_rate = dp_predictor.predict_r_from_ar1(current_age, post_tax_net,
                                             prev_cpi, current_cpi)
  else:
    s_rate = cur_ann_spend / np.maximum(post_tax_net, 1e-7)

  if use_winning_threshold:
    # 勝利しきい値を考慮した A の計算
    a_opt = dp_predictor.get_a_opt_with_winning_threshold(
        current_age,
        post_tax_net,
        cur_ann_spend,
        last_gross_withdraw=prev_gross_ann_spend,
        z_score_for_winning=z_score_for_winning,
        z_score_for_next_spend=z_score_for_next_spend,
        precomputed_r=s_rate)
  else:
    # 通常の DP モデル
    a_opt = dp_predictor.predict_a_opt(current_age, s_rate)

  # 比率を [min_a, max_a] にクリップ
  a_opt = np.clip(a_opt, min_a, max_a)

  debug_output = None
  if need_debug is not None and np.any(need_debug):
    # 全てのパスに対して計算するが、実際には呼び出し元でフィルタリングされることを想定
    # 来年の予測純支出
    pred_y_n = s_rate * post_tax_net
    # 現在の状態からの予測生存確率
    p_pred = dp_predictor.predict_p_surv(current_age, s_rate)
    # 勝利しきい値
    w_n = dp_predictor.calculate_winning_threshold(
        current_age,
        cur_ann_spend,
        last_gross_withdraw=prev_gross_ann_spend,
        z_score=z_score_for_winning)
    debug_output = DPDebugOutput(pY_N=pred_y_n, P_pred=p_pred, W_N=w_n)

  return DRResult(target_ratios=a_opt, debug=debug_output)
