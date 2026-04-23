"""
動的計画法（DP）に基づく最適戦略を用いたリバランス戦略。
"""

from typing import Dict, Union

import numpy as np

from .dp_predictor import DPOptimalStrategyPredictor


def calculate_optimal_strategy_dp(
    total_net: np.ndarray,
    cur_ann_spend: np.ndarray,
    rem_years: float,
    post_tax_net: np.ndarray,
    dp_predictor: DPOptimalStrategyPredictor,
    initial_age: int,
    use_winning_threshold: bool = True,
    z_score_for_winning: float = 2.326,
    z_score_for_next_spend: float = 0.0,
    min_a: float = 0.0,
    max_a: float = 1.0) -> Union[float, np.ndarray]:
  """
  DPベースの予測器を用いて、現在の年齢と支出率に対する最適な株式比率を算出します。
  
  Args:
    total_net: 現在の総資産（名目）。
    cur_ann_spend: 現在の年間支出額（名目）。
    rem_years: 残り年数。
    post_tax_net: 税引き後の純資産見積もり。
    dp_predictor: DPモデルに基づく予測器。
    initial_age: シミュレーション開始時の年齢。
    use_winning_threshold: 勝利しきい値ロジックを使用するかどうか。
    z_score_for_winning: 勝利しきい値の保守性を決める Z スコア。
    z_score_for_next_spend: 支出率計算の保守性を決める Z スコア。
    min_a: 株式比率の下限。
    max_a: 株式比率の上限。

  Returns:
    Union[float, np.ndarray]: 株式（オルカン）の配分比率。
  """
  n_sim = len(total_net)
  # 現在の年齢を計算
  current_age = int(round(initial_age + (max(0, 50.0 - rem_years))))

  if use_winning_threshold:
    # 勝利しきい値を考慮した A の計算
    # 簡略化のため、last_y_withdraw は現在の支出率から逆算、あるいはそのまま使用
    # ここでは cur_ann_spend を Y_{N-1} とみなして計算する
    a_opt = dp_predictor.get_a_opt_with_winning_threshold(
        current_age,
        post_tax_net,
        cur_ann_spend,
        z_score_for_winning=z_score_for_winning,
        z_score_for_next_spend=z_score_for_next_spend)
  else:
    # 通常の DP モデル
    s_rate = cur_ann_spend / np.maximum(post_tax_net, 1.0)
    a_opt = dp_predictor.predict_a_opt(current_age, s_rate)

  # 比率を [min_a, max_a] にクリップ
  a_opt = np.clip(a_opt, min_a, max_a)

  return a_opt
