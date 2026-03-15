"""
日本版トリニティ・スタディの戦略最適化を行うモジュール。

scipy.optimize.brute を用いたグリッドサーチにより、
目的関数（破産確率の最小化、10パーセンタイルの最大化、中央値の最大化）
を最適化する資産配分・借入比率・リバランス間隔の組み合わせを探索する。
"""

import enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.optimize

from core import Strategy, ZeroRiskAsset, simulate_strategy


class OptimizationTarget(enum.Enum):
  """
  最適化の目的関数を指定する列挙型。
  """
  MINIMIZE_RUIN_PROBABILITY_20Y = "minimize_ruin_probability_20y"
  MINIMIZE_RUIN_PROBABILITY_30Y = "minimize_ruin_probability_30y"
  MINIMIZE_RUIN_PROBABILITY_40Y = "minimize_ruin_probability_40y"
  MINIMIZE_RUIN_PROBABILITY_50Y = "minimize_ruin_probability_50y"
  MAXIMIZE_10_PERCENTILE = "maximize_10_percentile"
  MAXIMIZE_50_PERCENTILE = "maximize_50_percentile"


def create_strategy(i: float,
                    j: float,
                    k: float,
                    rebalance_interval: int = 0,
                    name: str = "Optimization Target",
                    zero_risk_asset: Optional[ZeroRiskAsset] = None,
                    zero_risk_ratio: float = 0.0) -> Strategy:
  """
  指定されたパラメータ(i, j, k)からシミュレーション用のStrategyオブジェクトを作成する。
  
  Args:
    i: オルカンの初期資産割合
    j: レバカンの初期資産割合
    k: 初期借入額の係数 (k * 1000 万円の借入)
    rebalance_interval: リバランス間隔 (月)
    name: 戦略の名前
    zero_risk_asset: ポートフォリオに含める無リスク資産（オプション）
    zero_risk_ratio: 無リスク資産の初期資産割合
    
  Returns:
    Strategy: 構築された戦略オブジェクト
  """
  ratio_dict: Dict[Union[str, ZeroRiskAsset], float] = {"オルカン": i, "レバカン": j}
  if zero_risk_asset is not None and zero_risk_ratio > 0.0:
    ratio_dict[zero_risk_asset] = zero_risk_ratio

  return Strategy(name=name,
                  initial_money=10000.0,
                  initial_loan=float(k * 1000.0),
                  yearly_loan_interest=2.125 / 100,
                  initial_asset_ratio=ratio_dict,
                  annual_cost=400.0,
                  inflation_rate=0.015,
                  selling_priority=["レバカン", "オルカン"],
                  rebalance_interval=rebalance_interval)


def evaluate_strategy(params: Tuple[float, float, float, int],
                      monthly_asset_prices: Dict[str, np.ndarray],
                      target: OptimizationTarget,
                      zero_risk_asset: Optional[ZeroRiskAsset] = None,
                      zero_risk_ratio: float = 0.0) -> float:
  """
  最適化探索 (scipy.optimize.brute) から呼び出される評価関数。
  
  与えられたパラメータで戦略を構築・シミュレーションし、指定された目的関数に
  対応する評価スコアを返す (すべて最小化問題として解けるよう符号を調整)。
  
  Args:
    params: 探索空間のパラメータ。
            i: オルカンの初期資産割合
            j: レバカンの初期資産割合
            k: 初期借入額の係数 (k * 1000 万円)
            r: リバランス間隔 (0 または 12)
    monthly_asset_prices: 事前計算された各資産の月次価格推移。
    target: 評価する目的関数 (OptimizationTarget)。
    zero_risk_asset: ポートフォリオに含める無リスク資産（オプション）
    zero_risk_ratio: 無リスク資産の初期資産割合
  
  Returns:
    最適化ソルバー向けの評価スコア (float)。
    (i + j + zero_risk_ratio <= 1.0) の制約を満たさない場合は float('inf') を返す。
  """
  i, j, k, r = params

  # 制約条件: i + j + zero_risk_ratio <= 1.0 (浮動小数点の誤差を許容)
  if i + j + zero_risk_ratio > 1.0 + 1e-9:
    return float('inf')

  # 評価用の戦略を構築
  strategy = create_strategy(i,
                             j,
                             k,
                             int(r),
                             zero_risk_asset=zero_risk_asset,
                             zero_risk_ratio=zero_risk_ratio)

  # シミュレーション実行
  res = simulate_strategy(strategy, monthly_asset_prices)
  net_values = res.net_values
  sustained_months = res.sustained_months

  # 目的関数に応じたスコア計算
  if target == OptimizationTarget.MINIMIZE_RUIN_PROBABILITY_20Y:
    # 20年破産確率 (%) を計算
    ruin_prob = np.mean(sustained_months < 20 * 12) * 100.0
    return float(ruin_prob)
  elif target == OptimizationTarget.MINIMIZE_RUIN_PROBABILITY_30Y:
    # 30年破産確率 (%) を計算
    ruin_prob = np.mean(sustained_months < 30 * 12) * 100.0
    return float(ruin_prob)
  elif target == OptimizationTarget.MINIMIZE_RUIN_PROBABILITY_40Y:
    # 40年破産確率 (%) を計算
    ruin_prob = np.mean(sustained_months < 40 * 12) * 100.0
    return float(ruin_prob)
  elif target == OptimizationTarget.MINIMIZE_RUIN_PROBABILITY_50Y:
    # 50年破産確率 (%) を計算
    ruin_prob = np.mean(sustained_months < 50 * 12) * 100.0
    return float(ruin_prob)
  elif target == OptimizationTarget.MAXIMIZE_10_PERCENTILE:
    # 10パーセンタイル値を計算し、最大化のため負の値にする
    p10 = np.percentile(net_values, 10)
    return float(-p10)
  elif target == OptimizationTarget.MAXIMIZE_50_PERCENTILE:
    # 50パーセンタイル値を計算し、最大化のため負の値にする
    p50 = np.percentile(net_values, 50)
    return float(-p50)
  else:
    raise ValueError(f"Unknown target: {target}")


def optimize_strategy(
    monthly_asset_prices: Dict[str, np.ndarray],
    target: OptimizationTarget,
    zero_risk_asset: Optional[ZeroRiskAsset] = None,
    zero_risk_ratio: float = 0.0) -> Tuple[float, float, int, int, float]:
  """
  scipy.optimize.brute を使用し、指定された目的関数に従って最適な (i, j, k, r) を探索する。
  
  Args:
    monthly_asset_prices: 月次資産価格のシミュレーションデータ
    target: 最適化の目的 (OptimizationTarget)
    zero_risk_asset: ポートフォリオに含める無リスク資産（オプション）
    zero_risk_ratio: 無リスク資産の初期資産割合
    
  Returns:
    Tuple[float, float, int, int, float]: 最適な (i, j, k, r) とその時の（元の）スコア
  """
  # 探索範囲の定義
  # i: 0.0 から 1.0 まで 0.1 刻み (スライス終端は含まれないため 1.1)
  # j: 0.0 から 1.0 まで 0.1 刻み
  # k: 0 から 5 まで 1 刻み
  # r: リバランス間隔 (0, 12 の2通り)
  ranges = (
      slice(0.0, 1.1, 0.1),
      slice(0.0, 1.1, 0.1),
      slice(0, 6, 1),
      slice(0, 13, 12)  # 0, 12 のみを探索
  )

  # bruteによる全探索
  # Nsは各軸のグリッド点数を指定（スライスのステップ幅が優先される）
  # full_output=True で詳細な結果を取得
  res = scipy.optimize.brute(
      func=evaluate_strategy,
      ranges=ranges,
      args=(monthly_asset_prices, target, zero_risk_asset, zero_risk_ratio),
      full_output=True,
      finish=None  # 局所最適化（Nelder-Meadなど）を行わず、グリッド上の最小値をそのまま返す
  )

  best_params = res[0]
  best_score = res[1]

  best_i = float(best_params[0])
  best_j = float(best_params[1])
  best_k = int(round(best_params[2]))
  best_r = int(round(best_params[3]))

  # 最大化問題の場合はスコアの符号を元に戻す
  if target in (OptimizationTarget.MAXIMIZE_10_PERCENTILE,
                OptimizationTarget.MAXIMIZE_50_PERCENTILE):
    original_score = -float(best_score)
  else:
    original_score = float(best_score)

  return best_i, best_j, best_k, best_r, original_score
