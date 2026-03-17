"""
ダイナミックリバランスに関連するユーティリティ関数を提供するモジュール。
"""

import numpy as np


def calculate_optimal_strategy(S: np.ndarray, N: float) -> np.ndarray:
  """
  支出率 (S) と残り年数 (N) から最適なオルカン比率を計算する。
  
  Args:
    S: 年間支出額 / 純資産 (shape: (n_sim,))
    N: 残り年数 (スカラ)
    
  Returns:
    np.ndarray: 推奨されるオルカン比率 (shape: (n_sim,))
  """
  r_base = 0.04
  tax = 0.20315
  inflation = 0.02

  # 実質利回り (税引後利回り - インフレ率)
  r_eff = r_base * (1.0 - tax)
  i_ln = np.log(1.0 + inflation)
  delta = r_eff - i_ln

  # S が 0 の場合のゼロ除算を回避
  S_safe = np.maximum(S, 1e-10)

  # 1. 資産寿命 (N_ruin) の計算
  # S <= delta の場合は理論上無限 (999年とする)
  n_ruin = np.where(S_safe <= delta, 999.0,
                    np.log(1.0 - delta / S_safe) / (-delta))

  n = N / 50.0
  m = (N - n_ruin) / 50.0

  ratio = np.zeros_like(S_safe)

  # Region 1: N <= n_ruin (資産寿命内)
  mask1 = N <= n_ruin
  if np.any(mask1):
    S1 = S_safe[mask1]
    # n が極端に小さい場合のゼロ除算回避
    n_safe = max(n, 1e-6)
    ratio[mask1] = -0.8088 - 0.3832 * np.log(n_safe * S1) + 0.1134 * (
        1 / n_safe) - 0.2017 * np.log(S1) - 1.4146 * np.exp(S1)

  # Region 2: N > n_ruin (資産寿命超)
  mask2 = ~mask1
  if np.any(mask2):
    S2 = S_safe[mask2]
    m2 = np.maximum(m[mask2], 0.0001)
    ratio[mask2] = +0.6431 + 0.1640 * np.log(m2) - 0.0194 * (
        1 / S2) + 0.8301 * np.exp(S2) + 0.2235 * np.sqrt(m2)

  return np.clip(ratio, 0.0, 1.0)
