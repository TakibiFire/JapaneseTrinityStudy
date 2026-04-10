"""
ダイナミックリバランスに関連するユーティリティ関数を提供するモジュール。

このモジュールは、支出率(S)と目標年数(N)から生存確率を最大化するための
最適なオルカン比率を計算する近似式を提供します。
"""

import numpy as np


def calculate_optimal_strategy(
    s_rate: np.ndarray,
    remaining_years: float,
    base_yield: float = 0.04,
    tax_rate: float = 0.20315,
    inflation_rate: float = 0.0177
) -> np.ndarray:
  """
  支出率 (S) と残り年数 (N) から最適なオルカン比率を計算する。

  Args:
    s_rate: 年間支出額 / 純資産 (shape: (n_sim,))
    remaining_years: 残り年数 (スカラ)
    base_yield: 無リスク資産の利回り
    tax_rate: 譲渡所得税率
    inflation_rate: インフレ率

  Returns:
    np.ndarray: 推奨されるオルカン比率 (shape: (n_sim,))
  """
  # 実質利回り (税引後利回り - インフレ率)
  r_eff = base_yield * (1.0 - tax_rate)
  i_ln = np.log(1.0 + inflation_rate)
  delta = r_eff - i_ln

  # S が 0 の場合のゼロ除算を回避
  s_safe = np.maximum(s_rate, 1e-10)

  # 1. 資産寿命 (N_ruin) の計算
  # S <= delta の場合は理論上無限 (999年とする)
  log_arg = np.maximum(1.0 - delta / s_safe, 1e-10)
  n_ruin = np.where(s_safe <= delta, 999.0, np.log(log_arg) / (-delta))

  # 近似式用の変数 (tools/optimal_ratio_calc.js に準拠)
  n = remaining_years / 60.0
  m = (remaining_years - n_ruin) / 60.0

  ratio = np.zeros_like(s_safe)

  # 極端に大きな S をクリップして overflow を防ぐ (例: 純資産がほぼ0の場合 S が巨大になる)
  s_clipped = np.clip(s_safe, 1e-10, 100.0)

  # ガード付きの対数関数
  def safe_log(val, epsilon=1e-10):
    return np.log(np.maximum(val, epsilon))

  # Region 1: N <= n_ruin (資産寿命内)
  mask1 = remaining_years <= n_ruin
  if np.any(mask1):
    s1 = s_clipped[mask1]
    n_safe = np.maximum(n, 0.001)
    # g_ratio(S, n) = -0.8634 -0.7437 * log(n*S) +0.2169 * n^2 +0.0505 * 1/n -2.1119 * exp(S)
    ratio[mask1] = (
        -0.8634
        - 0.7437 * safe_log(n_safe * s1)
        + 0.2169 * (n_safe**2)
        + 0.0505 * (1.0 / n_safe)
        - 2.1119 * np.exp(s1)
    )

  # Region 2: N > n_ruin (資産寿命超)
  mask2 = ~mask1
  if np.any(mask2):
    s2 = s_clipped[mask2]
    m2 = np.maximum(m[mask2], 1e-10)
    n_safe = np.maximum(n, 0.001)
    # h_ratio(S, m) = -0.1543 +0.1227 * log(m) +0.6476 * log(S) -0.0538 * 1/(n*S) -1.4565 * log(n*S)
    ratio[mask2] = (
        -0.1543
        + 0.1227 * safe_log(m2)
        + 0.6476 * safe_log(s2)
        - 0.0538 * (1.0 / (n_safe * s2))
        - 1.4565 * safe_log(n_safe * s2)
    )

  return np.clip(ratio, 0.0, 1.0)
