"""
src/lib/dynamic_rebalance.py の単体テスト。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.lib.dynamic_rebalance import (calculate_optimal_strategy,
                                       calculate_safe_target_ratio)


def test_calculate_safe_target_ratio_default():
  # rem_years=35.0 で約 0.036332 になることを確認
  # (0.04 * (1 - 0.20315) - log(1 + 0.0177)) / (1 - exp(-delta * 35))
  # delta = 0.031874 - 0.017545 = 0.014329
  # s_safe = 0.014329 / (1 - exp(-0.014329 * 35)) = 0.014329 / 0.39439 = 0.036332
  assert calculate_safe_target_ratio(35.0) == pytest.approx(0.036332, abs=1e-5)


def test_calculate_safe_target_ratio_custom():
  # 10年、利回り5%、税金なし、インフレなし
  # delta = 0.05
  # s_safe = 0.05 / (1 - exp(-0.05 * 10)) = 0.05 / (1 - 0.60653) = 0.05 / 0.39347 = 0.12707
  ratio = calculate_safe_target_ratio(10.0,
                                      base_yield=0.05,
                                      tax_rate=0.0,
                                      inflation_rate=0.0)
  assert ratio == pytest.approx(0.12707, abs=1e-5)


def test_calculate_safe_target_ratio_long_term():
  # 非常に長い期間では実質利回りに近づくはず
  # delta = 0.014329
  ratio = calculate_safe_target_ratio(10000.0)
  # s_safe = delta / (1 - exp(-delta * 10000)) approx delta
  assert ratio == pytest.approx(0.014329, abs=1e-6)


def test_calculate_optimal_strategy_region1():
  """
  資産寿命内 (Region 1) の計算テスト。
  S=0.02 (2%), N=20 の場合、実測値は 0.85 なので 0.05 以内の誤差を期待。
  """
  s_rate = np.array([0.02])
  remaining_years = 20.0
  ratio = calculate_optimal_strategy(s_rate, remaining_years)

  assert isinstance(ratio, np.ndarray)
  assert ratio.shape == (1,)
  assert 0.0 <= ratio[0] <= 1.0
  # 実測値 0.85 に対して 0.05 以内の誤差を許容
  assert abs(ratio[0] - 0.85) < 0.05


def test_calculate_optimal_strategy_region2():
  """
  資産寿命超 (Region 2) の計算テスト。
  S=0.05 (5%), N=50 の場合、無リスク資産だけでは枯渇するため、
  オルカン比率は高くなるはず。
  """
  s_rate = np.array([0.05])
  remaining_years = 50.0
  ratio = calculate_optimal_strategy(s_rate, remaining_years)

  assert 0.0 <= ratio[0] <= 1.0
  # 5% 50年なら高比率なはず
  assert ratio[0] > 0.5


def test_calculate_optimal_strategy_vectorized():
  """
  ベクトル化された入力に対するテスト。
  """
  s_rate = np.array([0.02, 0.05, 0.10])
  remaining_years = 30.0
  ratio = calculate_optimal_strategy(s_rate, remaining_years)

  assert ratio.shape == (3,)
  assert np.all(ratio >= 0.0)
  assert np.all(ratio <= 1.0)


def test_calculate_optimal_strategy_clamping():
  """
  クランプ処理のテスト。
  """
  # 極端な値を入れてみる
  s_rate = np.array([0.0001, 0.5])
  remaining_years = 100.0
  ratio = calculate_optimal_strategy(s_rate, remaining_years)

  assert np.all(ratio >= 0.0)
  assert np.all(ratio <= 1.0)


def test_calculate_optimal_strategy_cpi_177():
  """
  CPI=1.77% での資産寿命計算の妥当性確認。
  delta = 4% * (1 - 0.20315) - ln(1 + 0.0177)
        = 0.031874 - 0.017545 = 0.014329
  S = 0.014 なら delta より小さいので ruin は 999
  S = 0.02 なら delta より大きいので ruin は有限
  """
  delta = 0.04 * (1.0 - 0.20315) - np.log(1.0 + 0.0177)

  # S <= delta
  s_rate_inf = np.array([delta * 0.9])
  # 内部的に n_ruin = 999 になるはずなので、Region 1
  ratio_inf = calculate_optimal_strategy(s_rate_inf, 50.0)

  # S > delta
  s_rate_fin = np.array([delta * 1.1])
  ratio_fin = calculate_optimal_strategy(s_rate_fin, 50.0)

  assert ratio_inf.shape == (1,)
  assert ratio_fin.shape == (1,)


def test_calculate_optimal_strategy_accuracy():
  """
  data/optimal_orukan_ratio.csv の実測データとの一致度を確認する。
  予測値と実測値の差が 0.05 以内のデータが 90% 以上であることを期待する。
  (近似式なのである程度の誤差は許容する)
  """
  csv_path = "data/optimal_orukan_ratio.csv"
  if not os.path.exists(csv_path):
    pytest.skip(f"{csv_path} not found")

  df = pd.read_csv(csv_path)

  spend_ratios = df.iloc[:, 0].values
  years = [int(c) for c in df.columns[1:]]

  total_count = 0
  match_count = 0

  for i, s in enumerate(spend_ratios):
    # s が文字列の場合があるので float に変換
    s_val = float(s)
    actual_ratios = df.iloc[i, 1:].values.astype(float)

    for j, y in enumerate(years):
      pred = calculate_optimal_strategy(np.array([s_val]), float(y))[0]
      actual = actual_ratios[j]

      if abs(pred - actual) < 0.05: # 許容誤差を 0.05 に締める
        match_count += 1
      total_count += 1

  accuracy = match_count / total_count
  print(f"Accuracy (diff < 0.05): {accuracy:.2%}")
  # 90% 以上の一致を期待
  assert accuracy > 0.90
