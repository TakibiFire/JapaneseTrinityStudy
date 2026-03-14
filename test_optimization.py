import numpy as np
import pytest

from core import Asset, generate_monthly_asset_prices
from optimization import (OptimizationTarget, evaluate_strategy,
                          optimize_strategy)


@pytest.fixture
def dummy_monthly_prices():
  """
  テスト用の月次資産価格データを生成するフィクスチャ。
  """
  assets = [
      Asset(name="オルカン", yearly_cost=0.0005775, leverage=1),
      Asset(name="レバカン", yearly_cost=0.00422, leverage=2)
  ]
  # 計算量を減らすためにシミュレーション回数(パス数)と期間を最小化
  return generate_monthly_asset_prices(
      assets=assets,
      years=1,  # 1年間
      n_sim=10,  # 10パス
      seed=42)


def test_evaluate_strategy_constraints(dummy_monthly_prices):
  """
  最適化関数における (i + j <= 1.0) の制約が正しく機能し、
  違反した際に float('inf') を返すか検証する。
  """
  target = OptimizationTarget.MINIMIZE_RUIN_PROBABILITY
  # 1.0を超える場合
  score = evaluate_strategy((0.6, 0.5, 0.0, 12), dummy_monthly_prices, target)
  assert score == float('inf')

  # 1.0ぴったりの場合 (許容されるはず)
  score = evaluate_strategy((0.5, 0.5, 0.0, 12), dummy_monthly_prices, target)
  assert score != float('inf')


def test_evaluate_strategy_targets(dummy_monthly_prices):
  """
  各最適化目的（破産確率の最小化、各パーセンタイルの最大化）に対して、
  適切な型のスコアが計算され返されるかを検証する。
  """
  params = (0.5, 0.5, 5.0, 12)  # i=0.5, j=0.5, k=5, r=12

  # 破産確率の最小化 (float値を返す)
  score_ruin = evaluate_strategy(params, dummy_monthly_prices,
                                 OptimizationTarget.MINIMIZE_RUIN_PROBABILITY)
  assert isinstance(score_ruin, float)
  assert 0.0 <= score_ruin <= 100.0

  # 10%ileの最大化 (内部では負の値として最小化問題になる)
  score_10p = evaluate_strategy(params, dummy_monthly_prices,
                                OptimizationTarget.MAXIMIZE_10_PERCENTILE)
  assert isinstance(score_10p, float)

  # 50%ileの最大化
  score_50p = evaluate_strategy(params, dummy_monthly_prices,
                                OptimizationTarget.MAXIMIZE_50_PERCENTILE)
  assert isinstance(score_50p, float)


def test_optimize_strategy(dummy_monthly_prices):
  """
  scipy.optimize.brute を用いた全体最適化関数を小規模データで実行し、
  各最適化パラメータとスコアが妥当な範囲に収まっているか検証する。
  """
  # テストの実行時間を短縮するため、ダミーの小規模データを使用する
  # 破産確率の最小化を対象にテスト
  target = OptimizationTarget.MINIMIZE_RUIN_PROBABILITY
  best_i, best_j, best_k, best_r, best_score = optimize_strategy(
      dummy_monthly_prices, target)

  # 制約条件 i + j <= 1.0 が満たされているか
  assert best_i + best_j <= 1.0 + 1e-9

  # kの範囲チェック (0 ~ 5)
  assert 0 <= best_k <= 5

  # 破産確率は0以上100以下
  assert 0.0 <= best_score <= 100.0
