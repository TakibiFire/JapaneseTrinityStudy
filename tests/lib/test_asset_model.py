import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.lib import asset_model


def test_process_returns():
  """日次と月次リターンの計算テスト"""
  dates = pd.date_range("2020-01-01", periods=5)
  df = pd.DataFrame({
      "Date": dates,
      "SP500": [100, 101, 102.01, 103.0301, np.nan],
      "ACWI": [50, 50.5, 51.005, np.nan, 52.0251],
      "BTC": [1000, 1100, 1210, 1331, 1464.1]
  })

  # 日次リターンの確認
  daily_rets = asset_model.process_returns(df, "D")
  assert "SP500_simple" in daily_rets.columns
  assert "BTC_log" in daily_rets.columns

  # 100 -> 101 (1%)
  assert pytest.approx(daily_rets["SP500_simple"].iloc[1], 1e-5) == 0.01
  # 1000 -> 1100 (10%)
  assert pytest.approx(daily_rets["BTC_simple"].iloc[1], 1e-5) == 0.10
  # log(1.10)
  assert pytest.approx(daily_rets["BTC_log"].iloc[1], 1e-5) == np.log(1.10)

  # 月次リターンの確認（月をまたぐデータを作成してテストする）
  dates_m = pd.date_range("2020-01-01", periods=60, freq="D")
  df_m = pd.DataFrame({
      "Date": dates_m,
      "SP500": np.linspace(100, 200, 60),
      "ACWI": np.linspace(50, 100, 60),
      "BTC": np.linspace(1000, 2000, 60)
  })
  monthly_rets = asset_model.process_returns(df_m, "ME")
  assert len(monthly_rets) == 2  # Jan, Feb
  assert not pd.isna(monthly_rets["SP500_simple"].iloc[1])


def test_fit_normal_simple():
  """モデルA（単純リターンへの正規分布フィッティング）のテスト"""
  np.random.seed(42)
  data = pd.Series(np.random.normal(loc=0.05, scale=0.1, size=100))
  res = asset_model.fit_normal_simple(data)

  assert "mu" in res
  assert "std" in res
  assert "loglik" in res
  assert "aic" in res
  assert "bic" in res
  assert "mse" in res
  assert res["mu"] == pytest.approx(0.05, abs=0.05)
  assert res["std"] == pytest.approx(0.1, abs=0.05)


def test_fit_normal_log():
  """モデルB（対数リターンへの正規分布フィッティング）のテスト"""
  np.random.seed(42)
  data = pd.Series(np.random.normal(loc=0.02, scale=0.05, size=100))
  res = asset_model.fit_normal_log(data)

  assert "mu" in res
  assert res["mu"] == pytest.approx(0.02, abs=0.05)
  assert res["std"] == pytest.approx(0.05, abs=0.05)


def test_distribution_lists():
  """分布リストの分割が正しく行われているかのテスト"""
  assert len(asset_model.SYMMETRIC_DISTRIBUTIONS) > 0
  assert len(asset_model.ASYMMETRIC_DISTRIBUTIONS) > 0
  assert len(asset_model.ALL_DISTRIBUTIONS) == len(
      asset_model.SYMMETRIC_DISTRIBUTIONS) + len(
          asset_model.ASYMMETRIC_DISTRIBUTIONS)
  # 重複がないことの確認
  assert len(set(asset_model.ALL_DISTRIBUTIONS)) == len(
      asset_model.ALL_DISTRIBUTIONS)


def test_find_best_distribution():
  """モデルC（最適な連続分布探索）のテスト"""
  np.random.seed(42)
  # 明確にラプラス分布に従うデータを生成
  data = pd.Series(stats.laplace.rvs(loc=0.05, scale=0.1, size=500))

  # 探索のテスト
  distributions = [stats.norm, stats.laplace, stats.t]
  res = asset_model.find_best_distribution(data,
                                           distributions=distributions,
                                           bins=20)

  assert res is not None
  assert len(res) > 0
  assert "name" in res[0]
  assert "params" in res[0]
  assert "mse" in res[0]
  assert res[0]["name"] in ["laplace", "t", "norm"]  # 期待されるいずれかが選ばれる

  # 空のデータの場合
  assert asset_model.find_best_distribution(pd.Series([])) is None


def test_find_best_distribution_with_fixed_mean():
  """期待値固定の最適分布探索テスト"""
  np.random.seed(42)
  # 平均0.1のデータを生成
  mu_true = 0.1
  data = pd.Series(stats.norm.rvs(loc=mu_true, scale=0.05, size=500))

  distributions = [stats.norm, stats.johnsonsu]
  res = asset_model.find_best_distribution_with_fixed_mean(
      data, distributions=distributions, bins=20)

  assert res is not None
  assert len(res) > 0
  best_model = res[0]

  # 理論的期待値がデータの経験的平均と一致していることを確認
  emp_mean = data.mean()
  dist_obj = getattr(stats, best_model["name"])
  theo_mean = dist_obj.mean(*best_model["params"])
  assert pytest.approx(theo_mean, 1e-10) == emp_mean

  # 空のデータの場合
  assert asset_model.find_best_distribution_with_fixed_mean(pd.Series(
      [])) is None


def test_find_best_distribution_failures():
  """フィッティング時の例外ハンドリングおよび異常値テスト"""

  class FailingDist:
    name = "failing_dist"

    def fit(self, data, floc=None):
      raise ValueError("Mock fitting error")

  class PDFErrorDist:
    name = "pdferror_dist"

    def fit(self, data, floc=None):
      return (0.0, 1.0)

    def mean(self, *params):
      return 0.0

    def pdf(self, x, *params):
      raise ValueError("Mock PDF error")

  class NaNPDFDist:
    name = "nanpdf_dist"

    def fit(self, data, floc=None):
      return (0.0, 1.0)

    def mean(self, *params):
      return 0.0

    def pdf(self, x, *params):
      return np.full_like(x, np.nan)

  data = pd.Series([1.0, 2.0, 3.0])
  mock_dists = [FailingDist(), PDFErrorDist(), NaNPDFDist()]

  # 通常の分布探索
  res1 = asset_model.find_best_distribution(data, distributions=mock_dists)
  assert res1 is None

  # 平均固定の分布探索
  res2 = asset_model.find_best_distribution_with_fixed_mean(
      data, distributions=mock_dists)
  assert res2 is None


def test_calculate_mrgbm():
  """MR-GBMパラメータ推定のテスト"""
  np.random.seed(42)
  # 平均回帰するデータ（Ornstein-Uhlenbeck過程）を生成
  theta = 2.0
  mu = np.log(100.0)
  sigma = 0.2
  dt = 1.0 / 252.0

  n = 1000
  x = np.zeros(n)
  x[0] = mu
  for t in range(1, n):
    dw = np.random.normal(0, np.sqrt(dt))
    x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dw

  prices = pd.Series(np.exp(x))
  res = asset_model.calculate_mrgbm(prices, dt)

  assert res is not None
  assert "theta" in res
  assert "mu" in res
  assert "sigma" in res

  # 推定値が真値にある程度近いことを確認
  assert res["theta"] > 0
  assert res["mu"] == pytest.approx(mu, abs=0.5)
  assert res["sigma"] == pytest.approx(sigma, abs=0.05)

  # 発散する系列（ランダムウォーク＋トレンド）の場合はNaNが返るはず
  rw_prices = pd.Series(np.exp(np.cumsum(np.random.normal(0.01, 0.01, n))))
  res_rw = asset_model.calculate_mrgbm(rw_prices, dt)
  assert np.isnan(res_rw["theta"])


def test_simulate_annual_stats_simple():
  """単利リターンの年次化シミュレーションテスト"""
  np.random.seed(42)
  # For normal distribution with mean 0.01 and std 0.05
  mu_m, std_m = 0.01, 0.05
  dist = stats.norm
  params = (mu_m, std_m)

  mean_a, std_a = asset_model.simulate_annual_stats_simple(dist,
                                                           params,
                                                           n_sims=100000)

  # Check if mean is close to (1.01)^12 - 1 = 0.1268
  assert np.isclose(mean_a, (1 + mu_m)**12 - 1, rtol=0.1)

  # For std, it's roughly sqrt(12) * std_m = 0.173 but actually higher due to compounding
  expected_std = np.sqrt(((1 + mu_m)**2 + std_m**2)**12 - (1 + mu_m)**24)
  assert np.isclose(std_a, expected_std, rtol=0.1)


def test_simulate_annual_stats_log():
  """対数リターンの年次化シミュレーションテスト"""
  np.random.seed(42)
  # For normal distribution with mean 0.01 and std 0.05
  mu_m, std_m = 0.01, 0.05
  dist = stats.norm
  params = (mu_m, std_m)

  mean_a, std_a = asset_model.simulate_annual_stats_log(dist,
                                                        params,
                                                        n_sims=100000)

  # Expected value = exp(12*mu_m + 12*std_m^2/2) - 1
  expected_mean = np.exp(12 * mu_m + 12 * (std_m**2) / 2) - 1
  assert np.isclose(mean_a, expected_mean, rtol=0.1)
