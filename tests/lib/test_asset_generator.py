import graphlib
from typing import List, Union

import numpy as np
import pytest
from scipy import stats

from src.lib.asset_generator import (Asset, AssetConfig, CpiAsset,
                                     DerivedAsset, ForexAsset,
                                     MonthlyARLogNormal, MonthlyDist,
                                     MonthlyLogDist, MonthlyLogNormal,
                                     MonthlySimpleNormal, YearlyLogNormal,
                                     YearlyLogNormalArithmetic,
                                     YearlySimpleNormal,
                                     generate_monthly_asset_prices)


def test_distribution_simple_normal():
  """MonthlySimpleNormal が指定した平均と標準偏差でリターンを生成することを確認する。"""
  mu, sigma = 0.01, 0.05
  dist = MonthlySimpleNormal(mu=mu, sigma=sigma)
  n_paths, n_months = 10000, 12
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)

  # 平均と標準偏差が許容範囲内であることを確認
  assert np.mean(returns) == pytest.approx(mu, abs=0.002)
  assert np.std(returns) == pytest.approx(sigma, abs=0.002)


def test_distribution_log_normal():
  """YearlyLogNormal が年次パラメータを正しく月次に変換して生成することを確認する。"""
  # 年次期待リターン 7%, ボラティリティ 15%
  mu_annual, sigma_annual = 0.07, 0.15
  dist = YearlyLogNormal(mu=mu_annual, sigma=sigma_annual)
  n_paths, n_months = 100000, 1
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)

  # 理論的な月次平均: exp((mu - 0.5*sigma^2)*dt + 0.5 * (sigma^2 * dt)) - 1 = exp(mu*dt) - 1
  dt = 1.0 / 12.0
  expected_mean = np.exp(mu_annual * dt) - 1.0
  # 理論的な月次分散: (exp(sigma^2 * dt) - 1) * exp(2*mu*dt)
  expected_var = (np.exp(sigma_annual**2 * dt) - 1.0) * np.exp(
      2 * mu_annual * dt)
  expected_std = np.sqrt(expected_var)

  assert np.mean(returns) == pytest.approx(expected_mean, abs=0.001)
  assert np.std(returns) == pytest.approx(expected_std, abs=0.001)


def test_distribution_monthly_log_normal():
  """MonthlyLogNormal がパラメータを正しく使用して生成することを確認する。"""
  mu, sigma = 0.01, 0.05
  dist = MonthlyLogNormal(mu=mu, sigma=sigma)
  n_paths, n_months = 100000, 1
  seed = 42

  ret = dist.generate((n_paths, n_months), seed)
  assert ret.shape == (n_paths, n_months)
  # 理論平均: exp(mu - 0.5*sigma^2 + 0.5*sigma^2) - 1 = exp(mu) - 1
  expected_mean = np.exp(mu) - 1.0
  np.testing.assert_allclose(np.mean(ret), expected_mean, rtol=0.1)


def test_distribution_yearly_simple_normal():
  """YearlySimpleNormal が年次パラメータから単純リターンを生成することを確認する。"""
  mu_annual, sigma_annual = 0.07, 0.15
  dist = YearlySimpleNormal(mu=mu_annual, sigma=sigma_annual)
  n_paths, n_months = 1000000, 1
  seed = 42

  ret = dist.generate((n_paths, n_months), seed)
  assert ret.shape == (n_paths, n_months)
  # 月次の平均は (1 + mu_annual)**(1/12) - 1, 標準偏差は 0.15 / sqrt(12) に近いはず
  np.testing.assert_allclose(np.mean(ret), (1 + mu_annual)**(1 / 12) - 1,
                             rtol=0.01)
  np.testing.assert_allclose(np.std(ret), 0.15 / np.sqrt(12), rtol=0.01)


def test_distribution_yearly_log_normal_arithmetic():
  """YearlyLogNormalArithmetic が算術平均から正しく幾何リターンを生成することを確認する。"""
  mu_annual, sigma_annual = 0.07, 0.15
  dist = YearlyLogNormalArithmetic(mu=mu_annual, sigma=sigma_annual)
  n_paths, n_months = 100000, 1
  seed = 42

  ret = dist.generate((n_paths, n_months), seed)
  assert ret.shape == (n_paths, n_months)
  # 期待される算術リターン（月次）は近似的に mu_annual / 12 になるが、
  # より正確には (1 + mu_annual)**(1/12) - 1 に近い。
  np.testing.assert_allclose(np.mean(ret), (1 + mu_annual)**(1 / 12) - 1,
                             rtol=0.1)


def test_distribution_scipy():
  """MonthlyDist が scipy.stats の分布を利用できることを確認する。"""
  # 自由度 5 の t 分布
  df_params = 5
  dist = MonthlyDist(stats.t, params=(df_params,))
  n_paths, n_months = 10000, 12
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)

  # t分布(df=5)の期待値は 0
  assert np.mean(returns) == pytest.approx(0, abs=0.05)
  # t分布(df=5)の標準偏差は sqrt(df / (df - 2)) = sqrt(5/3) = 1.29...
  assert np.std(returns) == pytest.approx(np.sqrt(5 / 3), abs=0.05)


def test_distribution_monthly_log_dist():
  """MonthlyLogDist が対数リターンから単利リターンへの変換を正しく行うことを確認する。"""
  # 正規分布をそのままラップした場合、対数リターン r が出力される
  # 単利リターン R = exp(r) - 1 になることを確認
  mu, sigma = 0.05, 0.0  # ボラティリティ0 (r = 0.05 固定)
  dist = MonthlyLogDist(stats.norm, params=(mu, sigma))
  n_paths, n_months = 10, 1
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)
  
  expected_simple_return = np.exp(mu) - 1.0
  np.testing.assert_allclose(returns, expected_simple_return, rtol=1e-5)


def test_engine_derived_log_correlation():
  """DerivedAsset において log_correlation=True が正しく対数リターン上で計算を行うことを確認する。"""
  # ベース資産: 対数リターン 0.05 (単利 exp(0.05)-1)
  # DerivedAsset: multiplier 2.0, ノイズ -0.01 (対数リターンのノイズ)
  # 期待される対数リターン = 0.05 * 2.0 - 0.01 = 0.09
  # 期待される単利リターン = exp(0.09) - 1
  
  # 単利リターンが入力されるので、MonthlySimpleNormal でベースを生成
  # 0.05の対数リターンに相当する単利リターン = exp(0.05)-1
  base_simple = np.exp(0.05) - 1.0
  
  configs = [
      Asset(name="Base", dist=MonthlySimpleNormal(base_simple, 0.0)),
      DerivedAsset(name="Derived",
                   base="Base",
                   multiplier=2.0,
                   noise_dist=MonthlyDist(stats.norm, params=(-0.01, 0.0)), # ノイズはMonthlyDist(対数リターン直接指定)を使う想定
                   log_correlation=True),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)
  
  # 初期価格1.0なので、1ヶ月後の価格は (1 + 単利リターン)
  expected_derived_price = np.exp(0.09)
  assert prices["Derived"][0, 1] == pytest.approx(expected_derived_price)


def test_engine_topological_sort_chain():
  """A -> B -> C のような深い依存関係が正しく解決されることを確認する。"""
  configs = [
      Asset(name="Base", dist=MonthlySimpleNormal(0.01, 0.0)),  # 固定リターン 1%
      DerivedAsset(name="Intermediate", base="Base", multiplier=2.0),  # 2%
      DerivedAsset(name="Final", base="Intermediate", multiplier=0.5),  # 1%
  ]
  n_paths, n_months = 1, 12
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # 1ヶ月目の価格を確認
  # Base: 1.01
  # Intermediate: 1.0 + (0.01 * 2.0) = 1.02
  # Final: 1.0 + (0.02 * 0.5) = 1.01
  assert prices["Base"][0, 1] == pytest.approx(1.01)
  assert prices["Intermediate"][0, 1] == pytest.approx(1.02)
  assert prices["Final"][0, 1] == pytest.approx(1.01)


def test_engine_cycle_detection():
  """循環依存がある場合にエラーが発生することを確認する。"""
  configs = [
      DerivedAsset(name="A", base="B"),
      DerivedAsset(name="B", base="A"),
  ]
  with pytest.raises(graphlib.CycleError):
    generate_monthly_asset_prices(configs, 1, 1, 42)


def test_engine_missing_dependency():
  """依存する資産が存在しない場合に ValueError が発生することを確認する。"""
  configs = [
      DerivedAsset(name="A", base="NonExistent"),
  ]
  with pytest.raises(
      ValueError,
      match="Base asset 'NonExistent' for derived asset 'A' not found"):
    generate_monthly_asset_prices(configs, 1, 1, 42)


def test_engine_trust_fee():
  """信託報酬が正しく差し引かれることを確認する。"""
  # 月次リターン 1% 固定, 信託報酬 1.2% (月次 0.1%)
  configs = [
      Asset(name="NoFee", dist=MonthlySimpleNormal(0.01, 0.0), trust_fee=0.0),
      Asset(name="WithFee",
            dist=MonthlySimpleNormal(0.01, 0.0),
            trust_fee=0.012),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # NoFee: 1.0 + 0.01 = 1.01
  # WithFee: 1.0 + 0.01 - (0.012 / 12) = 1.009
  assert prices["NoFee"][0, 1] == pytest.approx(1.01)
  assert prices["WithFee"][0, 1] == pytest.approx(1.009)


def test_engine_leverage():
  """レバレッジが正しく適用されることを確認する。"""
  # リターン 1% 固定, レバレッジ 2倍
  configs = [
      Asset(name="Normal", dist=MonthlySimpleNormal(0.01, 0.0), leverage=1.0),
      Asset(name="Leveraged", dist=MonthlySimpleNormal(0.01, 0.0),
            leverage=2.0),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # Normal: 1.0 + 1*0.01 = 1.01
  # Leveraged: 1.0 + 2*0.01 = 1.02
  assert prices["Normal"][0, 1] == pytest.approx(1.01)
  assert prices["Leveraged"][0, 1] == pytest.approx(1.02)


def test_engine_forex():
  """為替が正しく適用されることを確認する。"""
  # 資産リターン 1%, 為替リターン 5%
  configs: List[Union[AssetConfig, ForexAsset, CpiAsset]] = [
      Asset(name="Stock", dist=MonthlySimpleNormal(0.01, 0.0), forex="USDJPY"),
      ForexAsset(name="USDJPY", dist=MonthlySimpleNormal(0.05, 0.0)),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # Stock Multiplier: (1 + 0.01)
  # FX Multiplier: (1 + 0.05)
  # Final Price: 1.0 * 1.01 * 1.05 = 1.0605
  assert prices["Stock"][0, 1] == pytest.approx(1.0605)
  assert prices["USDJPY"][0, 1] == pytest.approx(1.05)


def test_engine_cpi_asset():
  """CpiAsset が期待通りに生成されることを確認する。"""
  # CPIリターン 2%
  configs: List[Union[AssetConfig, ForexAsset, CpiAsset]] = [
      CpiAsset(name="CPI", dist=MonthlySimpleNormal(0.02, 0.0)),
  ]
  n_paths, n_months = 1, 2
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # 1ヶ月目: 1.02, 2ヶ月目: 1.02^2 = 1.0404
  assert prices["CPI"][0, 1] == pytest.approx(1.02)
  assert prices["CPI"][0, 2] == pytest.approx(1.0404)


def test_engine_derived_with_noise():
  """DerivedAsset にノイズが正しく加算されることを確認する。"""
  # Base: 0% 固定
  # Derived: Base * 1.0 + Noise(10% 固定)
  configs = [
      Asset(name="Base", dist=MonthlySimpleNormal(0.0, 0.0)),
      DerivedAsset(name="Derived",
                   base="Base",
                   multiplier=1.0,
                   noise_dist=MonthlySimpleNormal(0.1, 0.0)),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # Base price: 1.0
  # Derived price: 1.0 + (0.0 * 1.0 + 0.1) = 1.1
  assert prices["Base"][0, 1] == pytest.approx(1.0)
  assert prices["Derived"][0, 1] == pytest.approx(1.1)


def test_engine_zero_floor():
  """資産価格が 0 未満にならないことを確認する。"""
  # -200% のリターン
  configs = [
      Asset(name="Crashed", dist=MonthlySimpleNormal(-2.0, 0.0)),
  ]
  n_paths, n_months = 1, 1
  seed = 42

  prices = generate_monthly_asset_prices(configs, n_paths, n_months, seed)

  # 1.0 + (-2.0) = -1.0 -> Floor at 0.0
  assert prices["Crashed"][0, 1] == 0.0
  # その後も 0.0 を維持することを確認
  configs_long = [Asset(name="Crashed", dist=MonthlySimpleNormal(-2.0, 0.0))]
  prices_long = generate_monthly_asset_prices(configs_long, 1, 5, 42)
  assert np.all(prices_long["Crashed"] >= 0.0)
  assert prices_long["Crashed"][0, 1] == 0.0
  assert prices_long["Crashed"][0, 5] == 0.0


def test_engine_derived_none_base():
  """DerivedAsset の base が None の場合にエラーが発生することを確認する。"""
  configs = [DerivedAsset(name="A", base=None)]
  with pytest.raises(ValueError,
                     match="DerivedAsset 'A' must specify a base asset"):
    generate_monthly_asset_prices(configs, 1, 1, 42)


def test_engine_missing_forex_config():
  """指定された Forex が設定リストに存在しない場合にエラーが発生することを確認する。"""
  configs = [Asset(name="A", dist=MonthlySimpleNormal(0, 0), forex="MissingFX")]
  with pytest.raises(ValueError,
                     match="Forex 'MissingFX' for asset 'A' not found"):
    generate_monthly_asset_prices(configs, 1, 1, 42)


def test_distribution_monthly_ar_log_normal():
  """MonthlyARLogNormal が AR プロセスを正しく生成することを確認する。"""
  # AR(1) パラメータ: c=0.01, phi1=0.5, sigma_e=0.01
  # 無条件平均 mu = c / (1 - phi1) = 0.01 / 0.5 = 0.02
  # 無条件分散 var = sigma_e^2 / (1 - phi1^2) = 0.0001 / 0.75 = 0.0001333...
  c, phis, sigma_e = 0.01, [0.5], 0.01
  dist = MonthlyARLogNormal(c=c, phis=phis, sigma_e=sigma_e)
  n_paths, n_months = 1, 10000
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)
  log_returns = np.log(1.0 + returns[0])

  # 無条件平均の確認
  assert np.mean(log_returns) == pytest.approx(0.02, abs=0.001)
  # 無条件標準偏差の確認 (sqrt(0.0001333) = 0.011547)
  assert np.std(log_returns) == pytest.approx(0.0115, abs=0.001)
  # 自己相関係数の確認
  corr = np.corrcoef(log_returns[1:], log_returns[:-1])[0, 1]
  assert corr == pytest.approx(0.5, abs=0.05)


def test_distribution_monthly_ar_initial_y():
  """MonthlyARLogNormal が指定した初期値から正しくスタートすることを確認する。"""
  # phi1 = 1.0 (ランダムウォーク), c = 0, sigma_e = 0
  # ならば初期値がずっと維持されるはず
  c, phis, sigma_e = 0.0, [1.0], 0.0
  initial_y = [0.123]
  dist = MonthlyARLogNormal(c=c,
                             phis=phis,
                             sigma_e=sigma_e,
                             initial_y=initial_y)
  n_paths, n_months = 1, 10
  seed = 42

  returns = dist.generate((n_paths, n_months), seed)
  log_returns = np.log(1.0 + returns[0])

  # 全ての値が 0.123 であることを確認
  assert np.all(log_returns == pytest.approx(0.123))
