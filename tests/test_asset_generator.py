import hashlib

import numpy as np
import pytest

from src.lib.asset_generator import (Asset, AssetConfigType, CpiAsset,
                                     Distribution, SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)


def test_slide_adjusted_cpi_behavior():
  """
    SlideAdjustedCpiAsset が以下の挙動を示すことを検証する：
    1. インフレ時：ベースCPIの上昇率から slide_rate/12 が差し引かれる。
    2. デフレ時：名目下限措置によりリターンが 0 になり、価格が維持される。
    """
  n_sim = 1
  n_months = 2
  seed = 42

  # 1ヶ月目は 1% インフレ、2ヶ月目は 1% デフレになるような CPI 分布をシミュレート
  class ManualDist(Distribution):

    def generate(self, shape, seed):
      return np.array([[0.01, -0.01]])  # 1% inflation, 1% deflation

  base_cpi = CpiAsset(name="BaseCPI", dist=ManualDist())
  # 年率 6% (月率 0.5%) のスライド調整
  slide_cpi = SlideAdjustedCpiAsset(name="SlideCPI",
                                    base_cpi="BaseCPI",
                                    slide_rate=0.06)

  configs: list[AssetConfigType] = [base_cpi, slide_cpi]
  prices = generate_monthly_asset_prices(configs,
                                         n_paths=n_sim,
                                         n_months=n_months,
                                         seed=seed)

  base_prices = prices["BaseCPI"][0]  # [1.0, 1.01, 1.01 * 0.99 = 0.9999]
  slide_prices = prices["SlideCPI"][0]

  # 1ヶ月目: base_ret = 0.01, slide = 0.005 -> adj_ret = 0.005. Price = 1.005
  assert slide_prices[1] == pytest.approx(1.005)

  # 2ヶ月目: base_ret = -0.01, slide = 0.005 -> adj_ret = max(0, -0.01 - 0.005) = 0.
  # Price = 1.005 * 1.0 = 1.005
  assert slide_prices[2] == pytest.approx(1.005)

  # 比較として BaseCPI は下がっているはず
  assert base_prices[2] == pytest.approx(1.01 * 0.99)


def test_slide_adjusted_cpi_dependency_error():
  """base_cpi が見つからない場合にエラーになることを検証する"""
  slide_cpi = SlideAdjustedCpiAsset(name="SlideCPI",
                                    base_cpi="MissingCPI",
                                    slide_rate=0.005)
  with pytest.raises(
      ValueError,
      match="Base CPI 'MissingCPI' for slide adjusted CPI 'SlideCPI' not found."
  ):
    configs: list[AssetConfigType] = [slide_cpi]
    generate_monthly_asset_prices(configs, n_paths=1, n_months=1, seed=42)
