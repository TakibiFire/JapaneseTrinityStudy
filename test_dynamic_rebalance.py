import unittest

import numpy as np

from dynamic_rebalance import calculate_optimal_strategy


class TestDynamicRebalance(unittest.TestCase):

  def test_calculate_optimal_strategy_consistency(self):
    # S=0.04, N=30 のとき
    # S = 0.04 の場合、n_ruin = 約 29.758年
    # したがって N=30 > n_ruin となり、Region 2 (資産寿命超) の計算式が適用される。
    # m = (30 - 29.758) / 50 = 0.00484
    # ratio = +0.6431 + 0.1640 * log(m) - 0.0194 * (1 / S) + 0.8301 * exp(S) + 0.2235 * sqrt(m)
    #       = 0.6431 + 0.1640 * log(0.00484) - 0.0194 * 25 + 0.8301 * exp(0.04) + 0.2235 * sqrt(0.00484)
    #       = 0.6431 - 0.8742 - 0.4850 + 0.8640 + 0.0155 = 0.1634
    S = np.array([0.04])
    N = 30.0
    ratio = calculate_optimal_strategy(S, N)
    self.assertAlmostEqual(ratio[0], 0.1629, places=3)

    # S=0.04, N=20 のとき (Region 1)
    # n = 0.4
    # S = 0.04
    # ratio = -0.8088 - 0.3832 * log(0.016) + 0.1134 * 2.5 - 0.2017 * log(0.04) - 1.4146 * exp(0.04)
    # ratio = -0.8088 - 0.3832 * (-4.1352) + 0.2835 - 0.2017 * (-3.2189) - 1.4146 * 1.0408
    # ratio = -0.8088 + 1.5846 + 0.2835 + 0.6493 - 1.4723 = 0.2363
    S = np.array([0.04])
    N = 20.0
    ratio = calculate_optimal_strategy(S, N)
    self.assertAlmostEqual(ratio[0], 0.2363, places=3)


if __name__ == "__main__":
  unittest.main()
