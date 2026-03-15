import unittest

import numpy as np
import pandas as pd

from core import (MU, N_SIM, SIGMA, TRADING_DAYS, YEARS, Asset, Cpi,
                  SimulationResult, Strategy, ZeroRiskAsset,
                  generate_cpi_paths, generate_monthly_asset_prices,
                  simulate_strategy)


class TestCore(unittest.TestCase):

  def test_generate_monthly_asset_prices_shape(self):
    """
    generate_monthly_asset_prices が生成する価格配列の形状と初期値が
    想定通りであるか検証する。
    """
    assets = [
        Asset(name="A", yearly_cost=0.01, leverage=1),
        Asset(name="B", yearly_cost=0.02, leverage=2)
    ]
    prices = generate_monthly_asset_prices(assets, years=5, n_sim=10)

    self.assertEqual(len(prices), 2)
    self.assertIn("A", prices)
    self.assertIn("B", prices)

    # 5年 * 12ヶ月 + 1(初期値) = 61ヶ月
    expected_shape = (10, 5 * 12 + 1)
    self.assertEqual(prices["A"].shape, expected_shape)
    self.assertEqual(prices["B"].shape, expected_shape)

    # 初期値は全て1.0
    self.assertTrue(np.allclose(prices["A"][:, 0], 1.0))
    self.assertTrue(np.allclose(prices["B"][:, 0], 1.0))

  def test_simulate_strategy_no_ruin(self):
    """
    資産価格が全く変動せず、取り崩しも行わない状況において、
    初期資金がそのまま維持され破産しないことを検証する。
    """
    # 資産は常に一定の価値 (ボラティリティ0, リターン0) となるようにする
    assets = [Asset(name="Safe", yearly_cost=0.0, leverage=1)]

    # 乱数の影響を排除するため、価格配列を自作
    n_sim = 5
    years = 2
    total_months = years * 12
    prices = {"Safe": np.ones((n_sim, total_months + 1))}

    strategy = Strategy(name="Test",
                        initial_money=100.0,
                        initial_loan=0.0,
                        yearly_loan_interest=0.0,
                        initial_asset_ratio={"Safe": 1.0},
                        annual_cost=0.0,
                        inflation_rate=0.0,
                        selling_priority=["Safe"])

    res = simulate_strategy(strategy, prices)
    net_values = res.net_values

    # 全く減らないので、初期資金がそのまま残るはず
    self.assertEqual(net_values.shape, (n_sim,))
    self.assertTrue(np.allclose(net_values, 100.0))

  def test_simulate_strategy_bankruptcy(self):
    """
    資産価値が急激に低下し、総資産が初期借入額を下回るケースにおいて、
    正しく破産 (最終純資産が 0) と判定されるか検証する。
    """
    # 資産価値が急激に下がるシナリオ
    assets = [Asset(name="Risky", yearly_cost=0.0, leverage=1)]

    n_sim = 3
    years = 1
    total_months = years * 12

    # 価格が1ヶ月目で半減し、その後も下がり続ける
    prices_array = np.zeros((n_sim, total_months + 1))
    prices_array[:, 0] = 1.0
    for i in range(1, total_months + 1):
      prices_array[:, i] = 0.5**i

    prices = {"Risky": prices_array}

    # 借入が大きく、価格低下ですぐに破産する
    strategy = Strategy(
        name="BankruptTest",
        initial_money=50.0,
        initial_loan=50.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={"Risky": 1.0},  # 総資金100.0すべてを投資
        annual_cost=0.0,
        inflation_rate=0.0,
        selling_priority=["Risky"])

    res = simulate_strategy(strategy, prices)
    net_values = res.net_values

    # 全員破産するため 0 になるはず
    self.assertTrue(np.allclose(net_values, 0.0))

  def test_simulate_strategy_cost_withdrawal(self):
    """
    毎月の生活費取り崩しによって資産が売却され、
    最終的な純資産から正確に取り崩し額が差し引かれているか検証する。
    """
    n_sim = 2
    years = 1
    total_months = 12

    # 価格は変動なし
    prices_array = np.ones((n_sim, total_months + 1))
    prices = {"AssetA": prices_array}

    strategy = Strategy(
        name="WithdrawalTest",
        initial_money=120.0,
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={"AssetA": 1.0},  # 全額投資、現金0
        annual_cost=12.0,  # 月1.0の取り崩し
        inflation_rate=0.0,
        selling_priority=["AssetA"])

    # 毎月1.0取り崩すので、12ヶ月で12.0減る
    # 最終的な純資産は 120.0 - 12.0 = 108.0
    res = simulate_strategy(strategy, prices)
    net_values = res.net_values

    self.assertTrue(np.allclose(net_values, 108.0))

  def test_generate_prices_leverage(self):
    """
    レバレッジ倍率の異なる資産に対し、生成される価格に
    期待通りのレバレッジ効果が反映されているかを検証する。
    """
    assets = [
        Asset(name="1x", yearly_cost=0.0, leverage=1, mu=0.1, sigma=0.0),
        Asset(name="2x", yearly_cost=0.0, leverage=2, mu=0.1, sigma=0.0)
    ]
    # ボラティリティ0で確実に上がるように設定
    prices = generate_monthly_asset_prices(assets, years=1, n_sim=1)

    # 最終月の価格を比較
    final_1x = prices["1x"][0, -1]
    final_2x = prices["2x"][0, -1]

    # 2倍レバレッジの方が最終価格が高くなるはず (リターンが正の場合)
    self.assertGreater(final_2x, final_1x)

  def test_simulate_strategy_tax(self):
    """
    資産売却時に、取得単価から正しく譲渡益が計算され、
    年末にその税額が翌年の必要現金に加算されているかを検証する。
    """
    n_sim = 1
    years = 2
    total_months = 24

    # 価格推移を設定
    # 初期価格 1.0 -> 1ヶ月目に 2.0 になり、その後ずっと 2.0 となる
    prices_array = np.full((n_sim, total_months + 1), 2.0)
    prices_array[:, 0] = 1.0
    prices = {"AssetA": prices_array}

    # 初期資産 200 を全額投資 (AssetA を 200口 保有)
    # 生活費として毎月 10 を取り崩す
    strategy = Strategy(
        name="TaxTest",
        initial_money=200.0,
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={"AssetA": 1.0},
        annual_cost=120.0,  # 月10.0の取り崩し
        inflation_rate=0.0,
        selling_priority=["AssetA"],
        tax_rate=0.2  # 20%の税率で計算を分かりやすくする
    )

    # [1年目の動き]
    # 各月 (1月〜12月) に現金が -10 になり、資産を売却する。
    # 現在の価格は 2.0 なので、売却額 10 を得るために必要な口数は 5口。
    # 取得単価は 1.0 なので、取得費は 5口 * 1.0 = 5.0。
    # 1ヶ月あたりの譲渡益 = 10 - 5 = 5.0。
    # これを12回繰り返すため、1年目の年間譲渡益 = 5.0 * 12 = 60.0。
    # 1年目の年末 (12月末) に確定する税金 = 60.0 * 0.2 = 12.0。

    # [2年目の動き]
    # 2年目の1月 (m=12) の取り崩し時に、生活費 10 に加えて前年の税金 12.0 が加算され、22.0 の現金が必要になる。
    # 資産価格は 2.0 なので、22.0 の現金を補填するために 11口 が売却される。
    # 取得費は 11口 * 1.0 = 11.0、譲渡益 = 22.0 - 11.0 = 11.0。
    # 2年目の2月〜12月は通常の生活費 10 の取り崩し (売却 5口、譲渡益 5.0)。
    # したがって、2年目に売却される合計口数は 11 + 5 * 11 = 66口。

    # [全体の残高計算]
    # 初期口数 = 200口
    # 1年目の売却口数 = 60口 (残 140口)
    # 2年目の売却口数 = 66口 (残 74口)
    # 2年末の総資産 = 残り 74口 * 2.0 + 現金 0 = 148.0

    res = simulate_strategy(strategy, prices)
    net_values = res.net_values
    self.assertTrue(np.allclose(net_values, 148.0))

  def test_simulate_strategy_rebalance(self):
    """
    指定された間隔でのリバランスにおいて、超過した資産の売却と
    不足した資産の購入、およびそれに伴う取得単価や税金の更新が
    正しく動作するかを検証する。
    """
    n_sim = 1
    years = 2
    total_months = 24

    prices_A = np.ones((n_sim, total_months + 1))
    prices_B = np.ones((n_sim, total_months + 1))

    # 1年目末 (m=11) に価格が変動
    # Aは2倍(2.0), Bは半分(0.5)になる。その後変動なし。
    prices_A[:, 12:] = 2.0
    prices_B[:, 12:] = 0.5

    prices = {"A": prices_A, "B": prices_B}

    # 初期資産100を50:50で投資 (A:50口, B:50口)
    strategy = Strategy(
        name="RebalanceTest",
        initial_money=100.0,
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={
            "A": 0.5,
            "B": 0.5
        },
        annual_cost=0.0,
        inflation_rate=0.0,
        selling_priority=["A", "B"],
        tax_rate=0.2,  # 税率20%
        rebalance_interval=12)

    # [1年目末 (m=11, 価格はprices[:, 12])]
    # Aの評価額 = 50 * 2.0 = 100.0
    # Bの評価額 = 50 * 0.5 = 25.0
    # 総純資産 = 125.0
    # 目標額はそれぞれ 125.0 * 0.5 = 62.5
    #
    # Aを売却: (100 - 62.5) = 37.5 分売却。価格2.0なので 18.75口。
    # 取得単価は1.0。譲渡益 = 37.5 - 18.75 * 1.0 = 18.75
    # Aの残口数 = 31.25口
    # 税金はm=11に確定: 18.75 * 0.2 = 3.75
    #
    # Bを購入: 37.5 分購入。価格0.5なので 75.0口。
    # Bの新しい口数 = 50 + 75 = 125.0口
    # Bの新しい平均取得単価 = (50 * 1.0 + 75 * 0.5) / 125.0 = 87.5 / 125.0 = 0.7
    #
    # [2年目初頭 (m=12)]
    # 1年目の税金3.75を支払うために資産を売却
    # 優先順位Aから売却。価格は2.0。3.75分売るには 1.875口。
    # 譲渡益 = 3.75 - 1.875 * 1.0 = 1.875
    # Aの残口数 = 31.25 - 1.875 = 29.375口
    #
    # [2年目末 (m=23)]
    # リバランスが発生。
    # Aの評価額 = 29.375 * 2.0 = 58.75
    # Bの評価額 = 125.0 * 0.5 = 62.5
    # 総純資産 = 121.25
    # 目標額はそれぞれ 60.625
    #
    # Bを売却: (62.5 - 60.625) = 1.875 分売却。価格0.5なので 3.75口。
    # Bの平均取得単価は0.7。取得費 = 3.75 * 0.7 = 2.625
    # 譲渡益 = 1.875 - 2.625 = -0.75
    #
    # Aを購入: 1.875 分購入。価格2.0なので 0.9375口。
    #
    # 最終的な総純資産 = 121.25 (この月のリバランスによる税金は翌年支払うため、純資産から引かれない)

    res = simulate_strategy(strategy, prices)
    net_values = res.net_values
    self.assertTrue(np.allclose(net_values, 121.25))

  def test_sustained_months_tracking(self):
    """
    破産が発生した月が sustained_months に正しく記録されるか検証する。
    """
    n_sim = 3
    years = 10
    total_months = years * 12

    # 価格推移を自作
    prices_array = np.ones((n_sim, total_months + 1))

    # パス0: 即破産 (現金不足で資産を売っても足りない状況を作る)
    # パス1: 5年(60ヶ月)で破産
    # パス2: 破産しない

    prices_array[0, 1:] = 0.0001  # ほぼ価値なし
    prices_array[1, 61:] = 0.0001  # 61ヶ月目から価値なし

    prices = {"Asset": prices_array}

    strategy = Strategy(
        name="SustainedTest",
        initial_money=10.0,
        initial_loan=90.0,  # 借入比率が高い
        yearly_loan_interest=0.0,
        initial_asset_ratio={"Asset": 1.0},
        annual_cost=0.0,
        inflation_rate=0.0,
        selling_priority=["Asset"])

    res = simulate_strategy(strategy, prices)
    sustained = res.sustained_months

    # パス0は m=0 で破産判定されるはず (総資産 100*0.0001 = 0.01 < 借入 90)
    self.assertEqual(sustained[0], 0)
    # パス1は m=60 で破産判定されるはず (総資産 100*0.0001 = 0.01 < 借入 90)
    self.assertEqual(sustained[1], 60)
    # パス2は最後まで破産しない
    self.assertEqual(sustained[2], total_months)

  def test_simulate_strategy_zero_risk_asset(self):
    """
    無リスク資産(ZeroRiskAsset)が含まれる場合、利回りが正しく計算されて現金に加算され、
    価格変動がなくキャピタルゲイン税がかからないことを検証する。
    """
    n_sim = 1
    years = 1
    total_months = 12

    # 乱数生成のダミー(今回は使用しないが関数呼び出し用に必要)
    prices = {"DummyAsset": np.ones((n_sim, total_months + 1))}

    # 100万円投資し、年利12% (月利1%)、税率20%
    # 毎月の利回りは 100 * 0.01 * (1 - 0.2) = 0.8万円
    # 年間で 0.8 * 12 = 9.6万円 が現金に加算される
    zr_asset = ZeroRiskAsset(name="Cash", yield_rate=0.12)

    strategy = Strategy(name="ZeroRiskTest",
                        initial_money=100.0,
                        initial_loan=0.0,
                        yearly_loan_interest=0.0,
                        initial_asset_ratio={zr_asset: 1.0},
                        annual_cost=0.0,
                        inflation_rate=0.0,
                        selling_priority=["Cash"],
                        tax_rate=0.2)

    res = simulate_strategy(strategy, prices)
    net_values = res.net_values

    # 最終的な純資産総額は 100.0 + 9.6 = 109.6 になるはず
    self.assertTrue(np.allclose(net_values, 109.6))

  def test_strategy_validation(self):
    """
    selling_priority に initial_asset_ratio に含まれない資産が指定された場合、
    ValueError が発生することを検証する。
    """
    with self.assertRaises(ValueError):
      Strategy(
          name="InvalidPriority",
          initial_money=100.0,
          initial_loan=0.0,
          yearly_loan_interest=0.0,
          initial_asset_ratio={"AssetA": 1.0},
          annual_cost=0.0,
          inflation_rate=0.0,
          selling_priority=["AssetB"]  # AssetAが存在するのにAssetBを指定
      )

    # ZeroRiskAssetの場合はそのnameで検証されるべき
    zr = ZeroRiskAsset("Cash", 0.05)
    with self.assertRaises(ValueError):
      Strategy(name="InvalidPriorityZR",
               initial_money=100.0,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={zr: 1.0},
               annual_cost=0.0,
               inflation_rate=0.0,
               selling_priority=["NotCash"])

    # 正しい場合はエラーにならない
    Strategy(name="ValidPriority",
             initial_money=100.0,
             initial_loan=0.0,
             yearly_loan_interest=0.0,
             initial_asset_ratio={
                 zr: 0.5,
                 "AssetA": 0.5
             },
             annual_cost=0.0,
             inflation_rate=0.0,
             selling_priority=["Cash", "AssetA"])

  def test_generate_cpi_paths(self):
    """
    generate_cpi_paths が生成するCPIパスの形状と初期値が想定通りか検証する。
    """
    cpis = [
        Cpi(name="Normal", mu=0.02, sigma=0.05),
        Cpi(name="Flat", mu=0.0, sigma=0.0)
    ]
    paths = generate_cpi_paths(cpis, years=2, n_sim=10)

    self.assertEqual(len(paths), 2)
    self.assertIn("Normal", paths)
    self.assertIn("Flat", paths)

    # 2年 * 12ヶ月 + 1(初期値) = 25ヶ月
    expected_shape = (10, 2 * 12 + 1)
    self.assertEqual(paths["Normal"].shape, expected_shape)
    self.assertEqual(paths["Flat"].shape, expected_shape)

    # 初期値は全て1.0
    self.assertTrue(np.allclose(paths["Normal"][:, 0], 1.0))
    self.assertTrue(np.allclose(paths["Flat"][:, 0], 1.0))

    # Flat は変動なし (全て1.0) のはず
    self.assertTrue(np.allclose(paths["Flat"], 1.0))


if __name__ == "__main__":
  unittest.main()
  unittest.main()
  unittest.main()
