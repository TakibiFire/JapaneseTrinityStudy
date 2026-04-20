import numpy as np
import pytest

from src.core import (DynamicSpending, SimulationResult, Strategy,
                      ZeroRiskAsset, simulate_strategy)
from src.lib.cashflow_generator import CashflowRule, CashflowType


def test_simulate_strategy_no_ruin():
  """
  資産価格が全く変動せず、取り崩しも行わない状況において、
  初期資金がそのまま維持され破産しないことを検証する。
  """
  n_sim = 5
  total_months = 24
  prices = {"Safe": np.ones((n_sim, total_months + 1))}

  strategy = Strategy(name="Test",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Safe": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["Safe"])

  res = simulate_strategy(strategy, prices)
  net_values = res.net_values

  # 全く減らないので、初期資金がそのまま残るはず
  assert net_values.shape == (n_sim,)
  assert np.allclose(net_values, 100.0)


def test_simulate_strategy_bankruptcy():
  """
  資産価値が急激に低下し、総資産が初期借入額を下回るケースにおいて、
  正しく破産 (最終純資産が 0未満) と判定されるか検証する。
  """
  n_sim = 3
  total_months = 12

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
      inflation_rate=None,
      selling_priority=["Risky"])

  res = simulate_strategy(strategy, prices)
  net_values = res.net_values

  # 全員破産するため、最終純資産は表示上初期借入額を下回る額(実質破綻)になっているはず。
  # 破産した時点の評価額 - initial_loan になる。
  # 1ヶ月目で 100*0.5 = 50 になり、loan=50 と同じだが厳密に < 50 ではない。
  # 2ヶ月目で 100*0.25 = 25 になり 25 < 50 で破産確定。
  # sustained_months は 1 (m=1, 2ヶ月目終了時) になるはず。
  assert np.all(res.sustained_months == 1)


def test_simulate_strategy_cost_withdrawal():
  """
  毎月の生活費取り崩しによって資産が売却され、
  最終的な純資産から正確に取り崩し額が差し引かれているか検証する。
  """
  n_sim = 2
  total_months = 12

  # 価格は変動なし
  prices_array = np.ones((n_sim, total_months + 1))
  prices = {"AssetA": prices_array}

  strategy = Strategy(
      name="WithdrawalTest",
      initial_money=120.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"AssetA": 1.0},
      annual_cost=12.0,  # 月1.0の取り崩し
      inflation_rate=None,
      selling_priority=["AssetA"])

  # 毎月1.0取り崩すので、12ヶ月で12.0減る
  # 最終的な純資産は 120.0 - 12.0 = 108.0
  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 108.0)


def test_simulate_strategy_annual_cost_list():
  """
  毎月の生活費取り崩し額がリストで指定された場合、
  各年の取り崩し額が正確に差し引かれているか検証する。
  """
  n_sim = 2
  total_months = 24

  prices_array = np.ones((n_sim, total_months + 1))
  prices = {"AssetA": prices_array}

  annual_cost_list = [12.0] * 50
  annual_cost_list[0] = 12.0
  annual_cost_list[1] = 24.0

  strategy = Strategy(
      name="WithdrawalListTest",
      initial_money=120.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"AssetA": 1.0},
      annual_cost=annual_cost_list,  # 1年目は月1.0、2年目は月2.0の取り崩し
      inflation_rate=0.0,  # inflation_rate=0.0(float)のテストも兼ねる
      selling_priority=["AssetA"])

  # 1年目で12.0、2年目で24.0取り崩すので、24ヶ月で36.0減る
  # inflation_rate=0.0 なので cpi_multiplier は常に 1.0
  # 最終的な純資産は 120.0 - 36.0 = 84.0
  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 84.0)


def test_annual_cost_validation():
  """
  annual_costの入力バリデーションを検証する（一部はStrategyに組み込まれずそのまま通る仕様もあるが、将来のために残す）。
  """
  # 異常系：空リスト（本実装では1年分もないとエラーになる。1年未満だとm//12でIndexError）
  strategy = Strategy(name="EmptyCostList",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=[],
                      inflation_rate=None,
                      selling_priority=["A"])
  prices = {"A": np.ones((1, 13))}
  with pytest.raises(IndexError):
    simulate_strategy(strategy, prices)


def test_dynamic_spending_validation():
  """
  DynamicSpendingを用いた場合のStrategy初期化時のバリデーションを検証する。
  """
  spending = DynamicSpending(initial_annual_spend=4.0,
                             target_ratio=0.04,
                             upper_limit=0.05,
                             lower_limit=-0.015)

  # 正常系: inflation_rate が None
  Strategy(name="ValidDynamicSpending",
           initial_money=100.0,
           initial_loan=0.0,
           yearly_loan_interest=0.0,
           initial_asset_ratio={"A": 1.0},
           annual_cost=spending,
           inflation_rate=None,
           selling_priority=["A"])


def test_dynamic_spending_initial_cost():
  """
  DynamicSpendingを用いた場合、初年度の年間支出額が
  initial_annual_spend として使用されることを検証する。
  """
  n_sim = 1
  total_months = 12
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 初期資産 100.0, ターゲット 5%, 初動 5.0 -> 初年度支出は 5.0
  spending = DynamicSpending(initial_annual_spend=5.0,
                             target_ratio=0.05,
                             upper_limit=0.05,
                             lower_limit=-0.015)
  strategy = Strategy(name="TestSpendingInitial",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=spending,
                      inflation_rate=None,
                      selling_priority=["A"])

  # 年間5.0取り崩すため、最終資産は 100.0 - 5.0 = 95.0
  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 95.0)


def test_dynamic_spending_ceiling():
  """
  DynamicSpendingを用いた場合、資産が急増しても
  前年の支出額 * (1 + upper_limit) に支出が抑えられることを検証する。
  """
  n_sim = 1
  total_months = 24

  # m=12 (1年目最後の取り崩し) 以降で価格が10倍に急増するシナリオ
  prices_array = np.ones((n_sim, total_months + 1))
  prices_array[:, 12:] = 10.0
  prices = {"A": prices_array}

  # 初期資産 100.0, ターゲット 5%, 上限 5%
  # 税金の影響を排除するため tax_rate=0 とする
  spending = DynamicSpending(initial_annual_spend=5.0,
                             target_ratio=0.05,
                             upper_limit=0.05,
                             lower_limit=-0.015)
  strategy = Strategy(name="TestSpendingCeiling",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=spending,
                      inflation_rate=None,
                      selling_priority=["A"],
                      tax_rate=0.0)

  # 1年目: m=0..10 は price 1.0。売却口数 = (5/12)*11 = 4.583333
  # m=11 は price 10.0。売却口数 = (5/12)/10 = 0.0416666
  # 1年目合計売却口数 = 4.625。残口数 = 95.375
  # 2年目年始評価額 = 95.375 * 10 = 953.75
  # 2年目支出 = min(953.75 * 0.05, 5.0 * 1.05) = min(47.6875, 5.25) = 5.25
  # 2年目売却口数 = 5.25 / 10 = 0.525
  # 最終口数 = 95.375 - 0.525 = 94.85
  # 最終評価額 = 94.85 * 10 = 948.5
  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 948.5)


def test_dynamic_spending_floor():
  """
  DynamicSpendingを用いた場合、資産が急落しても
  前年の支出額 * (1 + lower_limit) までしか支出が減らないことを検証する。
  """
  n_sim = 1
  total_months = 24

  # 1年目は価格変動なし
  # 2年目(m=12以降)に価格が半減するシナリオ
  prices_array = np.ones((n_sim, total_months + 1))
  prices_array[:, 12:] = 0.5
  prices = {"A": prices_array}

  # 初期資産 100.0, ターゲット 5%, 下限 -2%
  spending = DynamicSpending(initial_annual_spend=5.0,
                             target_ratio=0.05,
                             upper_limit=0.05,
                             lower_limit=-0.02)
  strategy = Strategy(name="TestSpendingFloor",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=spending,
                      inflation_rate=None,
                      selling_priority=["A"],
                      tax_rate=0.0)

  # 1年目: m=0..10 は price 1.0。売却口数 = 4.583333
  # m=11 は price 0.5。売却口数 = (5/12)/0.5 = 0.833333
  # 1年目合計売却口数 = 5.416666。残口数 = 94.583333
  # 2年目年始評価額 = 94.583333 * 0.5 = 47.291666
  # 2年目目標支出 = 47.291666 * 0.05 = 2.364583
  # 2年目下限支出 = 5.0 * (1 - 0.02) = 4.90
  # 2年目支出 = 4.90
  # 2年目売却口数 = 4.90 / 0.5 = 9.8
  # 最終口数 = 94.583333 - 9.8 = 84.783333
  # 最終評価額 = 84.783333 * 0.5 = 42.391666
  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 42.391666666666666)


def test_dynamic_spending_three_years():
  """
  DynamicSpendingを用いた場合、3年間の資産の増減に対し、
  天井(ceiling)と床(floor)の両方が年ごとに正しく適用されることを検証する。
  """
  n_sim = 1
  total_months = 36

  prices_array = np.ones((n_sim, total_months + 1))
  prices_array[:, 12:24] = 10.0
  prices_array[:, 24:] = 0.5
  prices = {"A": prices_array}

  spending = DynamicSpending(initial_annual_spend=5.0,
                             target_ratio=0.05,
                             upper_limit=0.05,
                             lower_limit=-0.02)
  strategy = Strategy(name="TestSpendingThreeYears",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=spending,
                      inflation_rate=None,
                      selling_priority=["A"],
                      tax_rate=0.0)

  # 1年目(m=0..11)
  # m=0..10 は price 1.0。売却口数 = 4.583333...
  # m=11 は price 10.0。売却口数 = (5/12)/10 = 0.041666...
  # 1年目合計売却口数 = 4.625。残口数 = 95.375
  # 2年目年始評価額(m=12) = 95.375 * 10 = 953.75
  #
  # 2年目(m=12..23)
  # 2年目目標支出 = 953.75 * 0.05 = 47.6875
  # 2年目上限支出 = 5.0 * 1.05 = 5.25 -> 上限適用
  # m=12..22 は price 10.0。売却口数 = (5.25/12)*11 / 10 = 0.48125
  # m=23 は price 0.5。売却口数 = (5.25/12) / 0.5 = 0.875
  # 2年目合計売却口数 = 1.35625。残口数 = 95.375 - 1.35625 = 94.01875
  # 3年目年始評価額(m=24) = 94.01875 * 0.5 = 47.009375
  #
  # 3年目(m=24..35)
  # 3年目目標支出 = 47.009375 * 0.05 = 2.35046875
  # 3年目下限支出 = 5.25 * (1 - 0.02) = 5.145 -> 下限適用
  # 3年目年間支出 = 5.145
  # m=24..35 は price 0.5。売却口数 = 5.145 / 0.5 = 10.29
  # 最終口数 = 94.01875 - 10.29 = 83.72875
  # 最終評価額(m=36) = 83.72875 * 0.5 = 41.864375

  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 41.864375)


def test_simulate_strategy_tax():
  """
  資産売却時に、取得単価から正しく譲渡益が計算され、
  年末にその税額が翌年の必要現金に加算されているかを検証する。
  """
  n_sim = 1
  total_months = 24

  # 初期価格 1.0 -> 1ヶ月目に 2.0 になり、その後ずっと 2.0 となる
  prices_array = np.full((n_sim, total_months + 1), 2.0)
  prices_array[:, 0] = 1.0
  prices = {"AssetA": prices_array}

  strategy = Strategy(
      name="TaxTest",
      initial_money=200.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"AssetA": 1.0},
      annual_cost=120.0,  # 月10.0の取り崩し
      inflation_rate=None,
      selling_priority=["AssetA"],
      tax_rate=0.2)

  # [1年目]
  # 毎月 10 を取り崩す。価格 2.0 なので 5口売却。取得費 5.0。譲渡益 5.0。
  # 1年間の譲渡益 = 5.0 * 12 = 60.0。
  # 12月末に確定する税金 = 60.0 * 0.2 = 12.0。
  #
  # [2年目]
  # m=12 (2年目1月) で 生活費10 + 前年税金12 = 22 を取り崩す。
  # 22/2 = 11口売却。
  # m=13〜23 (2月〜12月) は 通常通り10を取り崩す(5口 * 11ヶ月 = 55口)。
  # 合計売却口数: 1年目 60口 + 2年目 66口 = 126口。
  # 残口数: 200 - 126 = 74口。
  # 最終資産: 74 * 2.0 = 148.0

  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 148.0)


def test_simulate_strategy_rebalance():
  """
  指定された間隔でのリバランスにおいて、超過した資産の売却と
  不足した資産の購入が正しく動作するかを検証する。
  """
  n_sim = 1
  total_months = 24

  prices_A = np.ones((n_sim, total_months + 1))
  prices_B = np.ones((n_sim, total_months + 1))

  # 1年目末 (m=11) に価格が変動
  prices_A[:, 12:] = 2.0
  prices_B[:, 12:] = 0.5
  prices = {"A": prices_A, "B": prices_B}

  strategy = Strategy(name="RebalanceTest",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={
                          "A": 0.5,
                          "B": 0.5
                      },
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A", "B"],
                      tax_rate=0.2,
                      rebalance_interval=12)

  # [1年目末 (m=11)]
  # Aの評価額 = 50 * 2.0 = 100.0
  # Bの評価額 = 50 * 0.5 = 25.0
  # 総純資産 = 125.0
  # 目標額はそれぞれ 62.5
  # Aを 37.5 分売却。価格2.0なので 18.75口。譲渡益 18.75。A残口数 31.25口。
  # Bを 37.5 分購入。価格0.5なので 75.0口。B残口数 125.0口。
  # 税金はm=11に確定: 18.75 * 0.2 = 3.75
  #
  # [2年目初頭 (m=12)]
  # 税金3.75を支払うためにAから売却。価格2.0。1.875口売却。
  # A残口数 = 29.375口。
  #
  # [2年目末 (m=23)] リバランス
  # A評価額 = 29.375 * 2.0 = 58.75
  # B評価額 = 125.0 * 0.5 = 62.5
  # 総資産 = 121.25 (税金による差引き済み)

  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 121.25)


def test_sustained_months_tracking():
  """
  破産が発生した月が sustained_months に正しく記録されるか検証する。
  """
  n_sim = 3
  total_months = 120

  prices_array = np.ones((n_sim, total_months + 1))
  prices_array[0, 1:] = 0.0001
  prices_array[1, 61:] = 0.0001
  prices = {"Asset": prices_array}

  strategy = Strategy(name="SustainedTest",
                      initial_money=10.0,
                      initial_loan=90.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Asset": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["Asset"])

  res = simulate_strategy(strategy, prices)
  sustained = res.sustained_months

  assert sustained[0] == 0
  assert sustained[1] == 60
  assert sustained[2] == total_months


def test_simulate_strategy_zero_risk_asset():
  """
  無リスク資産(ZeroRiskAsset)が含まれる場合、利回りが正しく計算されて現金に加算され、
  価格変動がなくキャピタルゲイン税がかからないことを検証する。
  """
  n_sim = 1
  total_months = 12

  prices = {"DummyAsset": np.ones((n_sim, total_months + 1))}

  # 100万円投資し、年利12% (月利1%)、税率20%
  # 毎月の利回りは 100 * 0.01 * (1 - 0.2) = 0.8万円
  # 年間で 0.8 * 12 = 9.6万円
  zr_asset = ZeroRiskAsset(name="Cash", yield_rate=0.12)
  strategy = Strategy(name="ZeroRiskTest",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={zr_asset: 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["Cash"],
                      tax_rate=0.2)

  res = simulate_strategy(strategy, prices)
  assert np.allclose(res.net_values, 109.6)


def test_strategy_validation():
  """
  selling_priority に initial_asset_ratio に含まれない資産が指定された場合のエラー検証。
  """
  with pytest.raises(ValueError, match="Missing: {'AssetA'}"):
    Strategy(name="InvalidPriority",
             initial_money=100.0,
             initial_loan=0.0,
             yearly_loan_interest=0.0,
             initial_asset_ratio={"AssetA": 1.0},
             annual_cost=0.0,
             inflation_rate=None,
             selling_priority=["AssetB"])

  zr = ZeroRiskAsset("Cash", 0.05)
  with pytest.raises(ValueError, match="Missing: {'Cash'}"):
    Strategy(name="InvalidPriorityZR",
             initial_money=100.0,
             initial_loan=0.0,
             yearly_loan_interest=0.0,
             initial_asset_ratio={zr: 1.0},
             annual_cost=0.0,
             inflation_rate=None,
             selling_priority=["NotCash"])


def test_simulate_with_dynamic_rebalance():
  """
  ダイナミックリバランス用のコールバック関数が正しく機能するか検証する。
  """
  orukan = "オルカン"
  cash_asset = ZeroRiskAsset(name="無リスク資産", yield_rate=0.0)

  prices = {"オルカン": np.ones((10, 61)), "無リスク資産": np.ones((10, 61))}

  def dummy_rebalance_fn(net_value, annual_spend, remaining_years, post_tax_net):
    return {
        "オルカン": np.full_like(net_value, 0.5),
        "無リスク資産": np.full_like(net_value, 0.5)
    }

  strategy = Strategy(name="Test",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={
                          orukan: 1.0,
                          cash_asset: 0.0
                      },
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["オルカン", "無リスク資産"],
                      rebalance_interval=1,
                      dynamic_rebalance_fn=dummy_rebalance_fn)

  res = simulate_strategy(strategy, prices)
  assert len(res.net_values) == 10
  assert np.allclose(res.net_values, 1000.0)


def test_simulate_strategy_with_cpi_asset():
  """
  CpiAssetとして生成されたインフレパスが正しく適用されるかを検証する。
  """
  strategy = Strategy(name="TestCPIPath",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Stock": 1.0},
                      annual_cost=12.0,
                      inflation_rate="Japan_CPI",
                      selling_priority=["Stock"])

  monthly_prices = {
      "Stock": np.ones((1, 2)),
      "Japan_CPI":
          np.array([[1.0, 1.05]])  # 1ヶ月目に5%インフレ
  }

  res = simulate_strategy(strategy, monthly_prices)
  # 初期: Stock 100
  # month 0: cost = 1.0 * CPI(1.0) = 1.0
  #   Stock残り: 99
  # 終了後純資産: 99
  assert res.net_values[0] == pytest.approx(99.0)


def test_simulate_strategy_empty_prices():
  """monthly_asset_pricesが空の場合のフォールバック動作"""
  strategy = Strategy(name="TestEmpty",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={ZeroRiskAsset("Cash", 0.0): 1.0},
                      annual_cost=12.0,
                      inflation_rate=None,
                      selling_priority=["Cash"])
  res = simulate_strategy(strategy, {},
                          fallback_n_sim=2,
                          fallback_total_months=5)
  assert res.net_values.shape == (2,)
  assert res.sustained_months.shape == (2,)


def test_simulate_strategy_missing_cpi():
  """指定したCPIパスが存在しない場合のエラー"""
  strategy = Strategy(name="Test",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Stock": 1.0},
                      annual_cost=0.0,
                      inflation_rate="MissingCPI",
                      selling_priority=["Stock"])
  with pytest.raises(
      ValueError,
      match="CPI path 'MissingCPI' not found in monthly_asset_prices."):
    simulate_strategy(strategy, {"Stock": np.ones((1, 2))})


def test_simulate_strategy_extra_cashflow():
  """
  追加のキャッシュフロー（年金収入、一時的支出）が正しく反映されるかを検証する。
  """
  n_sim = 1
  total_months = 12

  prices = {"AssetA": np.ones((n_sim, total_months + 1))}

  # 毎月10.0の支出
  # 追加キャッシュフロー:
  # month=0: +20.0 (収入) -> 余った10.0が現金プールに追加される
  # month=1: -50.0 (支出) -> 10(定常支出) + 50(一時支出) = 60必要。前月の余り10を使っても50足りないので、50口売却される

  extra_cf = np.zeros(total_months)
  extra_cf[0] = 20.0
  extra_cf[1] = -50.0

  monthly_cashflows = {"MyCashflow": extra_cf}

  strategy = Strategy(
      name="ExtraCashflowTest",
      initial_money=100.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"AssetA": 1.0},
      annual_cost=120.0,  # 月10.0
      inflation_rate=None,
      selling_priority=["AssetA"],
      cashflow_rules=[CashflowRule("MyCashflow", CashflowType.EXTRAORDINARY)])

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # 初期状態:
  # AssetA: 100口 (評価額100)
  # Cash: 0

  # Month 0:
  # 定常支出 = 10
  # 追加CF = +20
  # 必要な現金 = 10 - 20 = -10 (10の余裕ができる)
  # Cash = 0 - (-10) = +10
  # 売却なし。AssetA = 100口

  # Month 1:
  # 定常支出 = 10
  # 追加CF = -50
  # 必要な現金 = 10 - (-50) = 60
  # Cash = 10 - 60 = -50 (50足りない)
  # 売却 = 50口 (価格1.0)
  # Cash = 0
  # AssetA = 50口

  # Month 2~11 (残り10ヶ月):
  # 毎月10の取り崩し -> 合計100口売却必要だが、AssetAは50口しかないので途中で破産する。

  # 確認: どこで破産するか？
  # Month 2: 40口残る
  # Month 3: 30口残る
  # Month 4: 20口残る
  # Month 5: 10口残る
  # Month 6: 0口残る
  # Month 7: 資産がマイナスになり破産

  assert res.sustained_months[0] == 7


def test_extra_cashflow_validation():
  """追加キャッシュフロー関連の入力チェック"""
  
  strategy = Strategy(name="ShapeTest",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("CF1", CashflowType.EXTRAORDINARY)])
                      
  prices = {"A": np.ones((2, 13))}  # n_sim=2, total_months=12
  
  # 存在しないキャッシュフロー名
  with pytest.raises(ValueError, match="Cashflow source 'CF1' not found in monthly_cashflows."):
    simulate_strategy(strategy, prices, monthly_cashflows={"CF2": np.zeros(12)})
    
  # shapeが異なる(長さが合わない)
  with pytest.raises(ValueError, match="Cashflow source 'CF1' has invalid shape"):
    simulate_strategy(strategy, prices, monthly_cashflows={"CF1": np.zeros(10)})
    
  # shapeが異なる(n_simが合わない)
  with pytest.raises(ValueError, match="Cashflow source 'CF1' has invalid shape"):
    simulate_strategy(strategy, prices, monthly_cashflows={"CF1": np.zeros((3, 12))})


def test_simulate_strategy_large_income_and_spend():
  """大きな収入でキャッシュプールが増え、大きな支出で正しく資産が売却されることを検証"""
  n_sim = 1
  total_months = 12

  prices = {"AssetA": np.ones((n_sim, total_months + 1))}
  prices["AssetA"][:, :] = 1.0  # 価格は1.0固定
  
  extra_cf = np.zeros(total_months)
  extra_cf[0] = 1000.0   # 月0: 1000の巨大収入
  extra_cf[2] = -500.0   # 月2: 500の巨大支出
  
  monthly_cashflows = {"BigCF": extra_cf}

  strategy = Strategy(name="LargeCFTest",
                      initial_money=100.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"AssetA": 1.0},
                      annual_cost=12.0,  # 月1.0の取り崩し
                      inflation_rate=None,
                      selling_priority=["AssetA"],
                      cashflow_rules=[CashflowRule("BigCF", CashflowType.EXTRAORDINARY)])

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)
  
  # 初期状態: AssetA=100口, Cash=0
  
  # 月0: 支出1.0, 収入1000.0 -> Cash = 0 - 1.0 + 1000.0 = +999.0
  # AssetA売却なし = 100口
  
  # 月1: 支出1.0 -> Cash = 999.0 - 1.0 = +998.0
  # AssetA売却なし = 100口
  
  # 月2: 支出1.0, 支出500.0 -> Cash = 998.0 - 1.0 - 500.0 = +497.0
  # AssetA売却なし = 100口
  
  # その後、月3〜11(残り9ヶ月)は毎月1.0の支出
  # 最終的なCash = 497.0 - 9.0 = 488.0
  # 最終的なAssetA = 100.0
  # 最終的な純資産 = 488.0 + 100.0 = 588.0
  
  assert np.allclose(res.net_values, 588.0)
  assert np.all(res.sustained_months == 12)  # 破産していない


def test_simulate_strategy_ndarray_initial_money():
  """
  initial_money に np.ndarray を指定した場合、各パスで異なる初期資金が
  正しく反映されることを検証する。
  """
  n_sim = 5
  total_months = 12
  prices = {"AssetA": np.ones((n_sim, total_months + 1))}

  # 各パスで異なる初期資金: [100, 200, 300, 400, 500]
  initial_money = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
  strategy = Strategy(name="NdArrayInitTest",
                      initial_money=initial_money,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"AssetA": 1.0},
                      annual_cost=12.0,  # 月1.0の取り崩し
                      inflation_rate=None,
                      selling_priority=["AssetA"])

  res = simulate_strategy(strategy, prices)

  # 各パスから 12.0 ずつ差し引かれるはず
  expected = initial_money - 12.0
  assert res.net_values.shape == (n_sim,)
  assert np.allclose(res.net_values, expected)


def test_simulate_strategy_ndarray_initial_money_shape_mismatch():
  """
  initial_money の配列サイズが n_sim と一致しない場合にエラーが発生することを検証。
  """
  n_sim = 5
  prices = {"A": np.ones((n_sim, 13))}
  initial_money = np.array([100.0, 200.0])  # n_sim=5 に対してサイズが合わない

  strategy = Strategy(name="ShapeMismatchTest",
                      initial_money=initial_money,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A"])

  with pytest.raises(ValueError, match="initial_money array shape"):
    simulate_strategy(strategy, prices)


def test_extra_cashflow_multiplier():
  """
  倍率関数が正しく適用されることを確認する。
  """
  n_sim = 1
  total_months = 12

  monthly_asset_prices = {"オルカン": np.ones((n_sim, total_months + 1))}
  monthly_cashflows = {"Job": np.full((n_sim, total_months), 10.0)}

  # 資産が 1050 未満なら 働く (1.0), 1050 以上なら 働かない (0.0)
  # 倍率関数は m % 12 == 0 でのみ評価される
  def multiplier_fn(m, net_worth, prev_net, prev_gross):
    return (net_worth < 1050.0).astype(float)

  strategy = Strategy(
      name="Conditional CF",
      initial_money=1000.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"オルカン": 1.0},
      annual_cost=0.0,
      inflation_rate=None,
      selling_priority=["オルカン"],
      cashflow_rules=[CashflowRule("Job", CashflowType.EXTRAORDINARY, multiplier_fn)])

  # m=0: NW=1000 -> Mult=1.0 for the year.
  res = simulate_strategy(strategy, monthly_asset_prices, monthly_cashflows)
  assert res.net_values[0] == 1120.0


def test_extra_cashflow_mixed():
  """
  静的と動的なキャッシュフローが混在しても正しく計算されることを確認する。
  """
  n_sim = 1
  total_months = 12

  monthly_asset_prices = {"オルカン": np.ones((n_sim, total_months + 1))}
  monthly_cashflows = {
      "Pension": np.full((n_sim, total_months), 5.0),
      "Job": np.full((n_sim, total_months), 10.0)
  }

  strategy = Strategy(
      name="Mixed CF",
      initial_money=1000.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"オルカン": 1.0},
      annual_cost=0.0,
      inflation_rate=None,
      selling_priority=["オルカン"],
      cashflow_rules=[
          CashflowRule("Pension", CashflowType.EXTRAORDINARY, None),
          CashflowRule("Job", CashflowType.EXTRAORDINARY,
                       lambda m, nw, pn, pg: np.array([0.5]))
      ])

  res = simulate_strategy(strategy, monthly_asset_prices, monthly_cashflows)
  # 1000 + (5 * 12) + (10 * 0.5 * 12) = 1120
  assert res.net_values[0] == 1120.0


def test_extra_cashflow_detailed_timing():
  """
  倍率関数が年1回（m=0, 12, ...）更新され、その年の各月に適用されることを検証する。
  """
  n_sim = 1
  total_months = 24

  monthly_asset_prices = {"A": np.ones((n_sim, total_months + 1))}
  # 毎月 10.0 の収入
  monthly_cashflows = {"Job": np.full((n_sim, total_months), 10.0)}

  # NW に応じて倍率を変える
  # 1年目 (m=0): NW=1000 -> mult=1.0
  # 2年目 (m=12): NW は 1000 + 10*12 = 1120 になっているはず -> mult=2.0
  def multiplier_fn(m, net_worth, prev_net, prev_gross):
    if m == 0:
      return np.array([1.0])
    elif m == 12:
      return np.array([2.0])
    return np.array([0.0])

  strategy = Strategy(name="TimingTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Job", CashflowType.EXTRAORDINARY, multiplier_fn)])

  res = simulate_strategy(strategy, monthly_asset_prices, monthly_cashflows)

  # 1年目の収入: 10 * 1.0 * 12 = 120
  # 2年目の収入: 10 * 2.0 * 12 = 240
  # 合計純資産: 1000 + 120 + 240 = 1360
  assert res.net_values[0] == 1360.0


def test_extra_cashflow_bankrupt_mask():
  """
  破産したパスでは倍率関数が正しく active_paths で制限されていることを検証。
  """
  n_sim = 2
  total_months = 12

  # パス0: 即破産するように
  # 初期 100, 借入 50. 合計 150 投資.
  # 1ヶ月後に価格 0.0001 -> 評価額 0.015.
  # 借入 50 なので、現金が 49.985 以上ないと破産.
  prices_array = np.ones((n_sim, total_months + 1))
  prices_array[0, 1:] = 0.0001
  prices = {"A": prices_array}

  # 収入を月 1.0 にする.
  monthly_cashflows = {"Job": np.full((n_sim, total_months), 1.0)}

  def multiplier_fn(m, net_worth, prev_net, prev_gross):
    # net_worth のサイズは active_paths の数になっているはず
    return np.ones_like(net_worth)

  strategy = Strategy(name="BankruptMaskTest",
                      initial_money=100.0,
                      initial_loan=50.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Job", CashflowType.EXTRAORDINARY, multiplier_fn)])

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # パス0は破産
  assert res.net_values[0] == 0.0
  # パス1はNW = 100 + 12 - 0 = 112
  assert res.net_values[1] == 112.0


def test_extra_cashflow_no_side_effect_on_ds():
  """
  動的支出計算と追加キャッシュフローが正しく共存することを検証。
  """
  n_sim = 1
  total_months = 24

  monthly_asset_prices = {"A": np.ones((n_sim, total_months + 1))}
  monthly_cashflows = {"Job": np.full((n_sim, total_months), 10.0)}

  def mult_fn(m, nw, pn, pg):
    return (nw > 1100.0).astype(float) + 1.0

  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=1.0, lower_limit=-1.0)

  strategy = Strategy(name="DS_Interaction",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=ds,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Job", CashflowType.EXTRAORDINARY, mult_fn)],
                      record_annual_spend=True)

  res = simulate_strategy(strategy, monthly_asset_prices, monthly_cashflows)

  # Year 1 (m=0): NW=1000. spend=100. mult=1.0. Job=120.
  # Year 1 end NW = 1000 - 100 + 120 = 1020.
  # Year 2 (m=12): NW=1020. spend=102. mult=1.0. Job=120.
  # Year 2 end NW = 1020 - 102 + 120 = 1038.
  assert res.annual_spends is not None
  assert res.annual_spends[0, 0] == pytest.approx(100.0)
  assert res.annual_spends[0, 1] == pytest.approx(102.0)
  assert np.allclose(res.net_values[0], 1038.0)


def test_conditional_work_multiplier_receives_prev_spending():
  """
  ExtraCashflowMultiplierFn が前年の年間支出額を正しく受け取れることを検証する。
  """
  n_sim = 1
  years = 2
  initial_money = 1000.0
  annual_cost = 100.0

  # 価格変動なし
  prices = {"Cash": np.ones((n_sim, years * 12 + 1))}

  received_spendings = []

  def mock_multiplier(m, net_worth, prev_net, prev_gross):
    received_spendings.append(prev_net.copy())
    return np.ones_like(net_worth)

  strategy = Strategy(name="Test",
                      initial_money=initial_money,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Cash": 1.0},
                      annual_cost=annual_cost,
                      inflation_rate=None,
                      selling_priority=["Cash"],
                      cashflow_rules=[CashflowRule("Work", CashflowType.EXTRAORDINARY, mock_multiplier)])

  simulate_strategy(strategy, prices, monthly_cashflows={"Work": np.zeros((n_sim, years * 12))})

  # m=0: 初期支出 100.0
  assert np.all(received_spendings[0] == pytest.approx(100.0))
  # m=12: 前年の支出 100.0
  assert np.all(received_spendings[1] == pytest.approx(100.0))


def test_conditional_work_logic():
  """
  資産残高に応じた条件付き労働が正しく資産推移に反映されることを検証する。
  """
  n_sim = 1
  years = 2
  initial_money = 1000.0
  annual_cost = 100.0  # 年間100

  prices = {"Cash": np.ones((n_sim, years * 12 + 1))}

  def work_if_low(m, net_worth, prev_net, prev_gross):
    # NW < 950 の時のみ働く
    # Year 0: NW=1000 -> 働かない
    # Year 1: 1年間の支出100後、NW=900 -> 働く
    return (net_worth < 950).astype(np.float64)

  strategy = Strategy(name="Test",
                      initial_money=initial_money,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Cash": 1.0},
                      annual_cost=annual_cost,
                      inflation_rate=None,
                      selling_priority=["Cash"],
                      cashflow_rules=[CashflowRule("Work", CashflowType.EXTRAORDINARY, work_if_low)])

  # 労働収入: 月50 (年600)
  work_cf = np.full((n_sim, years * 12), 50.0)
  res = simulate_strategy(strategy, prices, monthly_cashflows={"Work": work_cf})

  # Year 0: 収入なし。1年目末の資産 = 1000 - 100 = 900.
  # Year 1: 収入あり(600)。2年目末の資産 = 900 - 100 + 600 = 1400.
  assert res.net_values[0] == pytest.approx(1400.0)


def test_cashflow_type_include_in_annual_spend():
  """
  INCLUDE_IN_ANNUAL_SPEND の収入が年間支出トラッカーを減らし、
  翌年のダイナミックスペンディングの制限に影響を与えることを検証する。
  """
  n_sim = 1
  total_months = 24
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 毎月 5.0 の収入
  monthly_cashflows = {"Income": np.full((n_sim, total_months), 5.0)}

  # ターゲット 10% (1000 * 0.1 = 100/year)
  # 上限 5% (105), 下限 -5% (95)
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=0.05, lower_limit=-0.05)

  strategy = Strategy(name="IncludeTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=ds,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Income", CashflowType.REGULAR)],
                      record_annual_spend=True)

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # Year 1 (m=0..11):
  # ターゲット支出 = 100.0
  # 収入 = 5.0 * 12 = 60.0
  # 正味の定常支出 (net_reg_spend) = 100.0 - 60.0 = 40.0
  # annual_spends[0, 0] は正味の 40.0 になるはず
  assert res.annual_spends is not None
  assert res.annual_spends[0, 0] == pytest.approx(40.0)

  # Year 2 (m=12..23):
  # 年始 NW = 1000 - 40 = 960 (実際は withdrawal は 40)
  # ターゲット支出 = 960 * 0.1 = 96.0
  # 前年基本支出 (prev_base_spend_y) = 100.0 (追加キャッシュフローの影響を受けない)
  # 上限 = 100 * 1.05 = 105.0
  # 下限 = 100 * 0.95 = 95.0
  # ターゲット 96.0 は範囲内なのでそのまま採用
  # 正味の定常支出 = 96.0 (ベース) - 60.0 (収入) = 36.0
  assert res.annual_spends[0, 1] == pytest.approx(36.0)


def test_cashflow_type_isolated_vs_include():
  """
  ISOLATED と INCLUDE_IN_ANNUAL_SPEND の違いを検証する。
  ISOLATED は年間支出トラッカーに影響を与えない。
  """
  n_sim = 1
  total_months = 24
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 毎月 5.0 の収入
  monthly_cashflows = {"Income": np.full((n_sim, total_months), 5.0)}
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=0.05, lower_limit=-0.05)

  # Case 1: ISOLATED
  strategy_iso = Strategy(name="IsoTest",
                          initial_money=1000.0,
                          initial_loan=0.0,
                          yearly_loan_interest=0.0,
                          initial_asset_ratio={"A": 1.0},
                          annual_cost=ds,
                          inflation_rate=None,
                          selling_priority=["A"],
                          cashflow_rules=[CashflowRule("Income", CashflowType.EXTRAORDINARY)],
                          record_annual_spend=True)

  res_iso = simulate_strategy(strategy_iso, prices, monthly_cashflows=monthly_cashflows)

  # Year 1:
  # ターゲット支出 = 100.0
  # 収入 = 60.0 (ISOLATED なので tracker には入らない)
  # 正味支出 tracker = 100.0
  assert res_iso.annual_spends is not None
  assert res_iso.annual_spends[0, 0] == pytest.approx(100.0)
  # Year 2:
  # 年始 NW = 1000 - (100 - 60) = 960
  # ターゲット = 96.0
  # 前年 tracker = 100.0
  # 下限 = 95.0, 上限 = 105.0
  # ターゲット 96.0 は範囲内なのでそのまま採用
  assert res_iso.annual_spends[0, 1] == pytest.approx(96.0)


def test_cashflow_type_include_in_annual_spend_excess_income():
  """
  INCLUDE_IN_ANNUAL_SPEND の収入がベース支出を上回る場合を検証。
  """
  n_sim = 1
  total_months = 12
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 毎月 20.0 の収入 (年 240)
  monthly_cashflows = {"BigIncome": np.full((n_sim, total_months), 20.0)}
  # ターゲット 100/year
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=0.05, lower_limit=-0.05)

  strategy = Strategy(name="ExcessIncomeTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=ds,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("BigIncome", CashflowType.REGULAR)],
                      record_annual_spend=True)

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # Year 1:
  # 支出 100, 収入 240 -> 正味支出 -140
  assert res.annual_spends is not None
  assert res.annual_spends[0, 0] == pytest.approx(-140.0)
  # NW = 1000 - (-140) = 1140
  assert res.net_values[0] == pytest.approx(1140.0)


def test_strategy_duplicate_cashflow_rules():
  """キャッシュフロールールの重複チェックを検証。"""
  with pytest.raises(ValueError, match="Duplicate source_name found in cashflow_rules."):
    Strategy(name="DupTest",
             initial_money=100.0,
             initial_loan=0.0,
             yearly_loan_interest=0.0,
             initial_asset_ratio={"A": 1.0},
             annual_cost=10.0,
             inflation_rate=None,
             selling_priority=["A"],
             cashflow_rules=[
                 CashflowRule("CF1", CashflowType.EXTRAORDINARY),
                 CashflowRule("CF1", CashflowType.REGULAR)
             ])


def test_simulate_strategy_exp_regard_interest_tax_as_regular():
  """利息や税金を定常支出として扱う実験的フラグの検証。"""
  n_sim = 1
  total_months = 12
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 借入1000, 金利 12%/year = 1%/month = 10/month
  # 税金 0
  strategy = Strategy(name="ExpFlagTest",
                      initial_money=1000.0,
                      initial_loan=1000.0,
                      yearly_loan_interest=0.12,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["A"],
                      record_annual_spend=True)

  # フラグ False (デフォルト): 利息は tracker に入らない
  res1 = simulate_strategy(strategy, prices, exp_regard_interest_tax_as_regular=False)
  assert res1.annual_spends is not None
  assert res1.annual_spends[0, 0] == pytest.approx(0.0)

  # フラグ True: 利息 (10 * 12 = 120) が tracker に入る
  res2 = simulate_strategy(strategy, prices, exp_regard_interest_tax_as_regular=True)
  assert res2.annual_spends is not None
  assert res2.annual_spends[0, 0] == pytest.approx(120.0)


def test_cashflow_type_include_in_annual_spend_expense():
  """INCLUDE_IN_ANNUAL_SPEND の支出が DynamicSpending の制限を増やすことを検証。"""
  n_sim = 1
  total_months = 24
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 毎月 5.0 の追加支出 (年 60)
  monthly_cashflows = {"Expense": np.full((n_sim, total_months), -5.0)}
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=0.05, lower_limit=-0.05)

  strategy = Strategy(name="IncludeExpenseTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=ds,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Expense", CashflowType.REGULAR)],
                      record_annual_spend=True)

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # Year 1: 支出 100 + 追加支出 60 = 160
  assert res.annual_spends is not None
  assert res.annual_spends[0, 0] == pytest.approx(160.0)

  # Year 2:
  # 年始 NW = 1000 - 160 = 840. ターゲット = 84.
  # 前年基本支出 (prev_base_spend_y) = 100.0
  # 下限 = 100 * 0.95 = 95.0
  # ターゲット 84 は 下限 95 より小さいので 95 が採用される (ベース支出)
  # 正味の定常支出 = 95.0 (ベース) + 60.0 (追加支出) = 155.0
  assert res.annual_spends[0, 1] == pytest.approx(155.0)


def test_cashflow_type_isolated_expense():
  """ISOLATED の支出が DynamicSpending の制限に影響しないことを検証。"""
  n_sim = 1
  total_months = 24
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 毎月 5.0 の追加支出 (年 60)
  monthly_cashflows = {"Expense": np.full((n_sim, total_months), -5.0)}
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.1, upper_limit=0.05, lower_limit=-0.05)

  strategy = Strategy(name="IsoExpenseTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=ds,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Expense", CashflowType.EXTRAORDINARY)],
                      record_annual_spend=True)

  res = simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # Year 1: tracker はベース支出の 100 のまま
  assert res.annual_spends is not None
  assert res.annual_spends[0, 0] == pytest.approx(100.0)

  # Year 2: 年始 NW = 1000 - 100 - 60 = 840. ターゲット = 84.
  # 前年 tracker 100. 下限 95. 上限 105.
  # ターゲット 84 は下限 95 より小さいので 95 が採用される
  assert res.annual_spends[0, 1] == pytest.approx(95.0)


def test_dynamic_rebalance_cur_ann_spend_with_regular_cashflow():
  """ダイナミックリバランスに渡される cur_ann_spend が正味の定常支出であることを検証。"""
  n_sim = 1
  total_months = 12
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 定常収入 5.0/month (年60)
  monthly_cashflows = {"Income": np.full((n_sim, total_months), 5.0)}
  # 定常支出 10.0/month (年120)
  strategy_cost = 120.0

  received_cur_ann_spend = []

  def dummy_rebalance_fn(net_value, cur_ann_spend, remaining_years, post_tax_net):
    received_cur_ann_spend.append(cur_ann_spend.copy())
    return {"A": np.ones_like(net_value)}

  strategy = Strategy(name="DR_CF_Test",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=strategy_cost,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[CashflowRule("Income", CashflowType.REGULAR)],
                      rebalance_interval=1,
                      dynamic_rebalance_fn=dummy_rebalance_fn)

  simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # 定常支出 10.0 - 収入 5.0 = 5.0 (月額)
  # cur_ann_spend = 5.0 * 12 = 60.0
  for val in received_cur_ann_spend:
    assert val[0] == pytest.approx(60.0)


def test_extra_cashflow_multiplier_receives_prev_gross():
  """
  ExtraCashflowMultiplierFn が前年の正味支出と総支出の両方を正しく受け取れることを検証する。
  """
  n_sim = 1
  total_months = 24
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 定常収入 5.0/month (年60)
  monthly_cashflows = {"Income": np.full((n_sim, total_months), 5.0),
                       "Job": np.zeros((n_sim, total_months))}
  
  # 定常支出 10.0/month (年120)
  annual_cost = 120.0

  received_stats = []

  def mock_multiplier(m, net_worth, prev_net, prev_gross):
    received_stats.append((prev_net.copy(), prev_gross.copy()))
    return np.ones_like(net_worth)

  strategy = Strategy(name="GrossTest",
                      initial_money=1000.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"A": 1.0},
                      annual_cost=annual_cost,
                      inflation_rate=None,
                      selling_priority=["A"],
                      cashflow_rules=[
                          CashflowRule("Income", CashflowType.REGULAR),
                          CashflowRule("Job", CashflowType.EXTRAORDINARY, mock_multiplier)
                      ])

  simulate_strategy(strategy, prices, monthly_cashflows=monthly_cashflows)

  # m=0: 初期値
  assert received_stats[0][0][0] == pytest.approx(120.0)
  assert received_stats[0][1][0] == pytest.approx(120.0)

  # m=12: 1年目の実績
  # prev_net = 120 - 60 = 60
  # prev_gross = 120
  assert received_stats[1][0][0] == pytest.approx(60.0)
  assert received_stats[1][1][0] == pytest.approx(120.0)


def test_simulate_strategy_non_one_starting_price():
  """
  開始時の価格が 1.0 ではない場合でも、資産が正しく評価されることを検証する。
  (Money=200, Price=2.0 の場合、100ユニット保持され、価格変動がなければ 200 のまま)
  """
  n_sim = 1
  total_months = 12
  # 価格がずっと 2.0
  prices = {"Asset": np.full((n_sim, total_months + 1), 2.0)}

  strategy = Strategy(name="NonOnePriceTest",
                      initial_money=200.0,
                      initial_loan=0.0,
                      yearly_loan_interest=0.0,
                      initial_asset_ratio={"Asset": 1.0},
                      annual_cost=0.0,
                      inflation_rate=None,
                      selling_priority=["Asset"])

  res = simulate_strategy(strategy, prices)
  # 価格変動がないので、200.0 が維持されるはず。
  # 以前のバグではここで 200 * 2.0 = 400.0 になってしまっていた。
  assert np.allclose(res.net_values, 200.0)

  # 毎月 10.0 取り崩す。価格 2.0 なので 5ユニット売却。
  # 12ヶ月で 60ユニット売却。残り 40ユニット。
  # 最終価値 40 * 2.0 = 80.0
  # (もし平均取得単価が 1.0 だと、譲渡益が出て税金でさらに減る)
  strategy_withdrawal = Strategy(name="NonOnePriceWithdrawalTest",
                                 initial_money=200.0,
                                 initial_loan=0.0,
                                 yearly_loan_interest=0.0,
                                 initial_asset_ratio={"Asset": 1.0},
                                 annual_cost=120.0,
                                 inflation_rate=None,
                                 selling_priority=["Asset"],
                                 tax_rate=0.2)

  res_w = simulate_strategy(strategy_withdrawal, prices)
  # 取得単価が 2.0 なら譲渡益 0 なので、200 - 120 = 80.0
  assert np.allclose(res_w.net_values, 80.0)


