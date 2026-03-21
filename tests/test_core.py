import numpy as np
import pytest

from src.core import (DynamicSpending, SimulationResult, Strategy,
                      ZeroRiskAsset, simulate_strategy)


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
  spending = DynamicSpending(target_ratio=0.04,
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
  (初期資産 * target_ratio) として計算されることを検証する。
  """
  n_sim = 1
  total_months = 12
  prices = {"A": np.ones((n_sim, total_months + 1))}

  # 初期資産 100.0, ターゲット 5% -> 初年度支出は 5.0
  spending = DynamicSpending(target_ratio=0.05,
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
  spending = DynamicSpending(target_ratio=0.05,
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
  spending = DynamicSpending(target_ratio=0.05,
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

  spending = DynamicSpending(target_ratio=0.05,
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

  def dummy_rebalance_fn(net_value, annual_spend, remaining_years):
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
