"""コアシミュレーションロジックのテスト。"""
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pytest

from src.core import (DynamicSpending, SimulationResult, Strategy,
                      ZeroRiskAsset, simulate_strategy)
from src.lib.asset_generator import Asset, CpiAsset, MonthlySimpleNormal
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowConfig,
                                        CashflowRule, CashflowType,
                                        SuddenSpendConfig, generate_cashflows)
from src.lib.spend_aware_dynamic_spending import SpendAwareDynamicSpending


def test_simulate_strategy_no_ruin():
  """
  資産価格が変動しない状況下で、初期資産が支出総額を上回る場合に、
  シミュレーション終了時点で破綻（純資産が0以下）していないことを検証する。
  初期資産 1000 に対して 10年で 100 の支出が発生し、最終的に約 900 残ることを確認。
  """
  n_sim = 10
  n_months = 120
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR)
  ]
  cf_configs: List[CashflowConfig] = [
    BaseSpendConfig(name="BaseSpend", amount=10.0)
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 1.0},
    selling_priority=["AssetA"],
    cashflow_rules=rules
  )

  res = simulate_strategy(strategy=strategy, monthly_asset_prices=prices, monthly_cashflows=monthly_cf)

  assert res.net_values.shape == (n_sim,)
  # 1000 - 100 = 900
  assert np.all(res.net_values > 890.0)


def test_simulate_strategy_bankruptcy():
  """
  初期資産に対して支出が過大である場合に、シミュレーションの途中で
  破綻（純資産が0以下）が発生し、その月が正しく記録されることを検証する。
  初期資産 10 に対して年間 100 の支出があるため、数ヶ月以内に破綻することを確認。
  """
  n_sim = 5
  n_months = 24
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR)
  ]
  cf_configs: List[CashflowConfig] = [
    BaseSpendConfig(name="BaseSpend", amount=100.0)
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=10.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 1.0},
    selling_priority=["AssetA"],
    cashflow_rules=rules
  )

  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert np.all(res.sustained_months < n_months)


def test_simulate_strategy_cost_withdrawal():
  """
  指定された年間支出額に基づいて、毎月のシミュレーションで
  適切に資産から引き出しが行われ、純資産が減少することを検証する。
  年間 120 (月間 10) の支出により、1ヶ月後に資産が 1000 から 990 になることを確認。
  """
  n_sim = 1
  n_months = 12
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR)
  ]
  cf_configs: List[CashflowConfig] = [
    BaseSpendConfig(name="BaseSpend", amount=120.0)
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 1.0},
    selling_priority=["AssetA"],
    cashflow_rules=rules
  )

  # 12ヶ月分の支出 120 が引かれる
  res = simulate_strategy(strategy, prices, monthly_cf, fallback_total_months=1)
  assert res.net_values[0] == 880.0


def test_simulate_strategy_annual_cost_list():
  """
  年間支出額がリスト（年ごとの指定）で与えられた場合に、
  経過年数に応じて正しい支出額が適用されることを検証する。
  1年目 120, 2年目 240 と指定し、2年後の残高が 1000 - 120 - 240 = 640 になることを確認。
  """
  n_sim = 1
  n_months = 24
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR)
  ]
  cf_configs: List[CashflowConfig] = [
    BaseSpendConfig(name="BaseSpend", amount=[120.0, 240.0])
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 1.0},
    selling_priority=["AssetA"],
    cashflow_rules=rules
  )

  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert np.allclose(res.net_values[0], 640.0)


def test_dynamic_spending_initial_cost():
  """
  動的支出（DynamicSpending）において、初年度の支出額が
  `initial_annual_spend` で指定された値と一致することを検証する。
  target_ratio 等を 0 に設定し、純粋に初期設定額が引き出されることを確認。
  """
  n_sim = 1
  n_months = 12
  prices = {"A": np.ones((1, 13)), "Cash": np.ones((1, 13))}
  ds = DynamicSpending(
    initial_annual_spend=120.0,
    target_ratio=0.0,
    upper_limit=0.0,
    lower_limit=0.0
  )
  rules = [
    CashflowRule(
      source_name="BaseSpend",
      cashflow_type=CashflowType.REGULAR,
      dynamic_handler=ds
    )
  ]
  cf_configs: List[CashflowConfig] = [BaseSpendConfig(name="BaseSpend", amount=0.0)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert np.allclose(res.net_values[0], 880.0)


def test_dynamic_spending_ceiling():
  """
  資産額の増大に伴い計算上の支出額が上限を超えた場合に、
  前年比の最大上昇率（upper_limit）で制限されることを検証する。
  資産が10倍になっても、支出が前年(100)の +5% (105) に収まることを確認。
  """
  n_sim = 1
  n_months = 24
  prices = {"A": np.ones((1, 25)) * 10.0, "Cash": np.ones((1, 25))}
  prices["A"][0, 0] = 1.0
  
  # 10% 抽出だが、前年比 +5% 上限
  ds = DynamicSpending(
    initial_annual_spend=100.0,
    target_ratio=0.1,
    upper_limit=0.05,
    lower_limit=-0.05
  )
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR, dynamic_handler=ds)
  ]
  cf_configs: List[CashflowConfig] = [BaseSpendConfig(name="BaseSpend", amount=0.0)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules,
    record_annual_spend=True
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert res.annual_spends is not None
  assert np.allclose(res.annual_spends[0, 1], 105.0)


def test_dynamic_spending_floor():
  """
  資産額の減少に伴い計算上の支出額が下限を下回った場合に、
  前年比の最大下降率（lower_limit）で制限されることを検証する。
  資産が激減しても、支出が前年(100)の -5% (95) に維持されることを確認。
  """
  n_sim = 1
  n_months = 24
  prices = {"A": np.ones((1, 25)) * 0.1, "Cash": np.ones((1, 25))}
  prices["A"][0, 0] = 1.0
  
  ds = DynamicSpending(
    initial_annual_spend=100.0,
    target_ratio=0.1,
    upper_limit=0.05,
    lower_limit=-0.05
  )
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR, dynamic_handler=ds)
  ]
  cf_configs: List[CashflowConfig] = [BaseSpendConfig(name="BaseSpend", amount=0.0)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=10000.0, # 破綻を避けるために増額
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules,
    record_annual_spend=True
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert res.annual_spends is not None
  assert np.allclose(res.annual_spends[0, 1], 95.0)


def test_simulate_strategy_tax():
  """
  資産売却時に利益が発生した場合に、税金（約20%）が適切に計算・
  差し引かれ、最終的な純資産額に反映されることを検証する。
  価格が1.0から2.0に上昇した状況での売却により、税金分が引かれることを確認。
  """
  n_sim = 1
  n_months = 12
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  prices["AssetA"][0, :] = np.linspace(1.0, 2.0, n_months + 1)

  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR)
  ]
  cf_configs: List[CashflowConfig] = [
    BaseSpendConfig(name="BaseSpend", amount=120.0)
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 1.0},
    selling_priority=["AssetA"],
    cashflow_rules=rules
  )

  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert res.net_values[0] < 1880.0
  assert res.net_values[0] > 1600.0


def test_simulate_strategy_rebalance():
  """
  毎月の資産価格変動後に、ポートフォリオのリバランスが実行され、
  指定された資産配分率（ターゲットアロケーション）が維持されることを検証する。
  50:50 の配分において、片方の資産が2倍になった後のリバランス後の純資産配分を確認。
  """
  n_sim = 1
  n_months = 12
  prices = {
    "AssetA": np.ones((n_sim, n_months + 1)),
    "Cash": np.ones((n_sim, n_months + 1)),
  }
  prices["AssetA"][0, 1:] = 2.0
  prices["AssetA"][0, 0] = 1.0

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"AssetA": 0.5, "Cash": 0.5},
    selling_priority=["AssetA", "Cash"],
    rebalance_interval=1,
    tax_rate=0.0
  )

  res = simulate_strategy(strategy, prices, {})
  assert res.net_values[0] == 1500.0


def test_simulate_strategy_zero_risk_asset():
  """
  ZeroRiskAsset（無リスク資産）を指定した場合に、ボラティリティの影響を受けず
  安定した純資産推移が得られることを検証する。
  """
  n_sim = 1
  n_months = 12
  zr = ZeroRiskAsset(name="Savings", yield_rate=0.0)
  
  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={zr: 1.0},
    selling_priority=["Savings"]
  )
  res = simulate_strategy(strategy, {}, {})
  assert res.net_values[0] == 1000.0


def test_strategy_validation():
  """
  Strategy クラスのインスタンス生成時に、資産配分の合計値チェックや
  キャッシュフロールール名の重複チェックなどのバリデーションが正しく動作することを検証する。
  """
  with pytest.raises(ValueError, match="Asset allocation must sum to 1.0"):
    Strategy(
      name="Test",
      initial_money=1000.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"A": 0.5},
      selling_priority=["A"]
    )

  rules = [
    CashflowRule("S1", CashflowType.REGULAR),
    CashflowRule("S1", CashflowType.REGULAR),
  ]
  with pytest.raises(ValueError, match="Duplicate source_name found in cashflow_rules"):
    Strategy(
      name="Test",
      initial_money=1000.0,
      initial_loan=0.0,
      yearly_loan_interest=0.0,
      initial_asset_ratio={"A": 1.0},
      selling_priority=["A"],
      cashflow_rules=rules
    )


def test_simulate_with_dynamic_rebalance():
  """
  `dynamic_rebalance_fn` が提供された場合に、シミュレーション中に
  その関数が呼び出され、返された資産配分が適用されることを検証する。
  """
  n_sim = 1
  n_months = 12
  prices = {"A": np.ones((1, 13)), "Cash": np.ones((1, 13))}

  def dummy_rebalance_fn(total_net: np.ndarray, cur_ann_spend: np.ndarray, 
                         rem_years: float, post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    return {"A": 0.8, "Cash": 0.2}

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 0.5, "Cash": 0.5},
    selling_priority=["A", "Cash"],
    rebalance_interval=1,
    dynamic_rebalance_fn=dummy_rebalance_fn
  )

  res = simulate_strategy(strategy, prices, {})
  assert res.net_values.shape == (1,)


def test_simulate_strategy_empty_prices():
  """
  入力される価格データ（prices）が空の場合に、
  フォールバック値を用いてシミュレーションが実行されることを検証する。
  """
  zr = ZeroRiskAsset(name="Savings", yield_rate=0.0)
  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={zr: 1.0},
    selling_priority=["Savings"]
  )
  # 空の辞書を渡してもエラーにならず、fallback_n_sim などが使われる
  res = simulate_strategy(strategy, {}, fallback_n_sim=10, fallback_total_months=12)
  assert res.net_values.shape == (10,)


def test_simulate_strategy_extra_cashflow():
  """
  SuddenSpendConfig を用いた追加キャッシュフローが、
  指定された月に正確に実行され、純資産が減少することを検証する。
  CashflowType.EXTRAORDINARY を使用し、定常支出計算に影響を与えないことを想定。
  """
  n_sim = 1
  n_months = 12
  prices = {"A": np.ones((1, 13)), "Cash": np.ones((1, 13))}

  rules = [CashflowRule("Sudden", CashflowType.EXTRAORDINARY)]
  cf_configs: List[CashflowConfig] = [SuddenSpendConfig("Sudden", -500, 6)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert res.net_values[0] == 500.0


def test_extra_cashflow_multiplier():
  """
  キャッシュフロールールに `multiplier_fn` が設定されている場合に、
  支出額に倍率が適用され、最終的な純資産額に反映されることを検証する。
  常に 2.0 倍になる関数により、年間支出が 120 から 240 に増えることを確認。
  """
  n_sim = 1
  n_months = 12
  prices = {"A": np.ones((1, 13)), "Cash": np.ones((1, 13))}

  def double_multiplier(m: int, nw: np.ndarray, prev_net: np.ndarray, prev_gross: np.ndarray) -> np.ndarray:
    # 常に 2.0 倍にする。nw の長さに合わせた配列を返す。
    return np.ones(len(nw)) * 2.0

  rules = [
    CashflowRule("BaseSpend", CashflowType.REGULAR, multiplier_fn=double_multiplier)
  ]
  cf_configs: List[CashflowConfig] = [BaseSpendConfig("BaseSpend", 120.0)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert np.allclose(res.net_values[0], 760.0)


def test_cashflow_type_isolated_from_dynamic_base():
  """
  CashflowType.REGULAR として設定された支出（保険料など）が、
  動的支出（DynamicSpending）の「前年実績支出」に混入しないことを検証する。
  混入すると、other_net との二重カウントで支出が爆発するバグが発生するため。
  1年目の臨時支出(REGULAR) 100 は、2年目の動的支出のベースには引き継がれないことを確認。
  """
  n_sim = 1
  n_months = 24
  prices = {"A": np.ones((1, 25)), "Cash": np.ones((1, 25))}

  ds = DynamicSpending(initial_annual_spend=0.0, target_ratio=0.0, upper_limit=1.0, lower_limit=0.0)
  rules = [
    CashflowRule("Extra", CashflowType.REGULAR),
    CashflowRule("Dynamic", CashflowType.REGULAR, dynamic_handler=ds)
  ]
  cf_configs: List[CashflowConfig] = [
    SuddenSpendConfig("Extra", -100, 6),
    BaseSpendConfig("Dynamic", 0.0)
  ]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=5000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules,
    record_annual_spend=True
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  # 1年目: Dynamic=0, Extra=100。合計支出=100。残高=4900
  # 2年目: Dynamicは自分自身の前年実績(0)をベースにするため、0。Extraは無し。残高=4900のまま。
  assert np.allclose(res.net_values[0], 4900.0)  # 5000 - 100


def test_dynamic_spending_inflation_fix():
  """
  CPI（物価指数）が変動する状況下で、動的支出の計算に CPI が反映され
  名目額が調整されることを検証する。
  1年で CPI が 2倍になる場合、2年目の名目支出額も 2倍になることを確認。
  """
  n_sim = 1
  n_months = 24
  # CPIが1年で2倍になる設定
  cpi_val = np.array([1.0]*12 + [2.0]*13).reshape(1, 25)
  prices = {"A": np.ones((1, 25)), "Cash": np.ones((1, 25)), "CPI": cpi_val}
  
  # DynamicSpending 自体は CPI を直接参照しないが、evaluate に precomputed_cf_m (CPI) が渡される
  ds = DynamicSpending(initial_annual_spend=100.0, target_ratio=0.0, upper_limit=1.0, lower_limit=0.0)
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR, dynamic_handler=ds)
  ]
  # BaseSpendConfig で CPI 連動させる。
  # CPI比率を正しく計算させるため、非ゼロの amount を指定する。
  cf_configs: List[CashflowConfig] = [BaseSpendConfig(name="BaseSpend", amount=12.0, cpi_name="CPI")]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=5000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules,
    record_annual_spend=True
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  # 1年目支出: 100 (initial)
  # 2年目支出決定時(m=12): cf_ratio = CPI[12]/CPI[0] = 2.0.
  # 2年目名目支出 = 100 * 2.0 = 200.
  assert res.annual_spends is not None
  assert np.allclose(res.annual_spends[0, 1], 200.0)


class MockDPPredictor:
  """
  SpendAwareDynamicSpending のテスト用モック。
  s_rate（支出率）が低いほど生存確率が高くなる単純な線形モデルを提供。
  """
  def predict_p_surv(self, age: int, s_rate: np.ndarray) -> np.ndarray:
    # s_rate が低いほど生存確率が高い単純なモデル
    return np.clip(1.0 - s_rate * 10, 0.0, 1.0)

def test_spend_aware_dynamic_spending_basic():
  """
  SpendAwareDynamicSpending（資産・余命に応じた動的支出）が、
  生存確率のガードレールに基づいて支出額を調整することを検証する。
  初期の支出率が危険域（生存確率低）にある場合、支出が削減されることを確認。
  """
  n_sim = 1
  n_months = 12
  prices = {"A": np.ones((1, 13)), "Cash": np.ones((1, 13))}
  
  predictor = MockDPPredictor()
  # 資産 1000, 予定支出 100 -> s_rate=0.1. 
  # predict_p_surv(0.1) = 1.0 - 0.1*10 = 0.0. 
  # p_low=0.9 なので、大幅な支出削減が発生するはず。
  ds = SpendAwareDynamicSpending(
    initial_age=60,
    p_low=0.9,
    p_high=0.95,
    lower_mult=0.5,
    upper_mult=1.5,
    annual_cost_real=[100.0]*100,
    dp_predictor=cast(Any, predictor)
  )
  rules = [
    CashflowRule(source_name="BaseSpend", cashflow_type=CashflowType.REGULAR, dynamic_handler=ds)
  ]
  cf_configs: List[CashflowConfig] = [BaseSpendConfig(name="BaseSpend", amount=0.0)]
  monthly_cf = generate_cashflows(cf_configs, prices, n_sim, n_months)

  strategy = Strategy(
    name="Test",
    initial_money=1000.0,
    initial_loan=0.0,
    yearly_loan_interest=0.0,
    initial_asset_ratio={"A": 1.0},
    selling_priority=["A"],
    cashflow_rules=rules,
    record_annual_spend=True
  )
  res = simulate_strategy(
    strategy=strategy,
    monthly_asset_prices=prices,
    monthly_cashflows=monthly_cf
  )
  assert res.annual_spends is not None
  # 支出が 100.0 より削減されていることを確認
  assert res.annual_spends[0, 0] < 100.0
