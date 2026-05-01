import hashlib
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple, Union, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core import Strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType, CurveSpend,
                                      DynamicV1Adjustment, DynamicV1Rebalance,
                                      FxType, Gender, Lifeplan, PensionStatus,
                                      PredefinedStock, PredefinedZeroRisk,
                                      Setup, SpendAwareAdjustment,
                                      SpendAwareDPRebalance, StrategySpec,
                                      WorldConfig, create_experiment_setup)


@pytest.fixture
def baseline_setup():
  # 世界設定の最小構成
  world = WorldConfig(n_sim=10, n_years=10, start_age=50)
  # ライフプランの最小構成
  lifeplan = Lifeplan(base_spend=ConstantSpend(annual_amount=480),
                      retirement_start_age=50)
  # 戦略設定の最小構成
  strategy = StrategySpec(initial_money=1000,
                          initial_asset_ratio=((PredefinedStock.ORUKAN_155,
                                                1.0),),
                          selling_priority=(PredefinedStock.ORUKAN_155,))
  return Setup(name="baseline",
               world=world,
               lifeplan=lifeplan,
               strategy=strategy)


def test_setup_add_experiment_overrides(baseline_setup):
  """add_experiment が引数を正しくマージし、ベースラインを上書きすることを確認する。"""
  new_lifeplan = replace(baseline_setup.lifeplan, household_size=2)
  baseline_setup.add_experiment(name="exp1", overwrite_lifeplan=new_lifeplan)
  assert len(baseline_setup.experiments) == 1
  assert baseline_setup.experiments[0].name == "exp1"
  assert baseline_setup.experiments[0].lifeplan.household_size == 2
  assert baseline_setup.experiments[0].strategy == baseline_setup.strategy


def test_create_experiment_setup_deduplicates_worlds(baseline_setup):
  """異なる WorldConfig に対して、アセット生成が適切に実行されることを確認する。"""
  baseline_setup.add_experiment(name="exp_diff_world",
                                overwrite_world=replace(
                                    baseline_setup.world,
                                    cpi_type=CpiType.FIXED_0))
  compiled = create_experiment_setup(baseline_setup)
  assert len(compiled) == 2
  # 異なる World なので、monthly_prices の ID が異なるはず（ただし、モック化されていない場合）
  # 実際には generate_monthly_asset_prices が呼ばれる。
  assert compiled[0].monthly_prices is not compiled[1].monthly_prices


def test_create_experiment_setup_deduplicates_lifeplans_within_world(
    baseline_setup):
  """同じ World 内で、同一の Lifeplan が重複して生成されないことを確認する。"""
  baseline_setup.add_experiment(name="exp1")
  baseline_setup.add_experiment(name="exp2")
  compiled = create_experiment_setup(baseline_setup)
  assert len(compiled) == 3
  # 全て同じ World, Lifeplan なので、キャッシュフロー配列を共有しているはず
  # CompiledExperiment の monthly_cashflows は同じ Dict オブジェクト
  assert compiled[0].monthly_cashflows is compiled[1].monthly_cashflows
  assert compiled[1].monthly_cashflows is compiled[2].monthly_cashflows


def test_compile_lifeplan_constant_spend(baseline_setup):
  """ConstantSpend が正しい CashflowConfig (固定額) に変換されることを確認する。"""
  # CPI の影響を除外するために固定 CPI (0%) を使用
  baseline_setup.world = replace(baseline_setup.world, cpi_type=CpiType.FIXED_0)
  compiled = create_experiment_setup(baseline_setup)
  cf_rules = compiled[0].strategy.cashflow_rules
  # BaseSpend が存在することを確認
  base_spend_rule = next(r for r in cf_rules if "BaseSpend" in r.source_name)
  assert base_spend_rule is not None
  # 配列の中身が負の固定値であることを確認
  arr = compiled[0].monthly_cashflows[base_spend_rule.source_name]
  assert np.allclose(arr, -480 / 12)


def test_compile_lifeplan_curve_spend(baseline_setup):
  """CurveSpend が統計データに基づく CashflowConfig に変換されることを確認する。"""
  # CPI の影響を除外するために固定 CPI (0%) を使用
  baseline_setup.world = replace(baseline_setup.world, cpi_type=CpiType.FIXED_0)
  baseline_setup.lifeplan = replace(
      baseline_setup.lifeplan,
      base_spend=CurveSpend(first_year_annual_amount=480))
  compiled = create_experiment_setup(baseline_setup)
  base_spend_rule = next(r for r in compiled[0].strategy.cashflow_rules
                         if "BaseSpend" in r.source_name)
  arr = compiled[0].monthly_cashflows[base_spend_rule.source_name]
  # 最初の月は -480/12
  assert np.isclose(arr[0, 0], -480 / 12)
  # 経年で変化しているはず
  assert not np.all(arr == arr[0, 0])


def test_compile_lifeplan_pension_rigorous(baseline_setup):
  """年金の金額、期間、受給開始時期、およびステータス別の差異を厳密に検証する。"""
  # 50歳開始、55歳リタイア、65歳受給開始
  baseline_setup.world = replace(baseline_setup.world,
                                 start_age=50,
                                 n_years=40,
                                 cpi_type=CpiType.FIXED_0)
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.FULL,
                                    pension_start_age=65)

  compiled = create_experiment_setup(baseline_setup)
  cashflows = compiled[0].monthly_cashflows
  cf_rules = compiled[0].strategy.cashflow_rules

  # 1. 保険料 (PensionPremium)
  # 50歳から60歳まで（10年間 = 120ヶ月）
  premium_rule = next(r for r in cf_rules if "PensionPremium" in r.source_name)
  premium_arr = cashflows[premium_rule.source_name]
  expected_premium_monthly = -21.5 / 12.0
  assert np.allclose(premium_arr[0, :120], expected_premium_monthly)
  assert np.allclose(premium_arr[0, 120:], 0)

  # 2. 厚生年金 (PensionKousei)
  # 65歳から（15年後 = 180ヶ月目から）
  # 金額: 2.736 * (55 - 22) = 2.736 * 33 = 90.288 / 12 = 7.524
  kousei_rule = next(r for r in cf_rules if "PensionKousei" in r.source_name)
  kousei_arr = cashflows[kousei_rule.source_name]
  expected_kousei_monthly = (2.736 * (55 - 22)) / 12.0
  assert np.allclose(kousei_arr[0, :180], 0)
  assert np.allclose(kousei_arr[0, 180:], expected_kousei_monthly)

  # 3. 基礎年金 (PensionKiso)
  # 65歳から
  # 金額: 81.6 / 12 = 6.8
  kiso_rule = next(r for r in cf_rules if "PensionKiso" in r.source_name)
  kiso_arr = cashflows[kiso_rule.source_name]
  expected_kiso_monthly = 81.6 / 12.0
  assert np.allclose(kiso_arr[0, :180], 0)
  assert np.allclose(kiso_arr[0, 180:], expected_kiso_monthly)


def test_pension_status_variations(baseline_setup):
  """EXEMPT および UNPAID の場合の基礎年金額の違いを検証する。"""
  # 50歳開始、55歳リタイア、65歳受給
  baseline_setup.world = replace(baseline_setup.world,
                                 start_age=50,
                                 n_years=40,
                                 cpi_type=CpiType.FIXED_0)

  # EXEMPT: (81.6 * (55-22)/40 + 81.6 * (60-55)/40 * 0.5) = (81.6 * 33/40 + 81.6 * 5/40 * 0.5)
  # = 67.32 + 5.1 = 72.42 / 12 = 6.035
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.EXEMPT,
                                    pension_start_age=65)
  compiled_exempt = create_experiment_setup(baseline_setup)
  kiso_rule_e = next(r for r in compiled_exempt[0].strategy.cashflow_rules
                     if "PensionKiso" in r.source_name)
  assert np.isclose(
      compiled_exempt[0].monthly_cashflows[kiso_rule_e.source_name][0, 180],
      72.42 / 12.0)

  # UNPAID: (81.6 * (55-22)/40) = 81.6 * 33/40 = 67.32 / 12 = 5.61
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.UNPAID,
                                    pension_start_age=65)
  compiled_unpaid = create_experiment_setup(baseline_setup)
  kiso_rule_u = next(r for r in compiled_unpaid[0].strategy.cashflow_rules
                     if "PensionKiso" in r.source_name)
  assert np.isclose(
      compiled_unpaid[0].monthly_cashflows[kiso_rule_u.source_name][0, 180],
      67.32 / 12.0)


def test_pension_age_reduction_and_increase(baseline_setup):
  """繰り上げ(60歳)および繰り下げ(70歳)の影響を検証する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 start_age=50,
                                 n_years=40,
                                 cpi_type=CpiType.FIXED_0)

  # 60歳受給 (65歳から60ヶ月繰り上げ、0.4% * 60 = 24%減) -> 76%
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.FULL,
                                    pension_start_age=60)
  compiled_60 = create_experiment_setup(baseline_setup)
  kiso_rule_60 = next(r for r in compiled_60[0].strategy.cashflow_rules
                      if "PensionKiso" in r.source_name)
  # 81.6 * 0.76 = 62.016 / 12 = 5.168
  # 10年後（120ヶ月目）から開始
  arr_60 = compiled_60[0].monthly_cashflows[kiso_rule_60.source_name]
  assert np.allclose(arr_60[0, :120], 0)
  assert np.isclose(arr_60[0, 120], 5.168)

  # 70歳受給 (65歳から60ヶ月繰り下げ、0.7% * 60 = 42%増) -> 142%
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.FULL,
                                    pension_start_age=70)
  compiled_70 = create_experiment_setup(baseline_setup)
  kiso_rule_70 = next(r for r in compiled_70[0].strategy.cashflow_rules
                      if "PensionKiso" in r.source_name)
  # 81.6 * 1.42 = 115.872 / 12 = 9.656
  # 20年後（240ヶ月目）から開始
  arr_70 = compiled_70[0].monthly_cashflows[kiso_rule_70.source_name]
  assert np.allclose(arr_70[0, :240], 0)
  assert np.isclose(arr_70[0, 240], 9.656)


def test_compile_lifeplan_side_fire(baseline_setup):
  """side_fire_income_monthly が追加収入のキャッシュフローとして生成されることを確認する。"""
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    side_fire_income_monthly=10,
                                    side_fire_duration_months=12)
  # CPI の影響を除外するために固定 CPI (0%) を使用
  baseline_setup.world = replace(baseline_setup.world, cpi_type=CpiType.FIXED_0)
  compiled = create_experiment_setup(baseline_setup)
  side_fire_rule = next(r for r in compiled[0].strategy.cashflow_rules
                        if "SideFire" in r.source_name)
  arr = compiled[0].monthly_cashflows[side_fire_rule.source_name]
  # 最初の12ヶ月は 10
  assert np.allclose(arr[:, :12], 10)
  # 13ヶ月目以降は 0
  assert np.allclose(arr[:, 12:], 0)


def test_compile_assets_fx_params(baseline_setup):
  """各 FxType に対して正しいパラメータ（ボラティリティ等）が反映されていることを確認する。"""
  # 十分なサンプル数で統計的な差異を確認
  baseline_setup.world = replace(baseline_setup.world, n_sim=1000, seed=42)

  # USDJPY_LOW_RISK (sigma=0.05)
  baseline_setup.world = replace(baseline_setup.world,
                                 fx_type=FxType.USDJPY_LOW_RISK)
  compiled_low = create_experiment_setup(baseline_setup)
  fx_key_low = next(
      k for k in compiled_low[0].monthly_prices.keys() if "USDJPY" in k)
  prices_low = compiled_low[0].monthly_prices[fx_key_low]
  # 対数収益率の標準偏差を確認 (年率換算)
  log_returns_low = np.diff(np.log(prices_low), axis=1)
  std_low = np.std(log_returns_low) * np.sqrt(12)

  # USDJPY (sigma=0.1053)
  baseline_setup.world = replace(baseline_setup.world, fx_type=FxType.USDJPY)
  compiled_high = create_experiment_setup(baseline_setup)
  fx_key_high = next(
      k for k in compiled_high[0].monthly_prices.keys() if "USDJPY" in k)
  prices_high = compiled_high[0].monthly_prices[fx_key_high]
  log_returns_high = np.diff(np.log(prices_high), axis=1)
  std_high = np.std(log_returns_high) * np.sqrt(12)

  # 高リスク設定の方が標準偏差が大きいはず
  assert std_high > std_low
  assert np.isclose(std_low, 0.05, atol=0.01)
  assert np.isclose(std_high, 0.1053, atol=0.01)


def test_compile_assets_fx_modified(baseline_setup):
  """USDJPY_MODIFIED のパラメータが正しく反映されていることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 n_sim=1000,
                                 seed=42,
                                 fx_type=FxType.USDJPY_MODIFIED)
  compiled = create_experiment_setup(baseline_setup)
  fx_key = next(k for k in compiled[0].monthly_prices.keys() if "USDJPY" in k)
  prices = compiled[0].monthly_prices[fx_key]
  log_returns = np.diff(np.log(prices), axis=1)
  std_annual = np.std(log_returns) * np.sqrt(12)
  # mu=0.01, sigma=0.10
  assert np.isclose(std_annual, 0.10, atol=0.01)


def test_compile_assets_cpi_fixed_1_77(baseline_setup):
  """CpiType.FIXED_1_77 が正しく処理されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 cpi_type=CpiType.FIXED_1_77)
  compiled = create_experiment_setup(baseline_setup)
  assert "Japan_CPI" in compiled[0].monthly_prices


def test_compile_assets_sp500_155(baseline_setup):
  """PredefinedStock.SP500_155 が正しく処理されることを確認する。"""
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      initial_asset_ratio=((PredefinedStock.SP500_155, 1.0),),
      selling_priority=(PredefinedStock.SP500_155,))
  compiled = create_experiment_setup(baseline_setup)
  assert "SP500_155" in compiled[0].monthly_prices


def test_strategy_cash_ratio(baseline_setup):
  """PredefinedZeroRisk.CASH が比率に含まれる場合、Strategy に ZeroRiskAsset として反映されることを確認する。"""
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      initial_asset_ratio=((PredefinedZeroRisk.CASH, 0.5),
                           (PredefinedStock.ORUKAN_155, 0.5)),
      selling_priority=(PredefinedZeroRisk.CASH, PredefinedStock.ORUKAN_155))
  compiled = create_experiment_setup(baseline_setup)
  ratio = compiled[0].strategy.initial_asset_ratio
  # ZeroRiskAsset オブジェクトがキーになっていることを確認
  cash_key = next(
      k for k in ratio.keys() if hasattr(k, "name") and k.name == "CASH")
  assert ratio[cash_key] == 0.5


def test_dynamic_v1_rebalance_func_call(baseline_setup):
  """DynamicV1Rebalance によって生成された関数が呼び出し可能であることを確認する。"""
  baseline_setup.strategy = replace(baseline_setup.strategy,
                                    rebalance=DynamicV1Rebalance(
                                        risky_asset=PredefinedStock.ORUKAN_155,
                                        zero_risk_asset=PredefinedZeroRisk.CASH,
                                        interval_months=12))
  compiled = create_experiment_setup(baseline_setup)
  strategy = compiled[0].strategy
  assert strategy.dynamic_rebalance_fn is not None

  net_worth = np.array([1000.0] * 10)
  ann_spend = np.array([40.0] * 10)
  # 関数を呼び出して結果を検証
  res = strategy.dynamic_rebalance_fn(net_worth, ann_spend, 20.0, net_worth)
  assert "ORUKAN_155" in res
  assert "CASH" in res


def test_strategy_zero_risk_4pct(baseline_setup):
  """ZERO_RISK_4PCT が正しく Strategy に反映されることを確認する。"""
  from src.core import ZeroRiskAsset
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      initial_asset_ratio=((PredefinedZeroRisk.ZERO_RISK_4PCT, 1.0),),
      selling_priority=(PredefinedZeroRisk.ZERO_RISK_4PCT,))
  compiled = create_experiment_setup(baseline_setup)
  ratio = compiled[0].strategy.initial_asset_ratio
  key = next(k for k in ratio.keys()
             if isinstance(k, ZeroRiskAsset) and k.name == "ZERO_RISK_4PCT")
  assert isinstance(key, ZeroRiskAsset)
  assert key.yield_rate == 0.04


def test_rebalance_v1_rigorous(baseline_setup):
  """DynamicV1Rebalance の計算結果（s_rate, ratio）が正しいことを検証する。"""
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      rebalance=DynamicV1Rebalance(
          risky_asset=PredefinedStock.ORUKAN_155,
          zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
          interval_months=12))

  with patch(
      "src.lib.scenario_builder.calculate_optimal_strategy") as mock_calc:
    mock_calc.return_value = 0.6
    compiled = create_experiment_setup(baseline_setup)
    dr_fn = compiled[0].strategy.dynamic_rebalance_fn
    assert dr_fn is not None

    # テスト入力
    net_worth = np.array([1000.0] * 10)
    ann_spend = np.array([40.0] * 10)
    rem_years = 20.0
    post_tax_net = np.array([800.0] * 10)

    res = dr_fn(net_worth, ann_spend, rem_years, post_tax_net)

    # calculate_optimal_strategy への引数を検証
    # s_rate = 40.0 / 800.0 = 0.05
    args, kwargs = mock_calc.call_args
    assert np.allclose(args[0], 0.05)
    assert args[1] == 20.0
    assert kwargs["base_yield"] == 0.04
    assert kwargs["inflation_rate"] == 0.0177

    assert res["ORUKAN_155"] == 0.6
    assert res["ZERO_RISK_4PCT"] == 0.4


def test_create_experiment_setup_resolves_spend_aware_adjustment_call(
    baseline_setup):
  """SpendAwareDynamicSpending の evaluate を呼び出すパスを通す。"""
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      spend_adjustment=SpendAwareAdjustment(model_name="dummy.json"))
  with patch(
      "src.lib.scenario_builder.DPOptimalStrategyPredictor") as mock_predictor:
    predictor_instance = mock_predictor.return_value
    predictor_instance.predict_p_surv.return_value = np.array([0.95] * 10)
    compiled = create_experiment_setup(baseline_setup)
    handler = compiled[0].strategy.cashflow_rules[0].dynamic_handler
    assert handler is not None
    # evaluate を呼んでみる
    active_paths = np.array([True] * 10)
    net_worth = np.ones(10) * 1000
    zeros = np.zeros(10)
    res = handler.evaluate(m=12,
                           active_paths=active_paths,
                           current_net_worth=net_worth,
                           tax_cost_m=zeros,
                           prev_actual_amount=np.ones(10) * 40,
                           other_net_m=zeros,
                           precomputed_cf_m=np.ones(10) * 40,
                           precomputed_cf_prev_m=np.ones(10) * 40)
    assert isinstance(res, np.ndarray)


def test_create_experiment_setup_covers_dp_rebalance_call(baseline_setup):
  """DPリバランス関数の呼び出しパスを通す。"""
  baseline_setup.strategy = replace(baseline_setup.strategy,
                                    rebalance=SpendAwareDPRebalance(
                                        risky_asset=PredefinedStock.ORUKAN_155,
                                        zero_risk_asset=PredefinedZeroRisk.CASH,
                                        model_name="dummy.json"))
  with patch(
      "src.lib.scenario_builder.DPOptimalStrategyPredictor") as mock_predictor:
    mock_predictor.return_value = MagicMock()
    with patch(
        "src.lib.scenario_builder.calculate_optimal_strategy_dp") as mock_calc:
      mock_calc.return_value = 0.6
      compiled = create_experiment_setup(baseline_setup)
      strategy = compiled[0].strategy
      net_worth = np.array([1000.0] * 10)
      ann_spend = np.array([40.0] * 10)
      assert strategy.dynamic_rebalance_fn is not None
      res = strategy.dynamic_rebalance_fn(net_worth, ann_spend, 20.0, net_worth)
      assert res["ORUKAN_155"] == 0.6
      assert res["CASH"] == 0.4


def test_pension_premium_logic(baseline_setup):
  """保険料支払いの条件 (60歳未満 かつ FULLステータス) を検証する。"""
  # ケース1: 65歳開始 (60歳過ぎている) -> 保険料 0
  baseline_setup.world = replace(baseline_setup.world, start_age=65)
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    pension_status=PensionStatus.FULL)
  compiled_65 = create_experiment_setup(baseline_setup)
  # PensionPremium ルールが存在しないことを確認
  assert not any("PensionPremium" in r.source_name
                 for r in compiled_65[0].strategy.cashflow_rules)

  # ケース2: 50歳開始、EXEMPTステータス -> 保険料 0
  baseline_setup.world = replace(baseline_setup.world, start_age=50)
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    pension_status=PensionStatus.EXEMPT)
  compiled_exempt = create_experiment_setup(baseline_setup)
  assert not any("PensionPremium" in r.source_name
                 for r in compiled_exempt[0].strategy.cashflow_rules)


def test_mortality_gender_rigorous(baseline_setup):
  """性別に応じて正しい生命表が選択されることを検証する。"""
  from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES

  # 男性
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    mortality_gender=Gender.MALE)
  compiled_male = create_experiment_setup(baseline_setup)
  # CompiledExperiment から直接 Config を取り出すことはできないので、
  # _compile_lifeplan を直接テストするか、Mock を使うか、
  # あるいは内部のキャッシュフロー生成ロジックを信じるなら
  # 別の方法が必要。ここでは _compile_lifeplan の出力を検証する。
  from src.lib.cashflow_generator import MortalityConfig
  from src.lib.scenario_builder import _compile_lifeplan
  compiled_male_lp = _compile_lifeplan(baseline_setup.lifeplan,
                                       baseline_setup.world)
  male_cf = compiled_male_lp.configs
  mort_male = next(c for c in male_cf if c.name == "Mortality")
  assert isinstance(mort_male, MortalityConfig)
  assert np.array_equal(mort_male.mortality_rates, MALE_MORTALITY_RATES)

  # 女性
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    mortality_gender=Gender.FEMALE)
  compiled_female_lp = _compile_lifeplan(baseline_setup.lifeplan,
                                         baseline_setup.world)
  female_cf = compiled_female_lp.configs
  mort_female = next(c for c in female_cf if c.name == "Mortality")
  assert isinstance(mort_female, MortalityConfig)
  assert np.array_equal(mort_female.mortality_rates, FEMALE_MORTALITY_RATES)


def test_cashflow_types_rigorous(baseline_setup):
  """各キャッシュフローのタイプ (REGULAR / EXTRAORDINARY) が正しいことを検証する。"""
  from src.lib.cashflow_generator import CashflowType

  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    side_fire_income_monthly=10,
                                    side_fire_duration_months=12,
                                    pension_status=PensionStatus.FULL,
                                    pension_start_age=65,
                                    mortality_gender=Gender.MALE)
  compiled = create_experiment_setup(baseline_setup)
  rules = compiled[0].strategy.cashflow_rules

  # BaseSpend: REGULAR
  base_spend = next(r for r in rules if "BaseSpend" in r.source_name)
  assert base_spend.cashflow_type == CashflowType.REGULAR

  # SideFire: REGULAR
  side_fire = next(r for r in rules if "SideFire" in r.source_name)
  assert side_fire.cashflow_type == CashflowType.REGULAR

  # Pension...: REGULAR
  pension_kiso = next(r for r in rules if "PensionKiso" in r.source_name)
  assert pension_kiso.cashflow_type == CashflowType.REGULAR

  # Mortality: EXTRAORDINARY
  mortality = next(r for r in rules if "Mortality" in r.source_name)
  assert mortality.cashflow_type == CashflowType.EXTRAORDINARY


def test_create_experiment_setup_infers_assets_with_fx_type(baseline_setup):
  """World の FxType に応じて、適切な AssetConfig が生成されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world, fx_type=FxType.USDJPY)
  compiled = create_experiment_setup(baseline_setup)
  # ORUKAN_155 の価格に為替が影響しているはず
  # 1. monthly_prices に為替アセットが含まれていることを確認
  assert any("USDJPY" in k for k in compiled[0].monthly_prices.keys())
  assert "ORUKAN_155" in compiled[0].monthly_prices


def test_compile_assets_no_fx(baseline_setup):
  """FxType.NONE の場合、為替アセットが生成されないことを確認する。"""
  baseline_setup.world = replace(baseline_setup.world, fx_type=FxType.NONE)
  compiled = create_experiment_setup(baseline_setup)
  # monthly_prices に USDJPY が含まれていないことを確認
  assert not any("USDJPY" in k for k in compiled[0].monthly_prices.keys())


def test_create_experiment_setup_resolves_dynamic_v1_rebalance(baseline_setup):
  """DynamicV1Rebalance が Strategy に反映されることを確認する。"""
  baseline_setup.strategy = replace(baseline_setup.strategy,
                                    rebalance=DynamicV1Rebalance(
                                        risky_asset=PredefinedStock.ORUKAN_155,
                                        zero_risk_asset=PredefinedZeroRisk.CASH,
                                        interval_months=12))
  compiled = create_experiment_setup(baseline_setup)
  assert compiled[0].strategy.rebalance_interval == 12
  assert compiled[0].strategy.dynamic_rebalance_fn is not None


def test_create_experiment_setup_resolves_spend_aware_dp_rebalance(
    baseline_setup):
  """SpendAwareDPRebalance が DP リバランス関数を注入することを確認する。"""
  baseline_setup.strategy = replace(baseline_setup.strategy,
                                    rebalance=SpendAwareDPRebalance(
                                        risky_asset=PredefinedStock.ORUKAN_155,
                                        zero_risk_asset=PredefinedZeroRisk.CASH,
                                        model_name="dummy.json"))
  with patch(
      "src.lib.scenario_builder.DPOptimalStrategyPredictor") as mock_predictor:
    mock_predictor.return_value = MagicMock()
    compiled = create_experiment_setup(baseline_setup)
    assert len(compiled) == 1
    assert compiled[0].strategy.dynamic_rebalance_fn is not None


def test_create_experiment_setup_resolves_dynamic_v1_adjustment_default_spend(
    baseline_setup):
  """DynamicV1Adjustment が指定されない場合、ライフプランの初期支出額を基準にすることを確認する。"""
  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      spend_adjustment=DynamicV1Adjustment(target_ratio=0.04))
  compiled = create_experiment_setup(baseline_setup)
  handler = compiled[0].strategy.cashflow_rules[0].dynamic_handler
  from src.core import DynamicSpending
  assert isinstance(handler, DynamicSpending)
  # ConstantSpend(480) なので、handler の初期支出も 480 になるはず
  assert handler.initial_annual_spend == 480.0


def test_create_experiment_setup_resolves_dynamic_v1_adjustment_with_initial_spend(
    baseline_setup):
  """DynamicV1Adjustment が指定された initial_annual_spend を注入することを確認する。"""
  baseline_setup.strategy = replace(baseline_setup.strategy,
                                    spend_adjustment=DynamicV1Adjustment(
                                        target_ratio=0.04,
                                        initial_annual_spend=123.45))
  compiled = create_experiment_setup(baseline_setup)
  handler = compiled[0].strategy.cashflow_rules[0].dynamic_handler
  from src.core import DynamicSpending
  assert isinstance(handler, DynamicSpending)
  assert handler.initial_annual_spend == 123.45


def test_create_experiment_setup_with_record_annual_spend(baseline_setup):
  """create_experiment_setup の record_annual_spend フラグが Strategy に伝播することを確認する。"""
  compiled = create_experiment_setup(baseline_setup, record_annual_spend=True)
  assert compiled[0].strategy.record_annual_spend is True

  compiled_false = create_experiment_setup(baseline_setup,
                                           record_annual_spend=False)
  assert compiled_false[0].strategy.record_annual_spend is False


def test_create_experiment_setup_resolves_spend_aware_adjustment(
    baseline_setup):
  """SpendAwareAdjustment が DP スペンディングハンドラを注入し、統計データが渡されることを確認する。"""
  from src.lib.spend_aware_dynamic_spending import SpendAwareDynamicSpending

  # 統計データを使う支出設定
  baseline_setup.lifeplan = replace(
      baseline_setup.lifeplan,
      base_spend=CurveSpend(first_year_annual_amount=480))

  baseline_setup.strategy = replace(
      baseline_setup.strategy,
      spend_adjustment=SpendAwareAdjustment(model_name="dummy.json"))

  with patch(
      "src.lib.scenario_builder.DPOptimalStrategyPredictor") as mock_predictor:
    mock_predictor.return_value = MagicMock()
    compiled = create_experiment_setup(baseline_setup)

    handler = compiled[0].strategy.cashflow_rules[0].dynamic_handler
    assert isinstance(handler, SpendAwareDynamicSpending)
    # 統計データのカーブが渡されていることを確認 (Constant 1.0 ではないはず)
    assert len(handler.annual_cost_real) == baseline_setup.world.n_years
    assert not np.allclose(handler.annual_cost_real, 480.0)
    # 最初の要素は指定した 480.0 にスケーリングされているはず
    assert np.isclose(handler.annual_cost_real[0], 480.0)


def test_compile_lifeplan_curve_spend_no_normalization(baseline_setup):
  """first_year_annual_amount=None の場合、統計データの生の値が使用されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world, cpi_type=CpiType.FIXED_0)
  baseline_setup.lifeplan = replace(
      baseline_setup.lifeplan,
      base_spend=CurveSpend(first_year_annual_amount=None))
  compiled = create_experiment_setup(baseline_setup)
  base_spend_rule = next(r for r in compiled[0].strategy.cashflow_rules
                         if "BaseSpend" in r.source_name)
  arr = compiled[0].monthly_cashflows[base_spend_rule.source_name]
  # 生の金額（絶対値）が 1.0 を超えていることを確認（統計データなら月額 10万円以上はあるはず）
  assert np.all(arr < 0)
  assert np.abs(arr[0, 0]) > 1.0


def test_unhandled_enum_values(baseline_setup):
  """未知の Enum 値が渡された場合に ValueError が発生することを検証する。"""
  from src.lib.scenario_builder import (_build_strategy, _compile_assets,
                                        _compile_lifeplan)

  # 1. CpiType
  with pytest.raises(ValueError, match="未知の CPI タイプです"):
    invalid_cpi = cast(CpiType, -1)
    _compile_assets(set(), replace(baseline_setup.world, cpi_type=invalid_cpi))

  # 2. FxType
  with pytest.raises(ValueError, match="未知の為替タイプです"):
    invalid_fx = cast(FxType, -1)
    _compile_assets(set(), replace(baseline_setup.world, fx_type=invalid_fx))

  # 3. PensionStatus
  with pytest.raises(ValueError, match="未知の年金ステータスです"):
    invalid_pension = cast(PensionStatus, -1)
    _compile_lifeplan(
        replace(baseline_setup.lifeplan, pension_status=invalid_pension),
        baseline_setup.world)

  # 4. Gender
  with pytest.raises(ValueError, match="未知の性別です"):
    invalid_gender = cast(Gender, -1)
    _compile_lifeplan(
        replace(baseline_setup.lifeplan, mortality_gender=invalid_gender),
        baseline_setup.world)

  # 4.5. Unknown Stock Type in Asset Generator
  class DummyEnum:
    name = "DUMMY"

  with pytest.raises(ValueError, match="未知の株式タイプです"):
    invalid_stock = cast(PredefinedStock, DummyEnum())
    _compile_assets({invalid_stock}, baseline_setup.world)

  # 5. PredefinedAsset in _build_strategy (initial_asset_ratio)
  variant = baseline_setup.experiments[0] if baseline_setup.experiments else \
            from_setup_to_variant(baseline_setup)
  # 内部クラスへのアクセスが難しいため、create_experiment_setup を経由するか直接呼ぶ
  # ここでは _build_strategy を直接呼ぶためにモックを準備
  with pytest.raises(ValueError, match="未知の資産タイプです"):
    invalid_asset = cast(PredefinedStock, DummyEnum())
    bad_strategy = replace(baseline_setup.strategy,
                           initial_asset_ratio=((invalid_asset, 1.0),))
    from src.lib.scenario_builder import _ExperimentVariant
    v = _ExperimentVariant("bad", baseline_setup.lifeplan, bad_strategy,
                           baseline_setup.world)
    _build_strategy(v, {"BaseSpend": "bs"}, {}, {}, np.ones(10), False)

  # 6. PredefinedZeroRisk in dr_fn
  with pytest.raises(ValueError, match="リバランスの振り分け先に指定できない資産です"):
    bad_reb = DynamicV1Rebalance(risky_asset=PredefinedStock.ORUKAN_155,
                                 zero_risk_asset=cast(PredefinedZeroRisk, -1))
    bad_strategy = replace(baseline_setup.strategy, rebalance=bad_reb)
    v = _ExperimentVariant("bad_reb", baseline_setup.lifeplan, bad_strategy,
                           baseline_setup.world)
    compiled_strat = _build_strategy(v, {"BaseSpend": "bs"}, {}, {},
                                     np.ones(10), False)
    # ここで dr_fn を実行する必要がある
    dr_fn = compiled_strat.dynamic_rebalance_fn
    assert dr_fn is not None
    dr_fn(np.ones(10), np.ones(10), 10.0, np.ones(10))

  # 7. selling_priority
  with pytest.raises(ValueError, match="未知の資産タイプです"):
    invalid_asset = cast(PredefinedStock, DummyEnum())
    bad_strategy = replace(baseline_setup.strategy,
                           selling_priority=(invalid_asset,))
    v = _ExperimentVariant("bad_priority", baseline_setup.lifeplan,
                           bad_strategy, baseline_setup.world)
    _build_strategy(v, {"BaseSpend": "bs"}, {}, {}, np.ones(10), False)

  # 8. Invalid BaseSpend
  with pytest.raises(ValueError, match="未知の支出タイプです"):
    invalid_spend = cast(ConstantSpend, -1)
    _compile_lifeplan(
        replace(baseline_setup.lifeplan, base_spend=invalid_spend),
        baseline_setup.world)

  # 9. Invalid SpendAdjustment
  with pytest.raises(ValueError, match="未知の支出調整タイプです"):
    invalid_adj = cast(DynamicV1Adjustment, -1)
    bad_strategy = replace(baseline_setup.strategy,
                           spend_adjustment=invalid_adj)
    v = _ExperimentVariant("bad_adj", baseline_setup.lifeplan, bad_strategy,
                           baseline_setup.world)
    _build_strategy(v, {"BaseSpend": "bs"}, {}, {}, np.ones(10), False)


def from_setup_to_variant(setup: Setup):
  from src.lib.scenario_builder import _ExperimentVariant
  return _ExperimentVariant(setup.name, setup.lifeplan, setup.strategy,
                            setup.world)


def test_create_experiment_setup_returns_compiled_experiments_with_shared_memory(
    baseline_setup):
  """同じ World/Lifeplan を共有する実験が、メモリ上の同じ配列を参照していることを確認する。"""
  baseline_setup.add_experiment(name="exp1")
  compiled = create_experiment_setup(baseline_setup)
  assert len(compiled) == 2
  # Strategy は個別だが、prices と cashflows は共有されているはず
  assert compiled[0].monthly_prices is compiled[1].monthly_prices
  assert compiled[0].monthly_cashflows is compiled[1].monthly_cashflows


def test_compile_assets_additional_models(baseline_setup):
  """追加された資産モデル（SP500_30Y, ACWI_18Y 等）が正しく処理されることを確認する。"""
  models = [
      PredefinedStock.SP500_30Y, PredefinedStock.ACWI_18Y,
      PredefinedStock.ACWI_LOGNORMAL, PredefinedStock.ACWI_JSU,
      PredefinedStock.SIMPLE_7_15_ORUKAN
  ]
  for model in models:
    baseline_setup.strategy = replace(baseline_setup.strategy,
                                      initial_asset_ratio=((model, 1.0),),
                                      selling_priority=(model,))
    compiled = create_experiment_setup(baseline_setup)

    if model == PredefinedStock.SIMPLE_7_15_ORUKAN:
      assert "オルカン" in compiled[0].monthly_prices
    else:
      assert model.name in compiled[0].monthly_prices


def test_pension_household_size_doubling(baseline_setup):
  """世帯人数が2以上の場合、年金保険料が2倍になり、配偶者の基礎年金が追加されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 start_age=50,
                                 n_years=40,
                                 cpi_type=CpiType.FIXED_0)
  baseline_setup.lifeplan = replace(baseline_setup.lifeplan,
                                    retirement_start_age=55,
                                    pension_status=PensionStatus.FULL,
                                    pension_start_age=65,
                                    household_size=2)

  compiled = create_experiment_setup(baseline_setup)
  cashflows = compiled[0].monthly_cashflows
  cf_rules = compiled[0].strategy.cashflow_rules

  # 1. 保険料 (PensionPremium) が2倍
  premium_rule = next(r for r in cf_rules if "PensionPremium" in r.source_name)
  premium_arr = cashflows[premium_rule.source_name]
  # 1人分: -21.5 / 12 = -1.7916... -> 2人分: -43.0 / 12 = -3.5833...
  expected_premium_monthly = -(21.5 * 2.0) / 12.0
  assert np.allclose(premium_arr[0, :120], expected_premium_monthly)

  # 2. 配偶者の基礎年金 (PensionReceiptSpouseKiso)
  spouse_kiso_rule = next(
      r for r in cf_rules if "PensionReceiptSpouseKiso" in r.source_name)
  spouse_kiso_arr = cashflows[spouse_kiso_rule.source_name]
  # 81.6 / 12 = 6.8
  assert np.allclose(spouse_kiso_arr[0, 180:], 81.6 / 12.0)


@pytest.mark.parametrize("cpi_type, expected_mu, expected_sigma", [
    (CpiType.FIXED_1_0, 0.01, 0.0),
    (CpiType.FIXED_1_5, 0.015, 0.0),
    (CpiType.FIXED_2_0, 0.02, 0.0),
    (CpiType.FIXED_2_44, 0.0244, 0.0),
    (CpiType.FIXED_2_0_VOL_2_0, 0.02, 0.02),
    (CpiType.FIXED_2_0_VOL_4_13, 0.02, 0.0413),
    (CpiType.FIXED_2_44_VOL_4_13, 0.0244, 0.0413),
])
def test_compile_assets_cpi_variants(baseline_setup, cpi_type, expected_mu,
                                     expected_sigma):
  """追加された CPI バリアント (FIXED_1_0 等) が正しく処理されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 cpi_type=cpi_type,
                                 n_sim=100)
  compiled = create_experiment_setup(baseline_setup)
  prices = compiled[0].monthly_prices["Japan_CPI"]
  returns = np.diff(np.log(prices), axis=1)

  if expected_sigma == 0:
    expected_log_mu = np.log(1 + expected_mu) / 12
    assert np.isclose(np.mean(returns), expected_log_mu, atol=1e-5)
  else:
    assert np.isclose(np.std(returns) * np.sqrt(12), expected_sigma, atol=0.01)


def test_compile_assets_cpi_ar12(baseline_setup):
  """JAPAN_AR12 が正しく処理されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 cpi_type=CpiType.JAPAN_AR12,
                                 n_sim=5000,
                                 seed=42)
  compiled = create_experiment_setup(baseline_setup)
  prices = compiled[0].monthly_prices["Japan_CPI"]
  returns = np.diff(np.log(prices), axis=1)
  mu = np.mean(returns) * 12
  sigma = np.std(returns) * np.sqrt(12)
  # Observe values
  # JAPAN_AR12 mu: 0.013804591522419654, sigma: 0.019047852337144365
  assert np.isclose(mu, 0.0138, atol=1e-4)
  assert np.isclose(sigma, 0.0190, atol=1e-4)


def test_compile_assets_cpi_ar12_1981(baseline_setup):
  """JAPAN_AR12_1981 が正しく処理されることを確認する。"""
  baseline_setup.world = replace(baseline_setup.world,
                                 cpi_type=CpiType.JAPAN_AR12_1981,
                                 n_sim=5000,
                                 seed=42)
  compiled = create_experiment_setup(baseline_setup)
  prices = compiled[0].monthly_prices["Japan_CPI"]
  returns = np.diff(np.log(prices), axis=1)
  mu = np.mean(returns) * 12
  sigma = np.std(returns) * np.sqrt(12)
  # Observe values
  # JAPAN_AR12_1981 mu: 0.008471597082895313, sigma: 0.013307771701059729
  assert np.isclose(mu, 0.0085, atol=1e-4)
  assert np.isclose(sigma, 0.0133, atol=1e-4)
