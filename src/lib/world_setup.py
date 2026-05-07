"""
資産運用・取り崩しシミュレーションのための標準的な「世界」設定を構築するモジュール。
"""

from dataclasses import replace

from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (CpiType, CurveSpend, FxType, Lifeplan,
                                      PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup, StrategySpec,
                                      WorldConfig)


def create_standard_world(
    n_sim: int,
    start_age: int,
    end_age_inclusive: int,
    retirement_start_age: int,
    pension_start_age: int,
    seed: int = 42,
) -> Setup:
  """
  標準的な世界設定（Setupオブジェクト）を構築します。

  Args:
    n_sim: シミュレーション試行回数
    start_age: シミュレーション開始年齢
    end_age: シミュレーション終了年齢（この年齢の終わりまで）
    retirement_start_age: 定期的な給与収入が停止する年齢。
      将来の年金受給額の計算において、22歳からこの年齢（この年齢自体は含まない）までの期間、厚生年金に加入していたものとして扱われる。
      例えば 35歳でリタイアした場合、34.999...歳までの加入実績に基づいた年金額が計算される。
    pension_start_age: 年金受給開始年齢
    seed: 乱数シード

  Returns:
    Setup: 設定オブジェクト
  """
  years = end_age_inclusive + 1 - start_age

  world_config = WorldConfig(n_sim=n_sim,
                             n_years=years,
                             start_age=start_age,
                             seed=seed,
                             cpi_type=CpiType.JAPAN_AR12,
                             fx_type=FxType.USDJPY)

  lifeplan = Lifeplan(base_spend=CurveSpend(),
                      retirement_start_age=retirement_start_age,
                      pension_status=PensionStatus.FULL,
                      pension_start_age=pension_start_age,
                      household_size=1)

  strategy_spec = StrategySpec(
      initial_money=10000.0,  # プレースホルダ
      initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                           (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
      selling_priority=(PredefinedStock.ORUKAN_155,
                        PredefinedZeroRisk.ZERO_RISK_4PCT))

  return Setup("standard_world", world_config, lifeplan, strategy_spec)


def re40_pen60_95(n_sim: int, seed: int = 42) -> Setup:
  """
  開始40歳、リタイア40歳、年金開始60歳、終了94歳末のシナリオ設定を構築します。
  """
  return create_standard_world(n_sim=n_sim,
                               start_age=40,
                               end_age_inclusive=94,
                               retirement_start_age=40,
                               pension_start_age=60,
                               seed=seed)


def re50_pen70_95(n_sim: int, seed: int = 42) -> Setup:
  """
  開始50歳、リタイア50歳、年金開始70歳、終了94歳末のシナリオ設定を構築します。
  """
  return create_standard_world(n_sim=n_sim,
                               start_age=50,
                               end_age_inclusive=94,
                               retirement_start_age=50,
                               pension_start_age=70,
                               seed=seed)


def re60_pen70_95(n_sim: int, seed: int = 42) -> Setup:
  """
  開始60歳、リタイア60歳、年金開始70歳、終了94歳末のシナリオ設定を構築します。
  """
  return create_standard_world(n_sim=n_sim,
                               start_age=60,
                               end_age_inclusive=94,
                               retirement_start_age=60,
                               pension_start_age=70,
                               seed=seed)


def _create_re50_pen70_with_mult(n_sim: int,
                                 multiplier: float,
                                 seed: int = 42) -> Setup:
  """
  開始50歳、リタイア50歳、年金開始70歳、終了94歳末で、初期支出倍率を指定したシナリオ設定を構築します。
  """
  setup = re50_pen70_95(n_sim=n_sim, seed=seed)

  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      50, 1)[0]
  initial_annual_cost = base_spend_annual * multiplier

  # Lifeplan の支出設定を上書き
  setup.lifeplan = replace(
      setup.lifeplan,
      base_spend=CurveSpend(first_year_annual_amount=initial_annual_cost))

  return setup


def re50_pen70_95_m0_75(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 0.75, seed)


def re50_pen70_95_m1(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 1.0, seed)


def re50_pen70_95_m1_2(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 1.2, seed)


def re50_pen70_95_m1_5(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 1.5, seed)


def re50_pen70_95_m2(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 2.0, seed)


def re50_pen70_95_m3(n_sim: int, seed: int = 42) -> Setup:
  return _create_re50_pen70_with_mult(n_sim, 3.0, seed)


def _create_re60_pen70_with_mult(n_sim: int,
                                 multiplier: float,
                                 seed: int = 42) -> Setup:
  """
  開始60歳、リタイア60歳、年金開始70歳、終了94歳末で、初期支出倍率を指定したシナリオ設定を構築します。
  """
  setup = re60_pen70_95(n_sim=n_sim, seed=seed)

  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      60, 1)[0]
  initial_annual_cost = base_spend_annual * multiplier

  # Lifeplan の支出設定を上書き
  setup.lifeplan = replace(
      setup.lifeplan,
      base_spend=CurveSpend(first_year_annual_amount=initial_annual_cost))

  return setup


def re60_pen70_95_m0_75(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 0.75, seed)


def re60_pen70_95_m1(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 1.0, seed)


def re60_pen70_95_m1_2(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 1.2, seed)


def re60_pen70_95_m1_5(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 1.5, seed)


def re60_pen70_95_m2(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 2.0, seed)


def re60_pen70_95_m3(n_sim: int, seed: int = 42) -> Setup:
  return _create_re60_pen70_with_mult(n_sim, 3.0, seed)
