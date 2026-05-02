"""
資産運用・取り崩しシミュレーションのための標準的な「世界」設定を構築するモジュール。
"""

from src.lib.scenario_builder import (CpiType, CurveSpend, FxType, Lifeplan,
                                      PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup, StrategySpec,
                                      WorldConfig)


def create_standard_world(
    n_sim: int,
    start_age: int,
    end_age: int,
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
  years = end_age + 1 - start_age

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
