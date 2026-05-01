"""
DSv1 (従来のガードレール) と DSv2 (生存確率ベース) の挙動の違いを分析する調査スクリプト。

特定のパスを抽出して、実質支出額の推移を FixedSpend をリファレンスとして比較します。
DSv2 が DSv1 に対してどのような挙動（特に早期の支出拡大と、その後の破産リスク）
を示すかを詳細に分析するために使用します。
"""
import os
import sys

# プロジェクトルートをPYTHONPATHに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import replace

import numpy as np

from src.core import simulate_strategy
from src.lib.retired_spending import SpendingType, get_retired_spending_values
from src.lib.scenario_builder import (CurveSpend, DynamicV1Adjustment, FxType,
                                      Lifeplan, PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup,
                                      SpendAwareAdjustment,
                                      SpendAwareDPRebalance, StrategySpec,
                                      WorldConfig, create_experiment_setup)


def main():
  # 共通設定
  SEED = 42
  YEARS = 55
  START_AGE = 40
  N_SIM = 500
  MODELS_PATH = "data/optimal_strategy_v2_models.json"
  CPI_NAME = "Japan_CPI"

  spending_types = (SpendingType.CONSUMPTION,
                    SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION)
  base_spending_monthly = get_retired_spending_values(
      list(spending_types), target_ages=np.array([float(START_AGE)]))[0]
  BASE_SPEND_ANNUAL_WO_PENSION = base_spending_monthly * 12.0 / 10000.0

  baseline_world = WorldConfig(n_sim=N_SIM,
                               n_years=YEARS,
                               start_age=START_AGE,
                               fx_type=FxType.USDJPY,
                               seed=SEED)

  baseline_lifeplan = Lifeplan(base_spend=CurveSpend(
      first_year_annual_amount=BASE_SPEND_ANNUAL_WO_PENSION,
      spending_types=spending_types),
                               retirement_start_age=START_AGE,
                               pension_status=PensionStatus.FULL,
                               pension_start_age=60)

  rule_val = 3.0
  base_spend_annual_init = BASE_SPEND_ANNUAL_WO_PENSION + 20.4
  init_money = base_spend_annual_init / (rule_val / 100.0)

  setup = Setup(name="Comparison",
                world=baseline_world,
                lifeplan=baseline_lifeplan,
                strategy=StrategySpec(
                    initial_money=float(init_money),
                    initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                                         (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                          0.0)),
                    selling_priority=(PredefinedStock.ORUKAN_155,
                                      PredefinedZeroRisk.ZERO_RISK_4PCT),
                    rebalance=SpendAwareDPRebalance(
                        risky_asset=PredefinedStock.ORUKAN_155,
                        zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                        model_name=MODELS_PATH)))

  # 1. DSv1 (+1.0%, -1.5%)
  adj_v1 = DynamicV1Adjustment(
      target_ratio=rule_val / 100.0,
      upper_limit=0.01,
      lower_limit=-0.015,
      initial_annual_spend=BASE_SPEND_ANNUAL_WO_PENSION)
  setup.add_experiment(name="DSv1",
                       overwrite_strategy=replace(setup.strategy,
                                                  spend_adjustment=adj_v1))

  # 2. DSv2 (+1.0%, -1.5%)
  adj_v2_new = SpendAwareAdjustment(model_name=MODELS_PATH,
                                    p_low=0.95,
                                    p_high=0.9999,
                                    lower_mult=0.985,
                                    upper_mult=1.01)
  setup.add_experiment(name="DSv2_1_15",
                       overwrite_strategy=replace(setup.strategy,
                                                  spend_adjustment=adj_v2_new))

  compiled_exps = create_experiment_setup(setup, record_annual_spend=True)

  results = {}
  for exp in compiled_exps:
    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)
    results[exp.name] = (res, exp)
    print(f"--- {exp.name} ---")
    print(f"Survival Rate: {np.mean(res.sustained_months >= YEARS * 12):.4f}")

  res0, exp0 = results["Comparison"]
  res1, exp1 = results["DSv1"]
  res2, exp2 = results["DSv2_1_15"]

  # DSv1は生き残ったが DSv2は破産したパスを探す
  v1_survived = res1.sustained_months >= YEARS * 12
  v2_failed = res2.sustained_months < YEARS * 12
  diff_mask = v1_survived & v2_failed
  diff_indices = np.where(diff_mask)[0]

  print(f"\nPaths where DSv1 survived but DSv2 failed: {diff_indices}")

  if len(diff_indices) > 0:
    target_idx = diff_indices[0]
    print(f"\n--- Analysis for Path {target_idx} ---")

    cpi = exp1.monthly_prices[CPI_NAME][target_idx, 0::12]

    assert res0.annual_spends is not None
    assert res1.annual_spends is not None
    assert res2.annual_spends is not None
    v0_spend = res0.annual_spends[target_idx]
    v1_spend = res1.annual_spends[target_idx]
    v2_spend = res2.annual_spends[target_idx]

    print(f"Year | CPI | Fixed Real | DSv1 Real | DSv2 Real | Fixed Nom | DSv1 Nom | DSv2 Nom")
    for y in range(min(YEARS, len(v1_spend))):
      cpi_y = cpi[y + 1] if y + 1 < len(cpi) else cpi[-1]
      print(
          f"{y:4d} | {cpi_y:4.2f} | {v0_spend[y]/cpi_y:10.2f} | {v1_spend[y]/cpi_y:9.2f} | {v2_spend[y]/cpi_y:9.2f} | {v0_spend[y]:9.2f} | {v1_spend[y]:8.2f} | {v2_spend[y]:8.2f}"
      )
      if v1_spend[y] == 0 and v2_spend[y] == 0 and y > 10:
        break
      if y > 50:
        break


if __name__ == "__main__":
  main()
