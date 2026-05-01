"""
Spend-Aware Dynamic Spending のグリッドシミュレーション実行スクリプト。

引数 --exp_name によって異なる実験を実行します。
- simple: 単一のパラメータ設定でのデモ実行
- v1_v2_comp: DSv1 (従来のガードレール) と DSv2 (生存確率ベース) の比較

全戦略で DRv2 (DPベースの動的リバランス) を使用します。
"""

import argparse
import os
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from src.core import SimulationResult, simulate_strategy
from src.lib.retired_spending import SpendingType, get_retired_spending_values
from src.lib.scenario_builder import (CurveSpend, DynamicV1Adjustment, FxType,
                                      Lifeplan, PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup,
                                      SpendAwareAdjustment,
                                      SpendAwareDPRebalance, StrategySpec,
                                      WorldConfig, create_experiment_setup)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name",
                      type=str,
                      default="simple",
                      choices=["simple", "v1_v2_comp"])
  args = parser.parse_args()

  # 共通設定
  SEED = 42
  YEARS = 55
  START_AGE = 40
  N_SIM = 2000
  MODELS_PATH = "data/optimal_strategy_v2_models.json"
  CPI_NAME = "Japan_CPI"

  # ベースラインの支出額 (月額合計 -> 年額合計)
  spending_types = (SpendingType.CONSUMPTION,
                    SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION)
  base_spending_monthly = get_retired_spending_values(
      list(spending_types), target_ages=np.array([float(START_AGE)]))[0]
  BASE_SPEND_ANNUAL_WO_PENSION = base_spending_monthly * 12.0 / 10000.0

  # 年金設定 (旧コードの値を再現するために PensionStatus.FULL をベースに調整)
  # 旧コード: PREMIUM=20.4, TOTAL=99.4, KISO=81.6*0.76=62.016, KOUSEI=37.384
  # scenario_builder: PREMIUM=21.5, KISO=81.6*0.76=62.016, KOUSEI=2.736*(40-22)*0.76=37.428
  # わずかな差があるが、no-op 互換性のために、旧コードの値に合わせる必要がある場合は
  # scenario_builder 側を調整するか、ここでのパラメータを工夫する。
  # 今回は scenario_builder の標準ロジックを使用する。

  # ベースライン設定
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
  baseline_strategy = StrategySpec(
      initial_money=0.0,  # あとで設定
      initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                           (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
      selling_priority=(PredefinedStock.ORUKAN_155,
                        PredefinedZeroRisk.ZERO_RISK_4PCT),
      rebalance=SpendAwareDPRebalance(
          risky_asset=PredefinedStock.ORUKAN_155,
          zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
          model_name=MODELS_PATH))

  exp_setup = Setup(name="spend_aware_grid",
                    world=baseline_world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  if args.exp_name == "simple":
    rules = [4.0]
    strategies = ["FixedSpend", "SpendAware"]
  else:
    rules = [3.0, 3.5, 4.0, 4.5, 5.0]
    strategies = ["DRv2_DSv1", "DRv2_DSv2", "FixedSpend"]

  # 実験の追加
  for rule in rules:
    # 初期資産の計算 (旧コード互換: PREMIUM_ANNUAL=20.4 を使用)
    base_spend_annual_init = BASE_SPEND_ANNUAL_WO_PENSION + 20.4
    init_money = base_spend_annual_init / (rule / 100.0)

    for strat_name in strategies:
      current_init_money = init_money
      if strat_name == "FixedSpend":
        current_init_money = 100 * 10000  # 100億円

      new_strategy = replace(baseline_strategy,
                             initial_money=float(current_init_money))

      adjustment: Union[SpendAwareAdjustment, DynamicV1Adjustment, None] = None
      if strat_name == "FixedSpend":
        pass
      elif strat_name == "SpendAware" or strat_name == "DRv2_DSv2":
        adjustment = SpendAwareAdjustment(model_name=MODELS_PATH,
                                          p_low=0.95,
                                          p_high=0.9999,
                                          lower_mult=0.985,
                                          upper_mult=1.01)
      elif strat_name == "DRv2_DSv1":
        adjustment = DynamicV1Adjustment(target_ratio=rule / 100.0,
                                         upper_limit=0.01,
                                         lower_limit=-0.015,
                                         initial_annual_spend=BASE_SPEND_ANNUAL_WO_PENSION)

      new_strategy = replace(new_strategy, spend_adjustment=adjustment)

      exp_setup.add_experiment(name=f"Rule{rule}_{strat_name}",
                               overwrite_strategy=new_strategy)

  # コンパイル
  compiled_exps = create_experiment_setup(exp_setup, record_annual_spend=True)

  results_summary = []
  results_survival_probs = []
  results_spends = []

  # 実行 (ベースラインは飛ばす)
  print(f"シミュレーション実行中... (全 {len(rules) * len(strategies)} パターン)")

  res_by_rule: Dict[float, Dict[str, tuple]] = defaultdict(dict)

  for i, exp in enumerate(compiled_exps[1:]):
    rule = rules[i // len(strategies)]
    strat_name = strategies[i % len(strategies)]

    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)
    res_by_rule[rule][strat_name] = (res, exp)

    # 生存確率データの蓄積
    if strat_name != "FixedSpend":
      for y in range(YEARS + 1):
        survival_rate = np.mean(res.sustained_months >= y * 12)
        results_survival_probs.append({
            "rule": rule,
            "strategy": strat_name,
            "year": y,
            "survival_rate": survival_rate
        })

      surv_rate = np.mean(res.sustained_months == YEARS * 12)
      results_summary.append({
          "rule": rule,
          "strategy": strat_name,
          "survival_rate": surv_rate
      })

  # 支出データの集計 (DSv1, DSv2 両方が生存しているパスのみ)
  if args.exp_name == "v1_v2_comp":
    for rule, res_dict in res_by_rule.items():
      rv1_tup = res_dict.get("DRv2_DSv1")
      rv2_tup = res_dict.get("DRv2_DSv2")
      rfx_tup = res_dict.get("FixedSpend")

      if rv1_tup is not None and rv2_tup is not None and rfx_tup is not None:
        rv1, exp1 = rv1_tup
        rv2, exp2 = rv2_tup
        prices_cpi = exp1.monthly_prices[CPI_NAME]
        
        for y in range(YEARS):
          # DSv1, DSv2 両戦略が年末時点で生存しているパス
          active_mask = (rv1.sustained_months >= (y + 1) * 12) & \
                        (rv2.sustained_months >= (y + 1) * 12)

          if np.any(active_mask):
            for strat_name in strategies:
              res_obj, _ = res_dict[strat_name]
              if res_obj.annual_spends is not None:
                real_vals = res_obj.annual_spends[active_mask,
                                                  y] / prices_cpi[active_mask,
                                                                  y * 12]
                results_spends.append({
                    "rule": rule,
                    "strategy": strat_name,
                    "year": y + 1,
                    "p25": np.percentile(real_vals, 25),
                    "p50": np.percentile(real_vals, 50),
                    "p75": np.percentile(real_vals, 75)
                })

  # 結果の保存
  data_dir = f"data/spend_aware_dynamic_spending"
  os.makedirs(data_dir, exist_ok=True)

  df_summary = pd.DataFrame(results_summary)
  df_survival = pd.DataFrame(results_survival_probs)
  df_spends = pd.DataFrame(results_spends)

  summary_path = os.path.join(data_dir, f"{args.exp_name}_summary.csv")
  survival_path = os.path.join(data_dir, f"{args.exp_name}_survival.csv")
  spends_path = os.path.join(data_dir, f"{args.exp_name}_spends.csv")

  df_summary.to_csv(summary_path, index=False)
  df_survival.to_csv(survival_path, index=False)
  df_spends.to_csv(spends_path, index=False)

  print(f"\n結果を保存しました:\n- {summary_path}\n- {survival_path}\n- {spends_path}")


if __name__ == "__main__":
  main()
