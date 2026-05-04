"""
60歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。

実験設定:
- 期間: 35年 (60歳〜95歳)
- 試行回数: 5,000回
- 資産構成:
    - オルカン (ファットテール考慮・S&P500補完モデル, 信託報酬 0.05775%)
    - ゼロリスク資産 (利回り 4.0%)
- ダイナミックリバランス: 毎年実施 (資産寿命を最大化する最適比率)
- 為替: USDJPY (期待リターン 0%, リスク 10.53%)
- インフレ: AR(12) 粘着性モデル
- 初年度支出ベースライン: 540万/年 (60歳の出費平均45万 * 12か月)
  - ちなみにこの値は2人以上の世帯
  - 65歳以上単身無職世帯は 16.2万 (45万の 36%相当)
- 税率: 20.315%
- 年金: 60歳または65歳から受給 (世帯人数と開始年齢により変動)

可変条件:
- 年金受給開始年齢 (60, 65)
- ダイナミックスペンディングの有無
- 支出率のルール (資産額に対する比率)
- 初年度支出倍率
"""

import argparse
import os
from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.dynamic_rebalance import (calculate_optimal_strategy,
                                       calculate_safe_target_ratio)
from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (ConstantSpend, CpiType, CurveSpend,
                                      DynamicV1Adjustment, DynamicV1Rebalance,
                                      FixedRebalance, FxType, Lifeplan,
                                      PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup,
                                      SpendAwareDPRebalance, StrategySpec,
                                      WorldConfig, create_experiment_setup)
from src.lib.world_setup import re60_pen60_95

# 共通設定
YEARS = 36  # 60歳から95歳終了まで (36年間)
START_AGE = 60
SEED = 43


def get_optimal_pension_setup(
    base_spend_annual: float) -> Tuple[Setup, int, List[Tuple[int, float, float]]]:
  """
  optimal-pension 実験設定を生成する。

  Args:
    base_spend_annual: 初年度の基本支出額 (万円)。

  Returns:
    (Setup, int, combinations) のタプル。
  """
  spend_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
  spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  N_SIM = 2000
  pension_start_ages = [60, 65, 70, 75]

  # 1. ベースライン設定
  baseline_world = WorldConfig(
      n_sim=N_SIM,  # Overwritten
      n_years=YEARS,
      start_age=START_AGE,
      seed=SEED,
      cpi_type=CpiType.JAPAN_AR12,
      fx_type=FxType.USDJPY)

  baseline_lifeplan = Lifeplan(
      base_spend=ConstantSpend(annual_amount=0),  # Overwritten
      retirement_start_age=60,
      pension_status=PensionStatus.FULL,
      pension_start_age=65)  # Overwritten

  baseline_strategy = StrategySpec(
      initial_money=10000.0,  # Overwritten
      initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                           (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
      selling_priority=(PredefinedStock.ORUKAN_155,
                        PredefinedZeroRisk.ZERO_RISK_4PCT),
      rebalance=DynamicV1Rebalance(
          risky_asset=PredefinedStock.ORUKAN_155,
          zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
          interval_months=12))

  exp_setup = Setup(name="baseline",
                    world=baseline_world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  # 2. グリッドパラメータ
  combinations = list(
      product(pension_start_ages, spend_multipliers, spending_rules))

  for (pension_start, spend_mult, rule) in combinations:
    initial_annual_cost = base_spend_annual * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    new_lifeplan = replace(
        baseline_lifeplan,
        pension_start_age=pension_start,
        base_spend=CurveSpend(first_year_annual_amount=initial_annual_cost))

    new_strategy = replace(baseline_strategy,
                           initial_money=float(init_money),
                           spend_adjustment=None)

    exp_setup.add_experiment(
        name=f"P{pension_start}_Mult_{spend_mult}_Rule_{rule}%",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=new_strategy)

  return exp_setup, N_SIM, combinations


def get_pen60_lifeplan_setup(
    base_spend_annual: float
) -> Tuple[Setup, int, List[Tuple[float, float, str]]]:
  """
  pen60-lifeplan 実験設定を生成する。
  """
  spend_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
  spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  strategy_names = [
      "No dynamic rebalance", "固定最適比率", "DynamicV1Rebalance",
      "SpendAwareDPRebalance (re60)", "SpendAwareDPRebalance (re40)"
  ]
  N_SIM = 2000

  # 1. ベースライン設定 (re60_pen60_95)
  exp_setup = re60_pen60_95(n_sim=N_SIM, seed=SEED)
  exp_setup.name = "pen60-lifeplan"

  # 2. グリッドパラメータ
  combinations = list(product(spend_multipliers, spending_rules, strategy_names))

  for (spend_mult, rule, strat_name) in combinations:
    initial_annual_cost = base_spend_annual * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    new_lifeplan = replace(
        exp_setup.lifeplan,
        base_spend=CurveSpend(first_year_annual_amount=initial_annual_cost))

    # 戦略の設定
    spec = StrategySpec(
        initial_money=float(init_money),
        initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                             (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
        selling_priority=(PredefinedStock.ORUKAN_155,
                          PredefinedZeroRisk.ZERO_RISK_4PCT))

    if strat_name == "No dynamic rebalance":
      spec = replace(spec, rebalance=FixedRebalance())
    elif strat_name == "固定最適比率":
      fixed_ratio = calculate_optimal_strategy(s_rate=np.array([rule / 100.0]),
                                               remaining_years=YEARS,
                                               base_yield=0.04,
                                               tax_rate=0.20315,
                                               inflation_rate=0.0177)[0]
      spec = replace(spec,
                     initial_asset_ratio=((PredefinedStock.ORUKAN_155,
                                           fixed_ratio),
                                          (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                           1.0 - fixed_ratio)),
                     rebalance=FixedRebalance())
    elif strat_name == "DynamicV1Rebalance":
      spec = replace(spec,
                     rebalance=DynamicV1Rebalance(
                         risky_asset=PredefinedStock.ORUKAN_155,
                         zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT))
    elif strat_name == "SpendAwareDPRebalance (re60)":
      spec = replace(spec,
                     rebalance=SpendAwareDPRebalance(
                         risky_asset=PredefinedStock.ORUKAN_155,
                         zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                         model_name="data/optimal_strategy_dp/re60_pen60_95.json")
                     )
    elif strat_name == "SpendAwareDPRebalance (re40)":
      spec = replace(spec,
                     rebalance=SpendAwareDPRebalance(
                         risky_asset=PredefinedStock.ORUKAN_155,
                         zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                         model_name="data/optimal_strategy_dp/re40_pen60_95.json")
                     )

    exp_setup.add_experiment(
        name=f"Mult_{spend_mult}_Rule_{rule}%_{strat_name}",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=spec)

  return exp_setup, N_SIM, combinations


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="60歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="optimal-pension",
                      help="実験設定 (optimal-pension, pen60-lifeplan)")
  args = parser.parse_args()

  # 設定
  exp_type = args.exp_type
  assert exp_type in ("optimal-pension",
                      "pen60-lifeplan"), f"Unsupported exp_type: {exp_type}"

  data_dir = "data/all_60yr/"
  csv_path = os.path.join(data_dir, f"{exp_type}.csv")
  os.makedirs(data_dir, exist_ok=True)

  # 初年度支出ベースライン
  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      60, 1)[0]

  if exp_type == "optimal-pension":
    exp_setup, N_SIM, combinations = get_optimal_pension_setup(
        base_spend_annual)
  elif exp_type == "pen60-lifeplan":
    exp_setup, N_SIM, combinations = get_pen60_lifeplan_setup(base_spend_annual)
  else:
    raise KeyError(f"Unsupported {exp_type}")

  # 3. コンパイルとシミュレーション
  print(f"全 {len(combinations)} パターンのシミュレーションを実行中...")
  compiled_experiments = create_experiment_setup(exp_setup,
                                                 record_annual_spend=True)

  results: List[Dict[str, Any]] = []

  # ベースラインをスキップし、オリジナルの組み合わせとジップして結果を処理
  for i, (exp, combo) in enumerate(zip(compiled_experiments[1:], combinations)):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(combinations)}")

    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)

    if exp_type == "optimal-pension":
      pension_start, spend_mult, rule = combo
      strat_name = "DynamicV1Rebalance"
    else:  # pen60-lifeplan
      spend_mult, rule, strat_name = combo
      pension_start = 60  # re60_pen60_95 固定

    initial_annual_cost = base_spend_annual * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    base_row: Dict[str, Union[float, int, str]] = {
        "pension_start_age": pension_start,
        "spend_multiplier": spend_mult,
        "strategy": strat_name,
        "spending_rule": rule,
        "initial_money": init_money,
        "initial_annual_cost": initial_annual_cost,
    }

    # 1. 生存確率
    row_survival = base_row.copy()
    row_survival["value_type"] = "survival"
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row_survival[str(year)] = survival_rate
    results.append(row_survival)

    # 2. 支出額のパーセンタイル
    if res.annual_spends is not None:
      p25 = np.percentile(res.annual_spends, 25, axis=0)
      p50 = np.percentile(res.annual_spends, 50, axis=0)
      p75 = np.percentile(res.annual_spends, 75, axis=0)

      for name, p_values in [("spend25p", p25), ("spend50p", p50),
                             ("spend75p", p75)]:
        row_p = base_row.copy()
        row_p["value_type"] = name
        for year in range(1, YEARS + 1):
          row_p[str(year)] = p_values[year - 1]
        results.append(row_p)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
  print(f"完了。結果を {csv_path} に保存しました。")



if __name__ == "__main__":
  main()
