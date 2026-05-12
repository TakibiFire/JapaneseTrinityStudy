"""
50歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。

実験設定:
- 期間: 45年 (50歳〜95歳)
- 試行回数: 5,000回 (optimal-pension) / 2,000回 (その他)
- 資産構成:
    - オルカン (ファットテール考慮・S&P500補完モデル, 信託報酬 0.05775%)
    - ゼロリスク資産 (利回り 4.0%)
- ダイナミックリバランス: 毎年実施 (資産寿命を最大化する最適比率)
- 為替: USDJPY (期待リターン 0%, リスク 10.53%)
- インフレ: AR(12) 粘着性モデル
- 初年度支出ベースライン: 統計データに基づく50歳時の平均支出
- 税率: 20.315%
- 年金保険料: 50-60歳まで国民年金保険料を支払い (1人: 21.5万/年)
- 年金受給: 受給開始年齢 (60, 65, 70, 75) に応じた受給額

実験タイプ (--exp_type):
- optimal-pension: 年金開始年齢別の生存確率を、支出レベルと支出率のグリッドで評価
- pen70-lifeplan: 年金70歳開始を前提に、複数のリバランス戦略を比較
- pen70-formula: 年金70歳開始・DP最適化戦略の生存確率を詳細グリッドで評価
- pen70-ds: pen70-formula に動的な支出調整 (SpendAwareAdjustment) を追加して評価
"""

import argparse
import os
from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.dynamic_rebalance import (calculate_optimal_strategy)
from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (
    CpiType, CurveSpend, DynamicV1Rebalance, FixedRebalance, FxType, Lifeplan,
    PensionStatus, PredefinedStock, PredefinedZeroRisk, Setup,
    SpendAwareAdjustment, SpendAwareDPRebalance, StrategySpec, WorldConfig,
    create_experiment_setup)
from src.lib.world_setup import re50_pen70_95

# 共通設定
YEARS = 45  # 50歳から95歳まで
START_AGE = 50
SEED = 43
N_SIM_DEFAULT = 5000


def get_optimal_pension_setup(
    base_spend_50_retired: float, pension_premium_annual: float
) -> Tuple[Setup, int, List[Tuple[int, float, float]]]:
  """
  optimal-pension 実験設定を生成する。

  Args:
    base_spend_50_retired: 50歳時の基本支出額（年金保険料除く、万円）。
    pension_premium_annual: 年間の国民年金保険料（万円）。

  Returns:
    (Setup, int, combinations) のタプル。
  """
  spend_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
  spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  N_SIM = N_SIM_DEFAULT
  pension_start_ages = [60, 65, 70, 75]

  # 1. ベースライン設定
  baseline_world = WorldConfig(n_sim=N_SIM,
                               n_years=YEARS,
                               start_age=START_AGE,
                               seed=SEED,
                               cpi_type=CpiType.JAPAN_AR12,
                               fx_type=FxType.USDJPY)

  baseline_lifeplan = Lifeplan(
      base_spend=CurveSpend(first_year_annual_amount=0),  # Overwritten
      retirement_start_age=50,
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
    # 初年度支出 (国民年金保険料含む) と初期資産
    # 国民年金保険料は固定額とし、生活費のみを倍率 (spend_mult) でスケーリングする。
    initial_annual_cost = base_spend_50_retired * spend_mult + pension_premium_annual
    init_money = initial_annual_cost / (rule / 100.0)

    # scenario_builder が自動的に国民年金保険料 (-21.5) を加算するため、
    # CurveSpend には残りの額を設定する。
    initial_annual_cost_wo_premium = base_spend_50_retired * spend_mult

    new_lifeplan = replace(
        baseline_lifeplan,
        pension_start_age=pension_start,
        base_spend=CurveSpend(
            first_year_annual_amount=initial_annual_cost_wo_premium))

    new_strategy = replace(baseline_strategy,
                           initial_money=float(init_money),
                           spend_adjustment=None)

    exp_setup.add_experiment(
        name=f"P{pension_start}_Mult_{spend_mult}_Rule_{rule}%",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=new_strategy)

  return exp_setup, N_SIM, combinations


def get_pen70_lifeplan_setup(
    base_spend_50_retired: float, pension_premium_annual: float
) -> Tuple[Setup, int, List[Tuple[float, float, str]]]:
  """
  pen70-lifeplan 実験設定を生成する。
  """
  spend_multipliers = [0.75, 1.0, 1.5, 2.0, 3.0]
  spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  strategy_names = [
      "No dynamic rebalance", "固定最適比率", "DynamicV1Rebalance",
      "SpendAwareDPRebalance (R70-aware)"
  ]
  N_SIM = 2000

  # 1. ベースライン設定 (re50_pen70_95)
  exp_setup = re50_pen70_95(n_sim=N_SIM, seed=SEED)
  exp_setup.name = "pen70-lifeplan"

  # 2. グリッドパラメータ
  combinations = list(product(spend_multipliers, spending_rules,
                              strategy_names))

  for (spend_mult, rule, strat_name) in combinations:
    # 初年度支出 (国民年金保険料含む) と初期資産
    initial_annual_cost = base_spend_50_retired * spend_mult + pension_premium_annual
    init_money = initial_annual_cost / (rule / 100.0)

    # scenario_builder が自動的に国民年金保険料 (-21.5) を加算するため、
    # CurveSpend には残りの額を設定する。
    initial_annual_cost_wo_premium = base_spend_50_retired * spend_mult

    new_lifeplan = replace(
        exp_setup.lifeplan,
        base_spend=CurveSpend(
            first_year_annual_amount=initial_annual_cost_wo_premium))

    # 戦略の設定
    spec = StrategySpec(initial_money=float(init_money),
                        initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                                             (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                              0.0)),
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
    elif strat_name == "SpendAwareDPRebalance (R70-aware)":
      # 倍率に応じたモデルを選択
      mult_map = {0.75: "m0_75", 1.0: "m1", 1.5: "m1_5", 2.0: "m2", 3.0: "m3"}
      mult_suffix = mult_map.get(spend_mult, "m1")

      model_path = f"data/optimal_strategy_dp/re50_pen70_95_{mult_suffix}.json"

      spec = replace(spec,
                     rebalance=SpendAwareDPRebalance(
                         risky_asset=PredefinedStock.ORUKAN_155,
                         zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                         model_name=model_path))

    exp_setup.add_experiment(
        name=f"Mult_{spend_mult}_Rule_{rule}%_{strat_name}",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=spec)

  return exp_setup, N_SIM, combinations


def get_pen70_formula_setup(
    base_spend_50_retired: float, pension_premium_annual: float
) -> Tuple[Setup, int, List[Tuple[float, float, str]]]:
  """
  pen70-formula 実験設定を生成する。
  """
  spend_multipliers = [0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
  spending_rules = [
      2.38, 2.5, 2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0
  ]
  strategy_names = ["SpendAwareDPRebalance (R70-aware)"]
  N_SIM = 2000

  # 1. ベースライン設定 (re50_pen70_95)
  exp_setup = re50_pen70_95(n_sim=N_SIM, seed=SEED)
  exp_setup.name = "pen70-formula"

  # 2. グリッドパラメータ
  combinations = list(product(spend_multipliers, spending_rules,
                              strategy_names))

  for (spend_mult, rule, strat_name) in combinations:
    # 初年度支出 (国民年金保険料含む) と初期資産
    initial_annual_cost = base_spend_50_retired * spend_mult + pension_premium_annual
    init_money = initial_annual_cost / (rule / 100.0)

    # scenario_builder が自動的に国民年金保険料 (-21.5) を加算するため、
    # CurveSpend には残りの額を設定する。
    initial_annual_cost_wo_premium = base_spend_50_retired * spend_mult

    new_lifeplan = replace(
        exp_setup.lifeplan,
        base_spend=CurveSpend(
            first_year_annual_amount=initial_annual_cost_wo_premium))

    # 戦略の設定
    spec = StrategySpec(initial_money=float(init_money),
                        initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                                             (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                              0.0)),
                        selling_priority=(PredefinedStock.ORUKAN_155,
                                          PredefinedZeroRisk.ZERO_RISK_4PCT))

    if strat_name == "SpendAwareDPRebalance (R70-aware)":
      # 倍率に応じたモデルを選択
      mult_map = {
          0.75: "m0_75",
          1.0: "m1",
          1.2: "m1_2",
          1.5: "m1_5",
          2.0: "m2",
          3.0: "m3"
      }
      mult_suffix = mult_map.get(spend_mult, "m1")

      model_path = f"data/optimal_strategy_dp/re50_pen70_95_{mult_suffix}.json"

      spec = replace(spec,
                     rebalance=SpendAwareDPRebalance(
                         risky_asset=PredefinedStock.ORUKAN_155,
                         zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                         model_name=model_path))

    exp_setup.add_experiment(
        name=f"Mult_{spend_mult}_Rule_{rule}%_{strat_name}",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=spec)

  return exp_setup, N_SIM, combinations


def get_pen70_ds_setup(
    base_spend_50_retired: float, pension_premium_annual: float
) -> Tuple[Setup, int, List[Tuple[float, float, str]]]:
  """
  pen70-ds 実験設定を生成する。
  SpendAwareAdjustment を有効化した pen70-formula 相当の設定。
  """
  spend_multipliers = [0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
  spending_rules = [
      2.38, 2.5, 2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0
  ]
  strategy_names = ["SpendAwareDPRebalance (R70-aware)"]
  N_SIM = 2000

  # 1. ベースライン設定 (re50_pen70_95)
  exp_setup = re50_pen70_95(n_sim=N_SIM, seed=SEED)
  exp_setup.name = "pen70-ds"

  # 2. グリッドパラメータ
  combinations = list(product(spend_multipliers, spending_rules,
                              strategy_names))

  for (spend_mult, rule, strat_name) in combinations:
    # 初年度支出 (国民年金保険料含む) と初期資産
    initial_annual_cost = base_spend_50_retired * spend_mult + pension_premium_annual
    init_money = initial_annual_cost / (rule / 100.0)

    # scenario_builder が自動的に国民年金保険料 (-21.5) を加算するため、
    # CurveSpend には残りの額を設定する。
    initial_annual_cost_wo_premium = base_spend_50_retired * spend_mult

    new_lifeplan = replace(
        exp_setup.lifeplan,
        base_spend=CurveSpend(
            first_year_annual_amount=initial_annual_cost_wo_premium))

    # 倍率に応じたモデルを選択
    mult_map = {
        0.75: "m0_75",
        1.0: "m1",
        1.2: "m1_2",
        1.5: "m1_5",
        2.0: "m2",
        3.0: "m3"
    }
    mult_suffix = mult_map.get(spend_mult, "m1")
    model_path = f"data/optimal_strategy_dp/re50_pen70_95_{mult_suffix}.json"

    # 戦略の設定
    spec = StrategySpec(
        initial_money=float(init_money),
        initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                             (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
        selling_priority=(PredefinedStock.ORUKAN_155,
                          PredefinedZeroRisk.ZERO_RISK_4PCT),
        rebalance=SpendAwareDPRebalance(
            risky_asset=PredefinedStock.ORUKAN_155,
            zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
            model_name=model_path),
        spend_adjustment=SpendAwareAdjustment(model_name=model_path,
                                              p_low=0.97,
                                              p_high=0.9999,
                                              lower_mult=0.98,
                                              upper_mult=1.01))

    exp_setup.add_experiment(
        name=f"Mult_{spend_mult}_Rule_{rule}%_{strat_name}",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=spec)

  return exp_setup, N_SIM, combinations


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="50歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。")
  parser.add_argument(
      "--exp_type",
      type=str,
      default="optimal-pension",
      help="実験設定 (optimal-pension, pen70-lifeplan, pen70-formula, pen70-ds)")
  args = parser.parse_args()

  # 設定
  exp_type = args.exp_type
  assert exp_type in ("optimal-pension", "pen70-lifeplan", "pen70-formula",
                      "pen70-ds"), f"Unsupported exp_type: {exp_type}"

  data_dir = "data/all_50yr/"
  csv_path = os.path.join(data_dir, f"{exp_type}.csv")
  os.makedirs(data_dir, exist_ok=True)

  # 初年度支出ベースライン (50歳、年金保険料除く)
  base_spend_50_retired = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      50, 1)[0]
  # 国民年金保険料 (1人)
  pension_premium_annual = 21.5

  exp_setup: Setup
  n_sim_val: int
  combinations: List[Any]

  if exp_type == "optimal-pension":
    exp_setup, n_sim_val, combinations = get_optimal_pension_setup(
        base_spend_50_retired, pension_premium_annual)
  elif exp_type == "pen70-lifeplan":
    exp_setup, n_sim_val, combinations = get_pen70_lifeplan_setup(
        base_spend_50_retired, pension_premium_annual)
  elif exp_type == "pen70-formula":
    exp_setup, n_sim_val, combinations = get_pen70_formula_setup(
        base_spend_50_retired, pension_premium_annual)
  elif exp_type == "pen70-ds":
    exp_setup, n_sim_val, combinations = get_pen70_ds_setup(
        base_spend_50_retired, pension_premium_annual)
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
    else:  # pen70-lifeplan, pen70-formula, pen70-ds
      spend_mult, rule, strat_name = combo
      pension_start = 70  # re50_pen70_95 固定

    initial_annual_cost = base_spend_50_retired * spend_mult + pension_premium_annual
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
      survival_rate = 1.0 - (bankrupt_count / n_sim_val)
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
