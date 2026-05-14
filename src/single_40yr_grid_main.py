"""
40歳リタイア開始・95歳開始（94歳末）までの生存確率を分析するグリッドサーチスクリプト（単身世帯版）。

実験設定:
- 期間: 55年 (40歳〜95歳開始まで)
- 試行回数: 2,000回
- 資産構成:
    - オルカン (ファットテール考慮・S&P500補完モデル, 信託報酬 0.05775%)
    - ゼロリスク資産 (利回り 4.0%)
- ダイナミックリバランス: 毎年実施 (資産寿命を最大化する最適比率)
- 為替: USDJPY (期待リターン 0%, リスク 10.53%)
- インフレ: AR(12) 粘着性モデル
- 初年度支出ベースライン: 単身世帯の2025年統計データに基づく40歳時の平均支出
- 税率: 20.315%
- 年金受給: 受給開始年齢 (60, 62, 65, 68, 71, 73, 75) に応じた受給額

実験タイプ (--exp_type):
- optimal-pension: 年金開始年齢別の生存確率を、支出レベルと支出率のグリッドで評価
"""

import argparse
import os
from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (CurveSpend, DynamicV1Rebalance,
                                      PredefinedStock, PredefinedZeroRisk,
                                      Setup, StrategySpec,
                                      create_experiment_setup)
from src.lib.world_setup import create_single_world

# 共通設定
YEARS = 55  # 40歳から95歳開始まで (55年間)
START_AGE = 40
SEED = 43


def get_optimal_pension_setup(
    base_spend_annual: float, pension_premium_annual: float
) -> Tuple[Setup, int, List[Tuple[int, float, float]]]:
  """
  optimal-pension 実験設定を生成する。

  Args:
    base_spend_annual: 初年度の基本支出額 (万円)。
    pension_premium_annual: 年間の国民年金保険料（万円）。

  Returns:
    (Setup, int, combinations) のタプル。
  """
  spend_multipliers = [0.6, 0.8, 1.0, 1.1, 1.2, 1.5, 1.8]
  spending_rules = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
  N_SIM = 2000
  pension_start_ages = [60, 62, 65, 68, 71, 73, 75]

  # 1. ベースライン設定 (create_single_world を使用)
  exp_setup = create_single_world(
      n_sim=N_SIM,
      start_age=START_AGE,
      end_age_inclusive=START_AGE + YEARS - 1,
      retirement_start_age=START_AGE,
      pension_start_age=65,  # 後で上書き
      seed=SEED)

  # 戦略のデフォルト設定 (DynamicV1Rebalance)
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

  exp_setup.strategy = baseline_strategy

  # 2. グリッドパラメータ
  combinations = list(
      product(pension_start_ages, spend_multipliers, spending_rules))

  for (pension_start, spend_mult, rule) in combinations:
    # 初年度支出 (国民年金保険料含む) と初期資産
    initial_annual_cost = base_spend_annual * spend_mult + pension_premium_annual
    init_money = initial_annual_cost / (rule / 100.0)

    # scenario_builder が自動的に国民年金保険料 (-21.5) を加算するため、
    # CurveSpend には残りの額を設定する。
    initial_annual_cost_wo_premium = base_spend_annual * spend_mult

    new_lifeplan = replace(
        exp_setup.lifeplan,
        pension_start_age=pension_start,
        base_spend=CurveSpend(
            first_year_annual_amount=initial_annual_cost_wo_premium,
            spending_types=(
                SpendingType.SINGLE_2025_CONSUMPTION, SpendingType.
                UNEMPLOYED_SINGLE_2025_NON_CONSUMPTION_EXCLUDE_PENSION)))

    new_strategy = replace(baseline_strategy,
                           initial_money=float(init_money),
                           spend_adjustment=None)

    exp_setup.add_experiment(
        name=f"P{pension_start}_Mult_{spend_mult}_Rule_{rule}%",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=new_strategy)

  return exp_setup, N_SIM, combinations


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="40歳リタイア開始・95歳開始（94歳末）までの生存確率を分析するグリッドサーチスクリプト（単身世帯版）。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="optimal-pension",
                      help="実験設定 (optimal-pension)")
  args = parser.parse_args()

  # 設定
  exp_type = args.exp_type
  assert exp_type == "optimal-pension", f"Unsupported exp_type: {exp_type}"

  data_dir = "data/single_40yr/"
  csv_path = os.path.join(data_dir, f"{exp_type}.csv")
  os.makedirs(data_dir, exist_ok=True)

  # 初年度支出ベースライン (単身世帯用)
  base_spend_annual = get_annual_retired_spending_values([
      SpendingType.SINGLE_2025_CONSUMPTION,
      SpendingType.UNEMPLOYED_SINGLE_2025_NON_CONSUMPTION_EXCLUDE_PENSION
  ], START_AGE, 1)[0]
  # 国民年金保険料 (1人: 21.5万/年)
  pension_premium_annual = 21.5
  print(f"x1 支出: {base_spend_annual}")

  exp_setup: Setup
  n_sim_val: int
  combinations: List[Any]

  if exp_type == "optimal-pension":
    exp_setup, n_sim_val, combinations = get_optimal_pension_setup(
        base_spend_annual, pension_premium_annual)
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

    pension_start, spend_mult, rule = combo
    strat_name = "DynamicV1Rebalance"

    initial_annual_cost = base_spend_annual * spend_mult + pension_premium_annual
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
