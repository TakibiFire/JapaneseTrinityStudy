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
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.dynamic_rebalance import calculate_safe_target_ratio
from src.lib.scenario_builder import (ConstantSpend, CpiType,
                                      DynamicV1Adjustment, DynamicV1Rebalance,
                                      FxType, Lifeplan, PensionStatus,
                                      PredefinedStock, PredefinedZeroRisk,
                                      Setup, StrategySpec, WorldConfig,
                                      create_experiment_setup)


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="60歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="P60-D1",
                      help="実験設定 (P-D-RANGE or P60-D1)")
  args = parser.parse_args()

  # 設定
  exp_type = args.exp_type
  assert exp_type in (
      # 年金受け取りの受給タイミングとDynamicSpendingをするかどうかの最適組み合わせを求める。
      "P-D-RANGE",
      # 年金受け取りの受給タイミング=60, DynamicSpending=ON が確定。
      # より詳細なパラメータで分析を行う。
      "P60-D1",
  ), f"Unsupported exp_type: {exp_type}"

  data_dir = "data/all_60yr/"
  csv_path = os.path.join(data_dir, f"{exp_type}.csv")

  # 共通設定
  YEARS = 35  # 60歳から95歳まで
  START_AGE = 60
  SEED = 42

  if exp_type == "P-D-RANGE":
    spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.5, 3.0]
    spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    N_SIM = 1000
    pension_start_ages = [60, 65]
    use_dynamic_spending_list = [False, True]
  elif exp_type == "P60-D1":
    spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
    spending_rules = [2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0]
    N_SIM = 3000
    pension_start_ages = [60]
    use_dynamic_spending_list = [True]
  else:
    raise KeyError(f"Unsupported {exp_type}")

  os.makedirs(data_dir, exist_ok=True)

  # 初年度支出ベースライン (60歳の出費平均45万 * 12か月)
  BASE_SPEND_ANNUAL = 540.0

  # セーフティな支出率 (DynamicSpending用)
  target_ratio = calculate_safe_target_ratio(YEARS)

  # 1. ベースライン設定
  baseline_world = WorldConfig(n_sim=N_SIM,
                               n_years=YEARS,
                               start_age=START_AGE,
                               seed=SEED,
                               cpi_type=CpiType.JAPAN_AR12,
                               fx_type=FxType.USDJPY)

  baseline_lifeplan = Lifeplan(
      base_spend=ConstantSpend(annual_amount=BASE_SPEND_ANNUAL),
      retirement_start_age=60,
      pension_status=PensionStatus.FULL,
      pension_start_age=65,
      household_size=1)

  baseline_strategy = StrategySpec(
      initial_money=10000.0,
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
  all_combinations = list(
      product(pension_start_ages, spend_multipliers, spending_rules,
              use_dynamic_spending_list))

  for (pension_start, spend_mult, rule,
       use_dyn_spend) in all_combinations:
    # 既存のロジックに従った初期資産と初年度支出の計算
    initial_annual_cost = BASE_SPEND_ANNUAL * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    new_lifeplan = replace(baseline_lifeplan,
                           pension_start_age=pension_start,
                           base_spend=ConstantSpend(
                               annual_amount=initial_annual_cost))

    new_strategy = replace(
        baseline_strategy,
        initial_money=float(init_money),
        spend_adjustment=DynamicV1Adjustment(
            target_ratio=rule / 100.0, upper_limit=0.03, lower_limit=0.0)
        if use_dyn_spend else None)

    exp_setup.add_experiment(
        name=
        f"P{pension_start}_Mult_{spend_mult}_Rule_{rule}%_Dyn_{use_dyn_spend}",
        overwrite_lifeplan=new_lifeplan,
        overwrite_strategy=new_strategy)

  # 3. コンパイルとシミュレーション
  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")
  compiled_experiments = create_experiment_setup(exp_setup)

  results: List[Dict[str, Any]] = []

  # ベースラインをスキップし、オリジナルの組み合わせとジップして結果を処理
  for i, (exp, (pension_start, spend_mult, rule,
            use_dyn_spend)) in enumerate(zip(compiled_experiments[1:], all_combinations)):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)

    initial_annual_cost = BASE_SPEND_ANNUAL * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    base_row: Dict[str, Union[float, int, str]] = {
        "pension_start_age": pension_start,
        "spend_multiplier": spend_mult,
        "spending_rule": rule,
        "use_dynamic_spending": 1 if use_dyn_spend else 0,
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
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
