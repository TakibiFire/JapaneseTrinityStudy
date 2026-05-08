"""
DP Experimental Approaches の評価を行うスクリプト。
4つのモデルと Winning Threshold の有無の組み合わせを評価します。
"""

import os
from dataclasses import replace
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (PredefinedStock, PredefinedZeroRisk,
                                      SpendAwareDPRebalance,
                                      create_experiment_setup)
from src.lib.world_setup import re60_pen70_95

# 共通設定
YEARS = 35
START_AGE = 60
SEED = 43
N_SIM_EVAL = 2000


def main():
  data_dir = "data/experiment_dp_grid/"
  csv_path = os.path.join(data_dir, "eval.csv")
  os.makedirs(data_dir, exist_ok=True)

  # 初年度支出ベースライン (60歳)
  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      60, 1)[0]

  # モデルのリスト
  model_files = [
      "re60_pen70_95_n1000", "re60_pen70_95_n2000",
      "re60_pen70_95_robust_n1000", "re60_pen70_95_robust_n2000"
  ]

  # 実験パラメータ
  spending_rules = [
      2.5, 2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0, 8.0
  ]
  win_threshold_options = [True, False]  # disable_win_threshold の値

  # ベースライン設定 (re60_pen70_95)
  exp_setup = re60_pen70_95(n_sim=N_SIM_EVAL, seed=SEED)
  exp_setup.name = "experimental_dp_eval"

  combinations = list(
      product(model_files, win_threshold_options, spending_rules))

  for (model_base, disable_win, rule) in combinations:
    # spending_rule に合わせて初期資産を調整
    init_money = base_spend_annual / (rule / 100.0)

    model_path = f"data/optimal_strategy_dp/experiments/{model_base}.json"

    # 戦略設定のみを上書きし、Lifeplan は baseline を維持
    spec = replace(exp_setup.strategy,
                   initial_money=float(init_money),
                   rebalance=SpendAwareDPRebalance(
                       risky_asset=PredefinedStock.ORUKAN_155,
                       zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                       model_name=model_path,
                       disable_win_threshold=disable_win))

    win_str = "win0" if disable_win else "win1"
    exp_setup.add_experiment(name=f"{model_base}_{win_str}_R{rule}",
                             overwrite_strategy=spec)

  # コンパイルとシミュレーション
  print(f"全 {len(combinations)} パターンのシミュレーションを実行中...")
  compiled_experiments = create_experiment_setup(exp_setup,
                                                 record_annual_spend=False)

  results: List[Dict[str, Any]] = []

  # ベースラインをスキップ
  for i, (exp, combo) in enumerate(zip(compiled_experiments[1:], combinations)):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(combinations)}")

    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)

    model_base, disable_win, rule = combo

    # 最終的な生存確率のみを記録 (95歳開始時点)
    bankrupt_count = (res.sustained_months < YEARS * 12).sum()
    survival_rate = 1.0 - (bankrupt_count / N_SIM_EVAL)

    results.append({
        "model": model_base,
        "disable_win_threshold": disable_win,
        "robust": "robust" in model_base,
        "n_sim_train": 1000 if "n1000" in model_base else 2000,
        "spending_rule": rule,
        "survival_rate": survival_rate
    })

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
  print(f"完了。結果を {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
