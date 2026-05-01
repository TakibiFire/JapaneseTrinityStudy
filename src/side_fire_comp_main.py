"""
Side FIRE (労働収入) が資産寿命に与える影響を分析するスクリプト。

実験設定:
- 初期資産: 1億円
- 投資先: オルカン100% (期待リターン7%, リスク15%, 信託報酬 0.05775%)
- 為替リスク: USDJPY (期待リターン0%, リスク10.53%)
- インフレ率: 1.77%, sigma=0
- 初期出費額: 400万円
- 税率: 20.315%
- シミュレーション期間: 50年
- 試行回数: 5000回
"""

import os
from dataclasses import replace

import numpy as np

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType,
                                      DynamicV1Rebalance, Lifeplan,
                                      PensionStatus, PredefinedStock,
                                      PredefinedZeroRisk, Setup, StrategySpec,
                                      WorldConfig, create_experiment_setup)
from src.lib.visualize import visualize_and_save

# 出力先ディレクトリ
IMG_DIR = "docs/imgs/side_fire/"
DATA_DIR = "docs/data/side_fire/"
TEMP_DIR = "temp/side_fire/"


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 50
  START_AGE = 50
  INITIAL_MONEY = 10000.0  # 1億円
  ANNUAL_COST = 400.0  # 400万円

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)
  os.makedirs(TEMP_DIR, exist_ok=True)

  # 1. シナリオビルダーの準備
  world = WorldConfig(n_sim=N_SIM,
                      n_years=YEARS,
                      start_age=START_AGE,
                      cpi_type=CpiType.FIXED_1_77)
  baseline_lifeplan = Lifeplan(base_spend=ConstantSpend(
      annual_amount=ANNUAL_COST),
                               retirement_start_age=START_AGE,
                               pension_status=PensionStatus.NONE)
  baseline_strategy = StrategySpec(
      initial_money=INITIAL_MONEY,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0), ),
      selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN, ))

  exp_setup = Setup(name="baseline",
                    world=world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  # -------------------------------------------------------------------------
  # Exp 1: Income Level relative to Expenses
  # -------------------------------------------------------------------------
  print("Experiment 1 実行中...")

  # 収入パターンの作成
  # 25% of 400 = 100/year = 8.333/month
  # 50% of 400 = 200/year = 16.666/month
  # 75% of 400 = 300/year = 25.0/month
  exp1_income_levels = {
      "なし": 0.0,
      "25%": 100.0 / 12.0,
      "50%": 200.0 / 12.0,
      "75%": 300.0 / 12.0
  }

  for label, monthly_amount in exp1_income_levels.items():
    new_lp = replace(baseline_lifeplan,
                     side_fire_income_monthly=monthly_amount,
                     side_fire_duration_months=5 * 12)

    # 固定100%オルカン
    exp_setup.add_experiment(name=f"固定+{label}", overwrite_lifeplan=new_lp)

    # ダイナミックリバランス
    new_strategy = replace(
        baseline_strategy,
        initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),
                             (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
        selling_priority=(PredefinedZeroRisk.ZERO_RISK_4PCT,
                          PredefinedStock.SIMPLE_7_15_ORUKAN),
        rebalance=DynamicV1Rebalance(
            risky_asset=PredefinedStock.SIMPLE_7_15_ORUKAN,
            zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT))
    exp_setup.add_experiment(name=f"ダイナ+{label}",
                             overwrite_lifeplan=new_lp,
                             overwrite_strategy=new_strategy)

  # -------------------------------------------------------------------------
  # Exp 2: Duration vs. Amount (Total 2000M)
  # -------------------------------------------------------------------------
  print("Experiment 2 実行中...")

  # Total 2000-man
  exp2_cases = {
      "一括": {
          "amount": 2000.0,
          "duration": 1
      },
      "400万×5年": {
          "amount": 400.0 / 12.0,
          "duration": 5 * 12
      },
      "200万×10年": {
          "amount": 200.0 / 12.0,
          "duration": 10 * 12
      },
      "100万×20年": {
          "amount": 100.0 / 12.0,
          "duration": 20 * 12
      }
  }

  for label, cfg in exp2_cases.items():
    new_lp = replace(baseline_lifeplan,
                     side_fire_income_monthly=float(cfg["amount"]),
                     side_fire_duration_months=int(cfg["duration"]))

    # 固定100%オルカン
    exp_setup.add_experiment(name=f"固定+  {label}", overwrite_lifeplan=new_lp)

    # ダイナミックリバランス
    new_strategy = replace(
        baseline_strategy,
        initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),
                             (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
        selling_priority=(PredefinedZeroRisk.ZERO_RISK_4PCT,
                          PredefinedStock.SIMPLE_7_15_ORUKAN),
        rebalance=DynamicV1Rebalance(
            risky_asset=PredefinedStock.SIMPLE_7_15_ORUKAN,
            zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT))
    exp_setup.add_experiment(name=f"ダイナ+  {label}",
                             overwrite_lifeplan=new_lp,
                             overwrite_strategy=new_strategy)

  # コンパイル
  compiled_experiments = create_experiment_setup(exp_setup)

  # 実行と結果収集
  results = {}
  for exp in compiled_experiments:
    if exp.name == "baseline":
      continue
    print(exp.name)
    print(list(exp.monthly_prices.keys()))
    results[exp.name] = simulate_strategy(exp.strategy, exp.monthly_prices,
                                          exp.monthly_cashflows)

  # -------------------------------------------------------------------------
  # 結果の保存と可視化
  # -------------------------------------------------------------------------

  # Exp 1 テーブル作成
  def get_survival_rate(res, years):
    return f"{np.mean(res.sustained_months >= years * 12) * 100.0:.1f}%"

  exp1_table = "| シナリオ | 戦略 | 20年生存確率 | 30年生存確率 | 50年生存確率 |\n"
  exp1_table += "| :--- | :--- | :--- | :--- | :--- |\n"
  for label in exp1_income_levels.keys():
    fixed_name = f"固定+{label}"
    dyna_name = f"ダイナ+{label}"
    fixed_res = results[fixed_name]
    dyna_res = results[dyna_name]
    display_label = f"{label} (収入なし)" if label == "なし" else label
    exp1_table += f"| **{display_label}** | オルカン100% | {get_survival_rate(fixed_res, 20)} | {get_survival_rate(fixed_res, 30)} | {get_survival_rate(fixed_res, 50)} |\n"
    exp1_table += f"| | ダイナミックリバランス | {get_survival_rate(dyna_res, 20)} | {get_survival_rate(dyna_res, 30)} | {get_survival_rate(dyna_res, 50)} |\n"

  with open(os.path.join(DATA_DIR, "exp1.md"), "w", encoding="utf-8") as f:
    f.write(exp1_table)

  exp1_results = {k: v for k, v in results.items() if k.startswith(("固定+", "ダイナ+")) and not k.startswith(("固定+  ", "ダイナ+  "))}
  visualize_and_save(exp1_results,
                     os.path.join(TEMP_DIR, "exp1_temp.html"),
                     distribution_image_file=os.path.join(
                         IMG_DIR, "exp1_distribution.svg"),
                     survival_image_file=os.path.join(IMG_DIR,
                                                      "exp1_survival.svg"),
                     title="Exp1: Income Level relative to Expenses",
                     open_browser=False)

  # Exp 2 テーブル作成
  exp2_table = "| シナリオ | 戦略 | 20年生存確率 | 50年生存確率 | 50年後の中央値資産 |\n"
  exp2_table += "| :--- | :--- | :--- | :--- | :--- |\n"
  for label in exp2_cases.keys():
    fixed_name = f"固定+  {label}"
    dyna_name = f"ダイナ+  {label}"
    fixed_res = results[fixed_name]
    dyna_res = results[dyna_name]

    def get_median_asset(res):
      return f"{np.median(res.net_values) / 10000.0:.1f}億円"

    exp2_table += f"| **{label}** | オルカン100% | {get_survival_rate(fixed_res, 20)} | {get_survival_rate(fixed_res, 50)} | {get_median_asset(fixed_res)} |\n"
    exp2_table += f"| | ダイナミックリバランス | {get_survival_rate(dyna_res, 20)} | {get_survival_rate(dyna_res, 50)} | {get_median_asset(dyna_res)} |\n"

  with open(os.path.join(DATA_DIR, "exp2.md"), "w", encoding="utf-8") as f:
    f.write(exp2_table)

  exp2_results = {k: v for k, v in results.items() if k.startswith(("固定+  ", "ダイナ+  "))}
  visualize_and_save(exp2_results,
                     os.path.join(TEMP_DIR, "exp2_temp.html"),
                     distribution_image_file=os.path.join(
                         IMG_DIR, "exp2_distribution.svg"),
                     survival_image_file=os.path.join(IMG_DIR,
                                                      "exp2_survival.svg"),
                     title="Exp2: Duration vs. Amount (Total 2000M)",
                     open_browser=False)

  print(f"完了。結果を {DATA_DIR} と {IMG_DIR} に保存しました。")


if __name__ == "__main__":
  main()
