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
from typing import Dict, List, Optional, Union, cast

import numpy as np

from src.core import (ExtraCashflowMultiplierFn, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowConfig,
                                        CashflowRule, CashflowType,
                                        PensionConfig, generate_cashflows)
from src.lib.dynamic_rebalance import calculate_optimal_strategy
from src.lib.visualize import create_styled_summary, visualize_and_save

# 出力先ディレクトリ
IMG_DIR = "docs/imgs/side_fire/"
DATA_DIR = "docs/data/side_fire/"
TEMP_DIR = "temp/side_fire/"


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 50
  SEED = 42
  INITIAL_MONEY = 10000.0  # 1億円
  ANNUAL_COST = 400.0  # 400万円
  CPI_NAME = "Japan_CPI"
  ORUKAN_NAME = "オルカン"
  USDJPY_NAME = "USDJPY"
  RISK_FREE_NAME = "無リスク資産"

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)
  os.makedirs(TEMP_DIR, exist_ok=True)

  # 1. アセット生成
  # オルカン: 期待リターン7%, リスク15%, 信託報酬 0.05775%
  orukan = Asset(name=ORUKAN_NAME,
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
                 trust_fee=0.0005775,
                 forex=USDJPY_NAME)
  # 為替リスク: USDJPY (期待リターン0%, リスク10.53%)
  usdjpy = ForexAsset(name=USDJPY_NAME,
                      dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  # インフレ率: 1.77%, sigma=0
  cpi = CpiAsset(name=CPI_NAME,
                 dist=YearlyLogNormalArithmetic(mu=0.0177, sigma=0.0))

  monthly_prices = generate_monthly_asset_prices(configs=[orukan, usdjpy, cpi],
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  risk_free_asset = ZeroRiskAsset(name=RISK_FREE_NAME, yield_rate=0.04)

  def dynamic_rebalance_fn(total_net, cur_ann_spend, rem_years, post_tax_net):
    # s_rate = 年間支出額 / 純資産
    s_rate = cur_ann_spend / np.maximum(total_net, 1.0)
    # rem_years が極端に小さい場合の不自然な挙動を防ぐため、1.0年を下限とする
    safe_rem_years = np.maximum(rem_years, 1.0)
    orukan_ratio = calculate_optimal_strategy(s_rate, safe_rem_years)
    return {ORUKAN_NAME: orukan_ratio, RISK_FREE_NAME: 1.0 - orukan_ratio}

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

  spend_config = BaseSpendConfig(name="生活費",
                                 amount=ANNUAL_COST,
                                 cpi_name=CPI_NAME)

  exp1_cf_configs: List[CashflowConfig] = [spend_config]
  for label, monthly_amount in exp1_income_levels.items():
    if monthly_amount > 0:
      exp1_cf_configs.append(
          PensionConfig(name=f"Income_{label}",
                        amount=monthly_amount,
                        start_month=0,
                        end_month=5 * 12,
                        cpi_name=CPI_NAME))

  exp1_monthly_cashflows = generate_cashflows(exp1_cf_configs,
                                              monthly_prices,
                                              n_sim=N_SIM,
                                              n_months=YEARS * 12)

  exp1_strategies = []
  for label in exp1_income_levels.keys():
    rules = [
        CashflowRule(source_name=spend_config.name,
                     cashflow_type=CashflowType.REGULAR)
    ]
    if label != "なし":
      rules.append(
          CashflowRule(source_name=f"Income_{label}",
                       cashflow_type=CashflowType.EXTRAORDINARY))

    # Fixed 100% Orukan (Exp-1-A)
    exp1_strategies.append(
        Strategy(
            name=f"固定+{label}",
            initial_money=INITIAL_MONEY,
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio={ORUKAN_NAME: 1.0},
            cashflow_rules=rules,
            selling_priority=[ORUKAN_NAME],
            rebalance_interval=1,
        ))

    # Dynamic Rebalance (Exp-1-B)
    exp1_strategies.append(
        Strategy(
            name=f"ダイナ+{label}",
            initial_money=INITIAL_MONEY,
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio={
                ORUKAN_NAME: 1.0,
                risk_free_asset: 0.0
            },
            cashflow_rules=rules,
            selling_priority=[RISK_FREE_NAME, ORUKAN_NAME],
            rebalance_interval=12,
            dynamic_rebalance_fn=dynamic_rebalance_fn,
        ))

  exp1_results = {}
  for s in exp1_strategies:
    exp1_results[s.name] = simulate_strategy(
        s, monthly_prices, monthly_cashflows=exp1_monthly_cashflows)

  # サマリー保存
  # 特定の形式のテーブルを作成 (Exp 1)
  def get_survival_rate(res, years):
    return f"{np.mean(res.sustained_months >= years * 12) * 100.0:.1f}%"

  exp1_table = "| シナリオ | 戦略 | 20年生存確率 | 30年生存確率 | 50年生存確率 |\n"
  exp1_table += "| :--- | :--- | :--- | :--- | :--- |\n"
  for label in exp1_income_levels.keys():
    fixed_name = f"固定+{label}"
    dyna_name = f"ダイナ+{label}"
    fixed_res = exp1_results[fixed_name]
    dyna_res = exp1_results[dyna_name]
    display_label = f"{label} (収入なし)" if label == "なし" else label
    exp1_table += f"| **{display_label}** | オルカン100% | {get_survival_rate(fixed_res, 20)} | {get_survival_rate(fixed_res, 30)} | {get_survival_rate(fixed_res, 50)} |\n"
    exp1_table += f"| | ダイナミックリバランス | {get_survival_rate(dyna_res, 20)} | {get_survival_rate(dyna_res, 30)} | {get_survival_rate(dyna_res, 50)} |\n"

  with open(os.path.join(DATA_DIR, "exp1.md"), "w", encoding="utf-8") as f:
    f.write(exp1_table)

  visualize_and_save(exp1_results,
                     os.path.join(TEMP_DIR, "exp1_temp.html"),
                     distribution_image_file=os.path.join(
                         IMG_DIR, "exp1_distribution.svg"),
                     survival_image_file=os.path.join(IMG_DIR,
                                                      "exp1_survival.svg"),
                     title="Exp1: Income Level relative to Expenses",
                     open_browser=False)

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

  exp2_cf_configs: List[CashflowConfig] = [spend_config]
  for label, cfg in exp2_cases.items():
    duration_val = int(cfg["duration"])
    amount_val = float(cfg["amount"])
    exp2_cf_configs.append(
        PensionConfig(name=f"Income_{label}",
                      amount=amount_val,
                      start_month=0,
                      end_month=duration_val,
                      cpi_name=CPI_NAME))

  exp2_monthly_cashflows = generate_cashflows(exp2_cf_configs,
                                              monthly_prices,
                                              n_sim=N_SIM,
                                              n_months=YEARS * 12)

  exp2_strategies = []
  for label in exp2_cases.keys():
    rules = [
        CashflowRule(source_name=spend_config.name,
                     cashflow_type=CashflowType.REGULAR),
        CashflowRule(source_name=f"Income_{label}",
                     cashflow_type=CashflowType.EXTRAORDINARY)
    ]

    # Fixed 100% Orukan
    exp2_strategies.append(
        Strategy(
            name=f"固定+  {label}",
            initial_money=INITIAL_MONEY,
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio={ORUKAN_NAME: 1.0},
            cashflow_rules=rules,
            selling_priority=[ORUKAN_NAME],
            rebalance_interval=1,
        ))

    # Dynamic Rebalance
    exp2_strategies.append(
        Strategy(
            name=f"ダイナ+  {label}",
            initial_money=INITIAL_MONEY,
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio={
                ORUKAN_NAME: 1.0,
                risk_free_asset: 0.0
            },
            cashflow_rules=rules,
            selling_priority=[RISK_FREE_NAME, ORUKAN_NAME],
            rebalance_interval=12,
            dynamic_rebalance_fn=dynamic_rebalance_fn,
        ))

  exp2_results = {}
  for s in exp2_strategies:
    exp2_results[s.name] = simulate_strategy(
        s, monthly_prices, monthly_cashflows=exp2_monthly_cashflows)

  # サマリー保存
  # 特定の形式のテーブルを作成 (Exp 2)
  exp2_table = "| シナリオ | 戦略 | 20年生存確率 | 50年生存確率 | 50年後の中央値資産 |\n"
  exp2_table += "| :--- | :--- | :--- | :--- | :--- |\n"
  for label in exp2_cases.keys():
    fixed_name = f"固定+  {label}"
    dyna_name = f"ダイナ+  {label}"
    fixed_res = exp2_results[fixed_name]
    dyna_res = exp2_results[dyna_name]

    def get_survival_rate(res, years):
      return f"{np.mean(res.sustained_months >= years * 12) * 100.0:.1f}%"

    def get_median_asset(res):
      return f"{np.median(res.net_values) / 10000.0:.1f}億円"

    exp2_table += f"| **{label}** | オルカン100% | {get_survival_rate(fixed_res, 20)} | {get_survival_rate(fixed_res, 50)} | {get_median_asset(fixed_res)} |\n"
    exp2_table += f"| | ダイナミックリバランス | {get_survival_rate(dyna_res, 20)} | {get_survival_rate(dyna_res, 50)} | {get_median_asset(dyna_res)} |\n"

  with open(os.path.join(DATA_DIR, "exp2.md"), "w", encoding="utf-8") as f:
    f.write(exp2_table)

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
