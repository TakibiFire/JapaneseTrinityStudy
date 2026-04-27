"""
年齢ごとの支出変動を考慮した場合の生存確率への影響を比較するシミュレーション。
"""

import os
import re
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core import Strategy, ZeroRiskAsset, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  max_years = 50
  seed = 42
  initial_money = 10000.0  # 万円
  tax_rate = 0.20315
  inflation_rate_val = 0.0177
  fee_acwi = 0.0005775

  # 共通アセット名
  cpi_name = "Japan_CPI_1.77pct"
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"

  # 1. 価格推移の生成
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name,
                 dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)),
      Asset(name=acwi_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=fee_acwi,
            forex=fx_name),
      CpiAsset(name=cpi_name,
               dist=YearlyLogNormalArithmetic(mu=inflation_rate_val, sigma=0.0))
  ]

  print(f"月次価格推移を生成中 (n_sim={n_sim}, years={max_years})...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=max_years * 12,
                                                       seed=seed)

  # 投資戦略の設定 (オルカン100%)
  ratio = 1.0
  base_annual_cost = 400.0

  results = {}
  print("各戦略のシミュレーションを実行中...")

  # 1. 支出一定 (400万円)
  spend_config_const = BaseSpendConfig(name="生活費",
                                       amount=base_annual_cost,
                                       cpi_name=cpi_name)
  cashflow_rules_const = [
      CashflowRule(source_name=spend_config_const.name,
                   cashflow_type=CashflowType.REGULAR)
  ]
  monthly_cashflows_const = generate_cashflows([spend_config_const],
                                               monthly_asset_prices, n_sim,
                                               max_years * 12)

  s_const = Strategy(name="1. 支出一定 (400万円)",
                     initial_money=initial_money,
                     initial_loan=0.0,
                     yearly_loan_interest=0.0,
                     initial_asset_ratio={acwi_name: ratio},
                     cashflow_rules=cashflow_rules_const,
                     tax_rate=tax_rate,
                     selling_priority=[acwi_name],
                     rebalance_interval=12)
  results[s_const.name] = simulate_strategy(
      s_const, monthly_asset_prices, monthly_cashflows=monthly_cashflows_const)

  start_ages = [30, 35, 40, 45, 50, 55, 60]
  for i, start_age in enumerate(start_ages):
    multipliers = get_retired_spending_multipliers(
        [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION], start_age,
        max_years)
    annual_costs = [base_annual_cost * m for m in multipliers]

    spend_config = BaseSpendConfig(name=f"生活費_{start_age}歳",
                                   amount=annual_costs,
                                   cpi_name=cpi_name)
    cashflow_rules = [
        CashflowRule(source_name=spend_config.name,
                     cashflow_type=CashflowType.REGULAR)
    ]
    monthly_cashflows = generate_cashflows([spend_config], monthly_asset_prices,
                                           n_sim, max_years * 12)

    strategy = Strategy(name=f"{i+2}. {start_age}歳",
                        initial_money=initial_money,
                        initial_loan=0.0,
                        yearly_loan_interest=0.0,
                        initial_asset_ratio={acwi_name: ratio},
                        cashflow_rules=cashflow_rules,
                        tax_rate=tax_rate,
                        selling_priority=[acwi_name],
                        rebalance_interval=12)
    results[strategy.name] = simulate_strategy(
        strategy, monthly_asset_prices, monthly_cashflows=monthly_cashflows)

  # 2. 可視化と保存
  img_dir = "docs/imgs/retired_spending"
  data_dir = "docs/data/retired_spending"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'survival.svg')
  distribution_image_file = os.path.join(img_dir, 'distribution.svg')
  html_file = 'temp/cost_per_age_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='年齢別支出の影響',
                     summary_title='年齢別支出の影響サマリー',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     open_browser=False)

  # 3. Markdownデータの出力
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[10, 20, 30, 40, 50])

  md_table = formatted_df.to_markdown(colalign=("left",) +
                                      ("right",) * len(formatted_df.columns))

  md_file = os.path.join(data_dir, 'result.md')
  with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_table)

  print(f"✅ {md_file} を作成しました。")
  print(f"✅ {survival_image_file} を作成しました。")
  print(f"✅ {distribution_image_file} を作成しました。")
  print(f"詳細な結果は {html_file} で確認できます。")


if __name__ == "__main__":
  main()
