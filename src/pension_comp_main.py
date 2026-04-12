"""
年金受給が資産寿命に与える影響を分析するスクリプト。

実験設定:
- 40歳でリタイア
- 初期資産: 1億円 (10,000万円)
- 年間支出: 400万円 (物価連動)
- 資産構成: オルカン 100%
- 年金受給: 65歳 (25年後 = 300ヶ月後) から開始
- 比較ケース:
  1. 年金なし
  2. 老齢基礎年金のみ (月約6.8万円 = 年約81.6万円)
  3. 老齢基礎年金＋厚生年金 (月約15万円 = 年約180万円)
"""

import os
from typing import Dict, List

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, PensionConfig,
                                        generate_cashflows)
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)

# 出力先ディレクトリ
IMG_DIR = "docs/imgs/pension/"
DATA_DIR = "docs/data/pension/"


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 50  # 40歳から90歳まで
  SEED = 42
  INITIAL_MONEY = 10000.0  # 1億円
  ANNUAL_COST = 400.0  # 400万円
  CPI_NAME = "Japan_CPI"

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成 (オルカン 7%, 15% + CPI 1.77%)
  orukan = Asset(name="オルカン",
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15))
  cpi = CpiAsset(name=CPI_NAME,
                 dist=YearlyLogNormalArithmetic(mu=0.0177, sigma=0.0))
  
  monthly_prices = generate_monthly_asset_prices(configs=[orukan, cpi],
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # 2. キャッシュフロー生成 (25年後から開始、物価連動)
  cf_configs: List[CashflowConfig] = [
      PensionConfig(name="Basic_Pension",
                    amount=6.8,
                    start_month=300,
                    cpi_name=CPI_NAME),
      PensionConfig(name="Full_Pension",
                    amount=15.0,
                    start_month=300,
                    cpi_name=CPI_NAME),
  ]
  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=N_SIM,
                                         n_months=YEARS * 12)

  # 3. 戦略定義
  strategies = [
      Strategy(name="年金なし",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"]),
      Strategy(name="老齢基礎年金のみ (月6.8万)",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"],
               extra_cashflow_sources=["Basic_Pension"]),
      Strategy(name="基礎＋厚生年金 (月15万)",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"],
               extra_cashflow_sources=["Full_Pension"]),
  ]

  # 4. シミュレーション実行
  print("年金影響シミュレーションを実行中...")
  results = {}
  for strategy in strategies:
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)
    results[strategy.name] = res

  # 5. 可視化と保存
  formatted_df, _ = create_styled_summary(results,
                                          bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(DATA_DIR, "result.md"), "w", encoding="utf-8") as f:
    f.write(formatted_df.to_markdown())

  _, chart = create_survival_probability_chart(results, max_years=YEARS)
  chart.save(os.path.join(IMG_DIR, "survival.svg"))

  print(f"完了。結果を {DATA_DIR} と {IMG_DIR} に保存しました。")


if __name__ == "__main__":
  main()
