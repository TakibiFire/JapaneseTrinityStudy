"""
Side FIRE (労働収入) が資産寿命に与える影響を分析するスクリプト。

実験設定:
- 40歳でリタイア
- 初期資産: 5000万円 (5,000万円)  -- 少なめに設定
- 年間支出: 300万円 (物価連動)
- 資産構成: オルカン 100%
- シミュレーション期間: 50年
- Side FIRE収入:
  - 最初の10年間 (120ヶ月)、月10万円 (年120万円) の労働収入を得る。
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
IMG_DIR = "docs/imgs/side_fire/"
DATA_DIR = "docs/data/side_fire/"


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 50
  SEED = 42
  INITIAL_MONEY = 5000.0  # 5000万円
  ANNUAL_COST = 300.0  # 300万円
  CPI_NAME = "Japan_CPI"

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成
  orukan = Asset(name="オルカン",
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15))
  cpi = CpiAsset(name=CPI_NAME,
                 dist=YearlyLogNormalArithmetic(mu=0.0177, sigma=0.0))
  
  monthly_prices = generate_monthly_asset_prices(configs=[orukan, cpi],
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # 2. キャッシュフロー生成
  # 最初の120ヶ月だけ収入を得るために、
  # 0ヶ月目から+10万、120ヶ月目から-10万することで相殺する。
  cf_configs: List[CashflowConfig] = [
      PensionConfig(name="Work_Income", amount=10.0, start_month=0),
      PensionConfig(name="Work_Stop", amount=-10.0, start_month=120),
  ]
  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=N_SIM,
                                         n_months=YEARS * 12)

  # 3. 戦略定義
  strategies = [
      Strategy(name="完全FIRE (労働収入なし)",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"]),
      Strategy(name="Side FIRE (10年間 月10万円の労働収入)",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"],
               extra_cashflow_sources=["Work_Income", "Work_Stop"]),
  ]

  # 4. シミュレーション実行
  print("Side FIRE影響シミュレーションを実行中...")
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
