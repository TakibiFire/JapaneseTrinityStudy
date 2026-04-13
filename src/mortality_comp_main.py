"""
死亡率を考慮した資産寿命（生存中の破産確率）を分析するスクリプト。

実験設定:
- 60歳でリタイア
- 初期資産: 5000万円
- 年間支出: 300万円 (物価連動)
- 資産構成: オルカン 100%
- シミュレーション期間: 40年 (60歳から100歳まで)
- 成功判定の定義:
  1. 破産せずにシミュレーション期間（100歳まで）を終える
  2. シミュレーション期間中に死亡する (死亡＝成功)
  ※ 本スクリプトでは、死亡時に巨大な収入を発生させることで、資産がマイナスになるのを防ぎ「成功」としてカウントする。
"""

import os
from typing import Dict, List

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, MortalityConfig,
                                        generate_cashflows)
from src.lib.life_table import MALE_MORTALITY_RATES
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)

# 出力先ディレクトリ
IMG_DIR = "docs/imgs/mortality/"
DATA_DIR = "docs/data/mortality/"


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 40
  SEED = 42
  INITIAL_MONEY = 5000.0
  ANNUAL_COST = 300.0
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
  # 死亡時に10億円の収入を発生させる
  cf_configs: List[CashflowConfig] = [
      MortalityConfig(name="Mortality",
                      mortality_rates=MALE_MORTALITY_RATES,
                      initial_age=60,
                      payout=100000.0) # 10億円
  ]
  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=N_SIM,
                                         n_months=YEARS * 12)

  # 3. 戦略定義
  strategies = [
      Strategy(name="100歳まで生存と仮定",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"]),
      Strategy(name="男性の平均的な死亡率を考慮",
               initial_money=INITIAL_MONEY,
               initial_loan=0.0,
               yearly_loan_interest=0.0,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=ANNUAL_COST,
               inflation_rate=CPI_NAME,
               selling_priority=["オルカン"],
               extra_cashflow_sources={"Mortality": None}),
  ]

  # 4. シミュレーション実行
  print("死亡率考慮シミュレーションを実行中...")
  results = {}
  for strategy in strategies:
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)
    results[strategy.name] = res

  # 5. 可視化と保存
  formatted_df, _ = create_styled_summary(results,
                                          bankruptcy_years=[10, 20, 30, 40])
  with open(os.path.join(DATA_DIR, "result.md"), "w", encoding="utf-8") as f:
    f.write(formatted_df.to_markdown())

  _, chart = create_survival_probability_chart(results, max_years=YEARS)
  chart.save(os.path.join(IMG_DIR, "survival.svg"))

  print(f"完了。結果を {DATA_DIR} と {IMG_DIR} に保存しました。")


if __name__ == "__main__":
  main()
