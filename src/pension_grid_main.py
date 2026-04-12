"""
年金受給が資産寿命に与える影響をグリッドサーチで分析するスクリプト。

実験設定:
- 期間: 60年
- 試行回数: 5,000回
- 資産構成: オルカン 100% (7%, 15%)
- CPI: AR(12) 粘着性モデル (1970年〜)
- 年金CPI: マクロ経済スライド適用 (年率0.5%抑制、名目下限あり)
- グリッドパラメータ:
  - (初期資産, 年間支出): (5000, 200), (10000, 400), (20000, 800)
  - 開始年齢: 30, 40, 50, 60
  - 年金開始年齢: 60 (24%減額), 65
  - 年金月額(名目): 0, 5.6, 14.4
"""

import os
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, AssetConfigType, CpiAsset,
                                     MonthlyARLogNormal, SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, PensionConfig,
                                        generate_cashflows)

# 設定
DATA_DIR = "data/"
CSV_PATH = os.path.join(DATA_DIR, "pension_grid_comp.csv")

def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 60
  SEED = 42
  CPI_NAME = "Japan_CPI"
  PENSION_CPI_NAME = "Pension_CPI"

  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成
  # オルカン (7%, 15%)
  orukan = Asset(name="オルカン",
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15))
  
  # AR(12) CPI (1970年〜)
  initial_y = [
      0.0027039223324009146, 0.0035938942545892623, 0.0026869698208253877,
      -0.0008948546458437107, 0.0017889092427246362, 0.0017857147602345312,
      -0.0008924587830196112, 0.007117467768863955, 0.003539826705123987,
      -0.0017683470567420034, -0.0008853475567242145, -0.00621947806702042
  ]
  phis_1970 = [
      0.15268125115684014, -0.10485953085717699, 0.04007371599591021,
      0.01877889962124559, 0.12481104840559767, 0.07426982556030279,
      0.10457421059971438, 0.028405474126351145, 0.08655547241690399,
      -0.11318585572419704, 0.09698211923123926, 0.36329524916212186
  ]
  cpi_dist = MonthlyARLogNormal(c=0.00018532,
                                phis=phis_1970,
                                sigma_e=0.00446792,
                                initial_y=initial_y)
  base_cpi = CpiAsset(name=CPI_NAME, dist=cpi_dist)
  
  # 年金用CPI (マクロ経済スライド 0.5% 抑制)
  pension_cpi = SlideAdjustedCpiAsset(name=PENSION_CPI_NAME,
                                      base_cpi=CPI_NAME,
                                      slide_rate=0.005)

  configs: List[AssetConfigType] = [orukan, base_cpi, pension_cpi]
  
  print(f"価格推移を生成中... (試行回数: {N_SIM}, 期間: {YEARS}年)")
  monthly_prices = generate_monthly_asset_prices(configs,
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # グリッドパラメータ
  initial_money_annual_cost_list = [(5000, 200), (10000, 400), (20000, 800)]
  initial_age_list = [30, 40, 50, 60]
  pension_start_age_list = [60, 65]
  initial_pension_nominal_list = [0, 5.6, 14.4]

  all_combinations = list(product(
      initial_money_annual_cost_list,
      initial_age_list,
      pension_start_age_list,
      initial_pension_nominal_list
  ))

  results: List[Dict[str, Any]] = []

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")
  for i, ((init_money, annual_cost), init_age, start_age, p_nominal) in enumerate(all_combinations):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    # 年金額の計算 (60歳開始なら24%減額)
    # p_nominal は月額(万円)
    actual_pension_monthly = p_nominal
    if start_age == 60:
      actual_pension_monthly = p_nominal * 0.76
    
    # 受給開始月 (0始まり)
    start_month = max(0, (start_age - init_age) * 12)
    
    # キャッシュフロー生成
    cf_configs: List[CashflowConfig] = []
    pension_source_name = None
    if actual_pension_monthly > 0:
      pension_source_name = f"Pension_{i}"
      cf_configs.append(PensionConfig(name=pension_source_name,
                                      amount=actual_pension_monthly,
                                      start_month=start_month,
                                      cpi_name=PENSION_CPI_NAME))
    
    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略
    strategy = Strategy(
        name=f"Pattern_{i}",
        initial_money=float(init_money),
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={"オルカン": 1.0},
        annual_cost=float(annual_cost),
        inflation_rate=CPI_NAME,
        selling_priority=["オルカン"],
        extra_cashflow_sources=[pension_source_name] if pension_source_name else []
    )

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 生存確率の記録
    row = {
        "initial_money": init_money,
        "initial_annual_cost": annual_cost,
        "initial_age": init_age,
        "pension_start_age": start_age,
        "initial_pension_nominal": p_nominal
    }
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row[str(year)] = survival_rate
    
    results.append(row)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {CSV_PATH} に保存しました。")

if __name__ == "__main__":
  main()
