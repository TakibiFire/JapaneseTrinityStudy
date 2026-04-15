"""
条件付き労働 (Sequence of Return Risk Mitigation) が資産寿命に与える影響をグリッドサーチで分析するスクリプト。

実験設定:
- 初期資産: 1億円
- 投資先: オルカン100% (期待リターン7%, リスク15%, 信託報酬 0.05775%)
- 為替リスク: USDJPY (期待リターン0%, リスク10.53%)
- CPI: AR(12) 粘着性モデル (1970年〜)
- 初期出費額: 400万円
- 税率: 20.315%
- シミュレーション期間: 50年
- 試行回数: 5000回

条件付き労働のルール:
- 年支出 / 総資産 が X 以上になった場合に1年間働く。
- 働くのは最大 Y 年目まで。
- 労働により Z% の 400万円 (CPI調整済み) を得る。

グリッドパラメータ:
- X (閾値): 4%, 5%, 6%, 8%, 10%, 12%
- Y (期間上限): 10, 20, 30, 40, 50 年
- Z (労働収入): 25%, 50%, 75%, 100%
- 加えて、全く働かないケース (Baseline) を比較対象とする。
"""

import os
from itertools import product
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, AssetConfigType, CpiAsset,
                                     ForexAsset, MonthlyARLogNormal,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, CashflowRule,
                                        CashflowType, PensionConfig,
                                        generate_cashflows)

# 設定
DATA_DIR = "data/"
CSV_PATH = os.path.join(DATA_DIR, "side_fire_cond_grid.csv")


def create_conditional_work_multiplier(threshold_x: float, max_year_y: int):
  """
  条件付き労働の倍率関数を作成する。
  """

  def multiplier_fn(m: int, net_worth: np.ndarray,
                    prev_spending: np.ndarray) -> np.ndarray:
    # m: 経過月数 (12の倍数)
    # net_worth: 現在の純資産 (n_sim,)
    # prev_spending: 前年の年間支出額 (n_sim,)

    # 期間上限のチェック
    if m >= max_year_y * 12:
      return np.zeros_like(net_worth)

    # 閾値のチェック (年支出 / 総資産 > X)
    # net_worth が 0 以下の場合も考慮
    safe_net_worth = np.maximum(net_worth, 1.0)
    condition = (prev_spending / safe_net_worth) >= threshold_x

    return condition.astype(np.float64)

  return multiplier_fn


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

  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成
  # オルカン: 期待リターン7%, リスク15%, 信託報酬 0.05775%
  orukan = Asset(name=ORUKAN_NAME,
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
                 trust_fee=0.0005775,
                 forex=USDJPY_NAME)
  # 為替リスク: USDJPY (期待リターン0%, リスク10.53%)
  usdjpy = ForexAsset(name=USDJPY_NAME,
                      dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  
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
  cpi = CpiAsset(name=CPI_NAME, dist=cpi_dist)

  print(f"価格推移を生成中... (試行回数: {N_SIM}, 期間: {YEARS}年)")
  configs: List[AssetConfigType] = [orukan, usdjpy, cpi]
  monthly_prices = generate_monthly_asset_prices(configs=configs,
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # グリッドパラメータ
  threshold_x_list = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
  max_year_y_list = [10, 20, 30, 40, 50]
  income_percent_z_list = [0.25, 0.50, 0.75, 1.0]

  all_combinations = list(
      product(threshold_x_list, max_year_y_list,
              income_percent_z_list))
  # Baseline (閾値101% = 決して働かない) を追加
  all_combinations.append((1.01, 0, 0.0))

  results: List[Dict[str, Any]] = []

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")
  for i, (x, y, z) in enumerate(all_combinations):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    scenario_name = "Baseline" if x > 1.0 else f"X={x:.0%}, Y={y}, Z={z:.0%}"

    # 労働収入の設定
    # Z% of 400万 = 400 * Z / 12 per month
    income_monthly = (ANNUAL_COST * z) / 12.0
    cf_configs: List[CashflowConfig] = [
        PensionConfig(
            name="ConditionalWork",
            amount=income_monthly,
            start_month=0,
            end_month=YEARS * 12,  # 倍率関数側で制御
            cpi_name=CPI_NAME)
    ]
    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略の設定
    multiplier_fn = create_conditional_work_multiplier(x, y)
    strategy = Strategy(
        name=f"X={x}_Y={y}_Z={z}",
        initial_money=INITIAL_MONEY,
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={ORUKAN_NAME: 1.0},
        annual_cost=ANNUAL_COST,
        inflation_rate=CPI_NAME,
        selling_priority=[ORUKAN_NAME],
        cashflow_rules=[
            CashflowRule(source_name="ConditionalWork",
                         cashflow_type=CashflowType.ISOLATED,
                         multiplier_fn=multiplier_fn)
        ])

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 結果の記録
    row = {
        "threshold_x": x,
        "max_year_y": y,
        "income_percent_z": z,
        "scenario": scenario_name
    }
    for year in range(1, YEARS + 1):
      survival_rate = np.mean(res.sustained_months >= year * 12)
      row[str(year)] = survival_rate
    results.append(row)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {CSV_PATH} に保存しました。")


if __name__ == "__main__":
  main()
