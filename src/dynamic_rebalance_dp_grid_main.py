"""
40歳からの55年間の資産運用・取り崩しシミュレーションを行い、
新旧のリバランス戦略（V1 vs V2/DP）を比較するグリッドサーチスクリプト。

実験設定:
- 期間: 55年 (40歳〜95歳)
- 試行回数: 5,000回
- 資産構成: FX, ACWI (fat tail), CPI, Pension CPI (slide_rate=0.005)
- 世帯設定: 1人世帯, 年金受給開始60歳 (H1_P60)
- 比較戦略:
    1. 固定最適比率 (Fixed Optimal Ratio)
    2. ダイナミック最適比率 (V1)
    3. Dynamic Rebalance DP (V2)
"""

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core import Strategy, ZeroRiskAsset, simulate_strategy
from src.lib.asset_generator import (AssetConfigType, DerivedAsset, ForexAsset,
                                     SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, CashflowRule,
                                        CashflowType, PensionConfig,
                                        generate_cashflows)
from src.lib.dynamic_rebalance import calculate_optimal_strategy
from src.lib.dynamic_rebalance_dp import DPOptimalStrategyPredictor
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers,
                                      get_retired_spending_values)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="DPベースの動的リバランス戦略とV1戦略を比較するグリッドシミュレーション")
  parser.add_argument("--exp_name",
                      type=str,
                      default="dp_comp",
                      help="実験名（出力ファイル名に使用）")
  args = parser.parse_args()

  # 設定
  EXP_NAME = args.exp_name
  YEARS = 55  # 40歳から95歳まで
  START_AGE = 40
  SEED = 42
  N_SIM = 5000
  DATA_DIR = "data/dynamic_rebalance_dp/"
  CSV_PATH = os.path.join(DATA_DIR, f"{EXP_NAME}.csv")
  MODELS_PATH = "data/optimal_strategy_v2_models.json"

  # 資産名
  CPI_NAME = "Japan_CPI"
  PENSION_CPI_NAME = "Pension_CPI"
  FX_NAME = "USDJPY_0_10.53"
  ZERO_RISK_NAME = "ゼロリスク資産"
  ORUKAN_NAME = "オルカン"

  # 定数
  TRUST_FEE = 0.0005775
  ZERO_RISK_YIELD = 0.04
  TAX_RATE = 0.20315
  CURRENT_YEAR = 2026
  MACRO_ECONOMIC_SLIDE_END_YEAR = 2057
  INFLATION_RATE = 0.0177  # V1計算用

  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成
  fx_asset = ForexAsset(name=FX_NAME,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  base_sp500 = get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y)
  base_acwi = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)
  orukan = DerivedAsset(name=ORUKAN_NAME,
                        base=base_acwi.name,
                        trust_fee=TRUST_FEE,
                        forex=FX_NAME)
  zr_asset_obj = ZeroRiskAsset(name=ZERO_RISK_NAME, yield_rate=ZERO_RISK_YIELD)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)
  pension_cpi = SlideAdjustedCpiAsset(
      name=PENSION_CPI_NAME,
      base_cpi=CPI_NAME,
      slide_rate=0.005,
      slide_end_month=(MACRO_ECONOMIC_SLIDE_END_YEAR - CURRENT_YEAR) * 12)

  configs: List[AssetConfigType] = [
      fx_asset, base_sp500, base_acwi, orukan, base_cpi, pension_cpi
  ]

  print(f"価格推移を生成中... (試行回数: {N_SIM}, 期間: {YEARS}年)")
  monthly_prices = generate_monthly_asset_prices(configs,
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # 2. キャッシュフローと支出の準備
  spending_types = [
      SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION
  ]
  spending_multipliers_by_age = get_retired_spending_multipliers(
      spending_types, start_age=START_AGE, num_years=YEARS, normalize=False)

  # ベースラインの支出額 (月額合計 -> 年額合計)
  base_spending_monthly = get_retired_spending_values(
      spending_types, target_ages=np.array([float(START_AGE)]))[0]
  # NOW: Use Japanese comment.
  # get_retired_spending_values returns monthly yen. Convert to annual man-yen.
  BASE_SPEND_ANNUAL_WO_PENSION = base_spending_monthly * 12.0 / 10000.0

  # 年金設定: 1人世帯, 年金開始60歳
  PENSION_START_AGE = 60
  PENSION_PREMIUM_ANNUAL = 20.4
  # 1人世帯・60歳開始時の年金合計額 (all_40yr_grid_main.py より)
  PENSION_TOTAL_ANNUAL = 99.4
  KISO_FULL_ANNUAL = 81.6
  REDUCTION_RATE = 0.76
  KISO_ANNUAL = KISO_FULL_ANNUAL * REDUCTION_RATE
  KOUSEI_ANNUAL = PENSION_TOTAL_ANNUAL - KISO_ANNUAL

  cf_configs: List[CashflowConfig] = []
  cf_rules: List[CashflowRule] = []

  # 保険料支払い: 40歳から60歳まで (20年間 = 240ヶ月)
  cf_configs.append(
      PensionConfig(name="Pension_Premium",
                    amount=-PENSION_PREMIUM_ANNUAL / 12.0,
                    start_month=0,
                    end_month=240,
                    cpi_name=CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Premium",
                   cashflow_type=CashflowType.REGULAR))

  # 年金受給: 60歳から (20年後 = 240ヶ月目から)
  receipt_start_month = (PENSION_START_AGE - START_AGE) * 12
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kousei",
                    amount=KOUSEI_ANNUAL / 12.0,
                    start_month=receipt_start_month,
                    cpi_name=CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kousei",
                   cashflow_type=CashflowType.REGULAR))

  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kiso",
                    amount=KISO_ANNUAL / 12.0,
                    start_month=receipt_start_month,
                    cpi_name=PENSION_CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kiso",
                   cashflow_type=CashflowType.REGULAR))

  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=N_SIM,
                                         n_months=YEARS * 12)

  # DP予測器の準備
  dp_predictor = DPOptimalStrategyPredictor(MODELS_PATH)

  # 3. グリッドループ
  spending_rules = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
  strategies_to_compare = [
      "固定最適比率", "ダイナミック最適比率 (V1)", "Dynamic Rebalance DP (V2)"
  ]

  results: List[Dict[str, Any]] = []

  print(
      f"全 {len(spending_rules) * len(strategies_to_compare)} パターンのシミュレーションを実行中..."
  )

  it = 0
  total_its = len(spending_rules) * len(strategies_to_compare)
  for rule in spending_rules:
    # 初期資産の計算
    # 支出倍率 spend_mult = 1.0 固定
    base_spend_annual = BASE_SPEND_ANNUAL_WO_PENSION + PENSION_PREMIUM_ANNUAL
    initial_annual_cost = base_spend_annual  # spend_mult=1.0
    init_money = initial_annual_cost / (rule / 100.0)
    initial_annual_cost_wo_pension = initial_annual_cost - PENSION_PREMIUM_ANNUAL

    # 支出設定 (トレンド考慮, normalized=False なのでそのまま倍率として使える or 金額として設定)
    # Strategy.annual_cost には float または List[float] を渡す。
    # ここでは年間支出のリストを渡す。
    annual_cost_setting = [
        (val * 12.0 / 10000.0) for val in spending_multipliers_by_age
    ]

    for strat_name in strategies_to_compare:
      if it % 10 == 0:
        print(f"Progress: {it}/{total_its}")
      it += 1

      # 戦略に応じた動的リバランス関数の定義
      if strat_name == "固定最適比率":
        fixed_ratio = calculate_optimal_strategy(
            s_rate=np.array([rule / 100.0]),
            remaining_years=YEARS,
            base_yield=ZERO_RISK_YIELD,
            tax_rate=TAX_RATE,
            inflation_rate=INFLATION_RATE)[0]

        def dynamic_rebalance_fn(total_net, annual_spend, rem_years,
                                 post_tax_net):
          return {ORUKAN_NAME: fixed_ratio, ZERO_RISK_NAME: 1.0 - fixed_ratio}

      elif strat_name == "ダイナミック最適比率 (V1)":

        def dynamic_rebalance_fn(total_net, annual_spend, rem_years,
                                 post_tax_net):
          s_rate = annual_spend / np.maximum(total_net, 1.0)
          ratio = calculate_optimal_strategy(s_rate=s_rate,
                                             remaining_years=rem_years,
                                             base_yield=ZERO_RISK_YIELD,
                                             tax_rate=TAX_RATE,
                                             inflation_rate=INFLATION_RATE)
          return {ORUKAN_NAME: ratio, ZERO_RISK_NAME: 1.0 - ratio}

      else:  # Dynamic Rebalance DP (V2)

        def dynamic_rebalance_fn(total_net, annual_spend, rem_years,
                                 post_tax_net):
          current_age = START_AGE + int(YEARS - rem_years)
          s_rate = annual_spend / np.maximum(post_tax_net, 1.0)
          predict_age = min(current_age, 94)
          ratio = dp_predictor.predict_a_opt(predict_age, s_rate)
          return {ORUKAN_NAME: ratio, ZERO_RISK_NAME: 1.0 - ratio}

      strategy = Strategy(name=f"{strat_name}_Rule{rule}",
                          initial_money=float(init_money),
                          initial_loan=0.0,
                          yearly_loan_interest=0.0,
                          initial_asset_ratio={
                              ORUKAN_NAME: 1.0,
                              zr_asset_obj: 0.0
                          },
                          annual_cost=annual_cost_setting,
                          inflation_rate=CPI_NAME,
                          tax_rate=TAX_RATE,
                          rebalance_interval=12,
                          dynamic_rebalance_fn=dynamic_rebalance_fn,
                          selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
                          record_annual_spend=True,
                          cashflow_rules=cf_rules)

      res = simulate_strategy(strategy,
                              monthly_prices,
                              monthly_cashflows=monthly_cashflows)

      # 結果の記録 (共通項目)
      base_row = {
          "spend_multiplier": 1.0,
          "strategy": strat_name,
          "spending_rule": rule,
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

      # 2. 支出額のパーセンタイル (特定の条件のみ)
      if rule == 4.0 and strat_name == "Dynamic Rebalance DP (V2)":
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
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {CSV_PATH} に保存しました。")


if __name__ == "__main__":
  main()
