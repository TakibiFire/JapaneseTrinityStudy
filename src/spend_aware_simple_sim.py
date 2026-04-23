"""
SpendAwareDynamicSpending (生存確率ベースのガードレール戦略) の挙動を確認するためのテストシミュレーションスクリプト。

このスクリプトは src/dynamic_rebalance_dp_grid_main.py と同じ資産構成・キャッシュフロー設定を用いて、
動的支出（ガードレール）の有無による生存確率と支出推移を比較します。

シミュレーション設定:
- 期間: 55年間 (40歳〜95歳)
- パス数: 100
- 構成: 1人世帯, 年金開始60歳 (H1_P60)
- ガードレール:
    - p_low: 0.85, p_high: 0.95
    - lower_mult: 0.90 (10%削減), upper_mult: 1.10 (10%増額)
"""

import os
from typing import List

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
from src.lib.dp_predictor import DPOptimalStrategyPredictor
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers,
                                      get_retired_spending_values)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)
from src.lib.spend_aware_dynamic_spending import SpendAwareDynamicSpending
from src.lib.visualize_all_yr import create_spend_percentile_chart


def main():
  # 設定
  YEARS = 55
  START_AGE = 40
  SEED = 42
  N_SIM = 2000
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

  # 1. アセット生成 (src/dynamic_rebalance_dp_grid_main.py と同一)
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
  BASE_SPEND_ANNUAL_WO_PENSION = base_spending_monthly * 12.0 / 10000.0

  # 年金設定
  PENSION_START_AGE = 60
  PENSION_PREMIUM_ANNUAL = 20.4
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
  receipt_start_month = max((PENSION_START_AGE - START_AGE) * 12, 0)
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

  # 支出リスト (万円/年)
  annual_cost_setting = [
      (val * 12.0 / 10000.0) for val in spending_multipliers_by_age
  ]

  # 動的リバランス関数
  def dynamic_rebalance_fn(total_net, annual_spend, rem_years, post_tax_net):
    predict_age = int(round(START_AGE + (YEARS - (rem_years - 0.25))))
    ratio = dp_predictor.get_a_opt_with_winning_threshold(
        predict_age, post_tax_net, annual_spend)
    return {ORUKAN_NAME: ratio, ZERO_RISK_NAME: 1.0 - ratio}

  # 3. 共通設定でシミュレーション (Rule 4% 相当の資産額)
  rule = 4.0
  base_spend_annual_init = BASE_SPEND_ANNUAL_WO_PENSION + PENSION_PREMIUM_ANNUAL
  init_money = base_spend_annual_init / (rule / 100.0)

  # (A) SpendAware 戦略
  spend_strategy = SpendAwareDynamicSpending(
      initial_age=START_AGE,
      p_low=0.85,
      p_high=0.95,
      lower_mult=0.99,
      upper_mult=1.01,
      annual_cost_real=annual_cost_setting,
      dp_predictor=dp_predictor)

  strategy_aware = Strategy(name="SpendAware",
                            initial_money=float(init_money),
                            initial_loan=0.0,
                            yearly_loan_interest=0.0,
                            initial_asset_ratio={
                                ORUKAN_NAME: 1.0,
                                zr_asset_obj: 0.0
                            },
                            annual_cost=spend_strategy,
                            inflation_rate=CPI_NAME,
                            tax_rate=TAX_RATE,
                            rebalance_interval=12,
                            dynamic_rebalance_fn=dynamic_rebalance_fn,
                            selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
                            record_annual_spend=True,
                            cashflow_rules=cf_rules)

  print("SpendAware シミュレーション実行中...")
  res_aware = simulate_strategy(strategy_aware,
                                monthly_prices,
                                monthly_cashflows=monthly_cashflows,
                                debug_indices=[0, 10, 50])

  # (B) 固定支出戦略 (Baseline)
  strategy_fixed = Strategy(name="FixedSpend",
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

  print("FixedSpend シミュレーション実行中...")
  res_fixed = simulate_strategy(strategy_fixed,
                                monthly_prices,
                                monthly_cashflows=monthly_cashflows,
                                debug_indices=[0, 10, 50])

  # 4. 結果表示
  surv_aware = np.mean(res_aware.sustained_months == YEARS * 12)
  surv_fixed = np.mean(res_fixed.sustained_months == YEARS * 12)

  print(f"\nSurvival Rate (SpendAware): {surv_aware:.2%}")
  print(f"Survival Rate (Fixed): {surv_fixed:.2%}")
  print(f"Improvement: {surv_aware - surv_fixed:+.2%}")

  # 5. 可視化
  if res_aware.annual_spends is not None and res_fixed.annual_spends is not None:
    # 名目額のまま計算
    nom_aware = res_aware.annual_spends
    nom_fixed = res_fixed.annual_spends

    def get_percentile_df(nominal_spends, sustained_months, is_dynamic):
      p25 = np.zeros(YEARS)
      p50 = np.zeros(YEARS)
      p75 = np.zeros(YEARS)
      
      for y in range(YEARS):
        # 破産していないパス（sustained_months が年末以降）のみを抽出して計算
        active_mask = sustained_months >= (y + 1) * 12
        active_vals = nominal_spends[active_mask, y]
        if len(active_vals) > 0:
          p25[y] = np.percentile(active_vals, 25)
          p50[y] = np.percentile(active_vals, 50)
          p75[y] = np.percentile(active_vals, 75)
        else:
          p25[y] = p50[y] = p75[y] = 0.0
      
      data = []
      data.append({"value_type": "spend25p", **{str(i+1): p25[i] for i in range(YEARS)}})
      data.append({"value_type": "spend50p", **{str(i+1): p50[i] for i in range(YEARS)}})
      data.append({"value_type": "spend75p", **{str(i+1): p75[i] for i in range(YEARS)}})
      df = pd.DataFrame(data)
      df["use_dynamic_spending"] = 1 if is_dynamic else 0
      return df

    # (A) 名目額のグラフ
    df_aware_nom = get_percentile_df(nom_aware, res_aware.sustained_months, True)
    df_fixed_nom = get_percentile_df(nom_fixed, res_fixed.sustained_months, False)
    df_plot_nom = pd.concat([df_aware_nom, df_fixed_nom])

    os.makedirs("docs/imgs", exist_ok=True)
    create_spend_percentile_chart(
        df_plot_nom,
        title="SpendAware vs Fixed Spend: Nominal Net Withdrawal",
        output_path="docs/imgs/spend_aware_test_chart_nom.svg",
        start_age=START_AGE,
        num_years=YEARS)
    print("\nChart saved to docs/imgs/spend_aware_test_chart_nom.svg")

    # (B) 実質額のグラフ
    def get_real_spends(nominal_spends, prices_cpi):
      real = np.zeros_like(nominal_spends)
      for y in range(YEARS):
        real[:, y] = nominal_spends[:, y] / prices_cpi[:, y * 12]
      return real

    real_aware = get_real_spends(nom_aware, monthly_prices[CPI_NAME])
    real_fixed = get_real_spends(nom_fixed, monthly_prices[CPI_NAME])

    df_aware_real = get_percentile_df(real_aware, res_aware.sustained_months, True)
    df_fixed_real = get_percentile_df(real_fixed, res_fixed.sustained_months, False)
    df_plot_real = pd.concat([df_aware_real, df_fixed_real])

    create_spend_percentile_chart(
        df_plot_real,
        title="SpendAware vs Fixed Spend: Real Net Withdrawal",
        output_path="docs/imgs/spend_aware_test_chart_real.svg",
        start_age=START_AGE,
        num_years=YEARS)
    print("Chart saved to docs/imgs/spend_aware_test_chart_real.svg")


if __name__ == "__main__":
  main()
