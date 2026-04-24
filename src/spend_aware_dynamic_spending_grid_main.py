"""
Spend-Aware Dynamic Spending のグリッドシミュレーション実行スクリプト。

引数 --exp_name によって異なる実験を実行します。
- simple: 単一のパラメータ設定でのデモ実行
- v1_v2_comp: DSv1 (従来のガードレール) と DSv2 (生存確率ベース) の比較

全戦略で DRv2 (DPベースの動的リバランス) を使用します。
"""

import argparse
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from src.core import (DynamicSpending, SimulationResult, Strategy,
                      ZeroRiskAsset, simulate_strategy)
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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name",
                      type=str,
                      default="simple",
                      choices=["simple", "v1_v2_comp"])
  args = parser.parse_args()

  # 共通設定
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

  print(f"価格推移を生成中... (exp={args.exp_name}, 試行回数: {N_SIM})")
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

  # DRv2: DPベースの動的リバランス
  def dynamic_rebalance_fn(total_net, annual_spend, rem_years, post_tax_net):
    predict_age = int(round(START_AGE + (YEARS - (rem_years - 0.25))))
    ratio = dp_predictor.get_a_opt_with_winning_threshold(
        predict_age, post_tax_net, annual_spend)
    return {ORUKAN_NAME: ratio, ZERO_RISK_NAME: 1.0 - ratio}

  results_summary = []
  results_survival_probs = []
  results_spends = []

  if args.exp_name == "simple":
    rules = [4.0]
    strategies = ["FixedSpend", "SpendAware"]
  else:
    rules = [3.0, 3.5, 4.0, 4.5, 5.0]
    strategies = ["DRv2_DSv1", "DRv2_DSv2", "FixedSpend"]

  for rule in rules:
    print(f"\n--- Rule {rule}% ---")
    base_spend_annual_init = BASE_SPEND_ANNUAL_WO_PENSION + PENSION_PREMIUM_ANNUAL
    init_money = base_spend_annual_init / (rule / 100.0)

    res_dict: Dict[str, SimulationResult] = {}

    for strat_name in strategies:
      # 戦略の設定
      annual_cost: Union[List[float], DynamicSpending, SpendAwareDynamicSpending]
      inflation_rate: Union[float, str, None] = CPI_NAME
      
      # FixedSpend (Baseline) の場合は潤沢な資産で実行する
      current_init_money = init_money
      if strat_name == "FixedSpend":
        current_init_money = 100 * 10000 # 100億円

      if strat_name == "FixedSpend":
        annual_cost = annual_cost_setting
      elif strat_name == "SpendAware" or strat_name == "DRv2_DSv2":
        # DSv2: 生存確率ベースの動的支出
        annual_cost = SpendAwareDynamicSpending(
            initial_age=START_AGE,
            p_low=0.85,
            p_high=0.97,
            lower_mult=0.99,  # -1%
            upper_mult=1.02,   # +2.0%
            annual_cost_real=annual_cost_setting,
            dp_predictor=dp_predictor)
        inflation_rate = 0.0  # SpendAware は名目ベースで計算するため 0 に設定
      elif strat_name == "DRv2_DSv1":
        # DSv1: core.py の DynamicSpending (Vanguard型) をそのまま使用
        annual_cost = DynamicSpending(
            initial_annual_spend=base_spend_annual_init,
            target_ratio=rule / 100.0,
            upper_limit=0.01,   # +1.0%
            lower_limit=-0.015  # -1.5%
        )

      strategy = Strategy(name=strat_name,
                          initial_money=float(current_init_money),
                          initial_loan=0.0,
                          yearly_loan_interest=0.0,
                          initial_asset_ratio={
                              ORUKAN_NAME: 1.0,
                              zr_asset_obj: 0.0
                          },
                          annual_cost=annual_cost,
                          inflation_rate=inflation_rate,
                          tax_rate=TAX_RATE,
                          rebalance_interval=12,
                          dynamic_rebalance_fn=dynamic_rebalance_fn,
                          selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
                          record_annual_spend=True,
                          cashflow_rules=cf_rules)

      print(f"{strat_name} 実行中...")
      res = simulate_strategy(strategy,
                             monthly_asset_prices=monthly_prices,
                             monthly_cashflows=monthly_cashflows)
      
      res_dict[strat_name] = res

      # 1. 生存確率データの蓄積 (visualize.py 用)
      # FixedSpend は生存確率の比較には含めない（巨大資金のため）
      if strat_name != "FixedSpend":
        for y in range(YEARS + 1):
          survival_rate = np.mean(res.sustained_months >= y * 12)
          results_survival_probs.append({
              "rule": rule,
              "strategy": strat_name,
              "year": y,
              "survival_rate": survival_rate
          })

        # 2. 最終生存確率サマリー
        surv_rate = np.mean(res.sustained_months == YEARS * 12)
        results_summary.append({
            "rule": rule,
            "strategy": strat_name,
            "survival_rate": surv_rate
        })

    # 3. 支出データの集計 (DSv1, DSv2 両方が生存しているパスのみ)
    if args.exp_name == "v1_v2_comp":
      rv1 = res_dict.get("DRv2_DSv1")
      rv2 = res_dict.get("DRv2_DSv2")
      rfx = res_dict.get("FixedSpend")
      
      if rv1 is not None and rv2 is not None and rfx is not None:
        prices_cpi = monthly_prices[CPI_NAME]
        for y in range(YEARS):
          # DSv1, DSv2 両戦略が年末時点で生存しているパス
          active_mask = (rv1.sustained_months >= (y + 1) * 12) & \
                        (rv2.sustained_months >= (y + 1) * 12)
          
          if np.any(active_mask):
            for strat_name in strategies:
              res_obj = res_dict[strat_name]
              if res_obj.annual_spends is not None:
                real_vals = res_obj.annual_spends[active_mask, y] / prices_cpi[active_mask, y * 12]
                results_spends.append({
                    "rule": rule,
                    "strategy": strat_name,
                    "year": y + 1,
                    "p25": np.percentile(real_vals, 25),
                    "p50": np.percentile(real_vals, 50),
                    "p75": np.percentile(real_vals, 75)
                })

  # 結果の保存
  data_dir = f"data/spend_aware_dynamic_spending"
  os.makedirs(data_dir, exist_ok=True)
  
  df_summary = pd.DataFrame(results_summary)
  df_survival = pd.DataFrame(results_survival_probs)
  df_spends = pd.DataFrame(results_spends)
  
  summary_path = os.path.join(data_dir, f"{args.exp_name}_summary.csv")
  survival_path = os.path.join(data_dir, f"{args.exp_name}_survival.csv")
  spends_path = os.path.join(data_dir, f"{args.exp_name}_spends.csv")
  
  df_summary.to_csv(summary_path, index=False)
  df_survival.to_csv(survival_path, index=False)
  df_spends.to_csv(spends_path, index=False)
  
  print(f"\n結果を保存しました:\n- {summary_path}\n- {survival_path}\n- {spends_path}")


if __name__ == "__main__":
  main()
