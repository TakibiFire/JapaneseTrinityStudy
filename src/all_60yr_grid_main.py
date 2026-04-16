"""
60歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。

実験設定:
- 期間: 35年 (60歳〜95歳)
- 試行回数: 5,000回
- 資産構成:
    - オルカン (ファットテール考慮・S&P500補完モデル, 信託報酬 0.05775%)
    - ゼロリスク資産 (利回り 4.0%)
- ダイナミックリバランス: 毎年実施 (資産寿命を最大化する最適比率)
- 為替: USDJPY (期待リターン 0%, リスク 10.53%)
- インフレ: AR(12) 粘着性モデル
- 初年度支出ベースライン: 540万/年 (60歳の出費平均45万 * 12か月)
  - ちなみにこの値は2人以上の世帯
  - 65歳以上単身無職世帯は 16.2万 (45万の 36%相当)
- 税率: 20.315%
- 年金: 60歳または65歳から受給 (世帯人数と開始年齢により変動)

可変条件:
- 年金受給開始年齢 (60, 65)
- ダイナミックスペンディングの有無
- 支出率のルール (資産額に対する比率)
- 初年度支出倍率
"""

import os
from itertools import product
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core import (DynamicSpending, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (AssetConfigType, DerivedAsset, ForexAsset,
                                     SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, CashflowRule,
                                        CashflowType, PensionConfig,
                                        generate_cashflows)
from src.lib.dynamic_rebalance import (calculate_optimal_strategy,
                                       calculate_safe_target_ratio)
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)

# 設定
EXP_TYPE = "P60-D1"
assert EXP_TYPE in (
    # 年金受け取りの受給タイミングとDynamicSpendingをするかどうかの最適組み合わせを求める。
    "P-D-RANGE",
    # 年金受け取りの受給タイミング=60, DynamicSpending=ON が確定。
    # より詳細なパラメータで分析を行う。
    "P60-D1",
)

DATA_DIR = "data/all_60yr/"
CSV_PATH = os.path.join(DATA_DIR, f"{EXP_TYPE}.csv")


def main():
  # 共通設定
  YEARS = 35  # 60歳から95歳まで
  START_AGE = 60
  SEED = 42
  CPI_NAME = "Japan_CPI"
  PENSION_CPI_NAME = "Pension_CPI"
  FX_NAME = "USDJPY_0_10.53"
  ZERO_RISK_NAME = "ゼロリスク資産"
  ORUKAN_NAME = "オルカン"

  TRUST_FEE = 0.0005775
  # 基礎年金満額: 81.6万, 厚生年金相当: 103.9万 (185.5 - 81.6)
  KISO_FULL_ANNUAL = 81.6
  KOUSEI_UNIT_ANNUAL = 103.9
  ZERO_RISK_YIELD = 0.04
  TAX_RATE = 0.20315
  CURRENT_YEAR = 2026
  MACRO_ECONOMIC_SLIDE_END_YEAR = 2057

  if EXP_TYPE == "P-D-RANGE":
    spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.5, 3.0]
    spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    N_SIM = 1000
    pension_start_ages = [60, 65]
    use_dynamic_spending_list = [False, True]
  elif EXP_TYPE == "P60-D1":
    spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
    spending_rules = [2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0]
    N_SIM = 3000
    pension_start_ages = [60]
    use_dynamic_spending_list = [True]
  else:
    raise KeyError(f"Unsupported {EXP_TYPE}")

  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. アセット生成
  # 為替 (USDJPY 0%, 10.53%)
  fx_asset = ForexAsset(name=FX_NAME,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))

  # オルカン (共通モデルから取得)
  base_sp500 = get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y)
  base_acwi = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)

  # 投資対象としてのオルカン (為替と信託報酬を適用)
  orukan = DerivedAsset(name=ORUKAN_NAME,
                        base=base_acwi.name,
                        trust_fee=TRUST_FEE,
                        forex=FX_NAME)

  # ゼロリスク資産 (利回り 4%)
  zr_asset_obj = ZeroRiskAsset(name=ZERO_RISK_NAME, yield_rate=ZERO_RISK_YIELD)

  # CPI (共通モデル)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)

  # 年金用CPI (マクロ経済スライド 0.5% 抑制)
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

  # 2. グリッドパラメータ
  BASE_SPEND_ANNUAL = 540.0  # 初年度支出ベースライン (45万 * 12ヶ月)

  all_combinations = list(
      product(pension_start_ages, spend_multipliers, spending_rules,
              use_dynamic_spending_list))

  results: List[Dict[str, Any]] = []

  # 年齢による支出倍率の取得 (60歳から35年間)
  spending_multipliers_by_age = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION],
      start_age=START_AGE,
      num_years=YEARS)

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")

  # ダイナミックリバランスの関数
  def dynamic_rebalance_fn(total_net, annual_spend, rem_years):
    s_rate = annual_spend / np.maximum(total_net, 1.0)
    # ゼロリスク資産の利回りを考慮して最適比率を計算
    orukan_ratio = calculate_optimal_strategy(
        s_rate=s_rate,
        remaining_years=rem_years,
        base_yield=ZERO_RISK_YIELD,
        tax_rate=TAX_RATE,
        inflation_rate=0.0177  # 近似式用の標準的なインフレ率
    )
    return {ORUKAN_NAME: orukan_ratio, ZERO_RISK_NAME: 1.0 - orukan_ratio}

  # セーフティな支出率 (DynamicSpending用)
  # ゼロリスク資産でも資産寿命が YEARS 年となるような支出率
  target_ratio = calculate_safe_target_ratio(YEARS)

  for i, (pension_start, spend_mult, rule,
          use_dyn_spend) in enumerate(all_combinations):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    # 初期支出と初期資産
    initial_annual_cost = BASE_SPEND_ANNUAL * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)

    # 支出設定
    annual_cost_setting: Union[float, List[float], DynamicSpending]
    inflation_rate_setting: Optional[str]
    if use_dyn_spend:
      # ダイナミックスペンディング (上限3%, 下限0%)
      # target_ratio は calculate_safe_target_ratio(YEARS) で求めた値を使用
      annual_cost_setting = DynamicSpending(
          initial_annual_spend=initial_annual_cost,
          target_ratio=target_ratio,
          upper_limit=0.03,
          lower_limit=0.0)
      # DynamicSpendingは名目で上限下限を扱うため、シミュレーション側のインフレ調整はオフ
      inflation_rate_setting = None
    else:
      # 年齢による支出トレンドを適用
      annual_cost_setting = [
          initial_annual_cost * m for m in spending_multipliers_by_age
      ]
      inflation_rate_setting = CPI_NAME

    # キャッシュフロー (年金)
    receipt_start_month = (pension_start - START_AGE) * 12
    reduction_rate = 0.76 if pension_start == 60 else 1.0

    kousei_annual = KOUSEI_UNIT_ANNUAL * reduction_rate
    kiso_annual = KISO_FULL_ANNUAL * reduction_rate

    cf_configs: List[CashflowConfig] = [
        # 厚生年金 (CPI連動)
        PensionConfig(name="Pension_Kousei",
                      amount=kousei_annual / 12.0,
                      start_month=receipt_start_month,
                      cpi_name=CPI_NAME),
        # 基礎年金 (マクロ経済スライド適用)
        PensionConfig(name="Pension_Kiso",
                      amount=kiso_annual / 12.0,
                      start_month=receipt_start_month,
                      cpi_name=PENSION_CPI_NAME)
    ]

    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略
    strategy = Strategy(
        name=
        f"P{pension_start}_Mult_{spend_mult}_Rule_{rule}%_Dyn_{use_dyn_spend}",
        initial_money=float(init_money),
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={
            ORUKAN_NAME: 1.0,
            zr_asset_obj: 0.0
        },  # 初期値
        annual_cost=annual_cost_setting,
        inflation_rate=inflation_rate_setting,
        tax_rate=TAX_RATE,
        rebalance_interval=12,
        dynamic_rebalance_fn=dynamic_rebalance_fn,
        selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
        record_annual_spend=True,  # パーセンタイル分析に必要
        cashflow_rules=[
            CashflowRule(source_name="Pension_Kousei",
                         cashflow_type=CashflowType.REGULAR),
            CashflowRule(source_name="Pension_Kiso",
                         cashflow_type=CashflowType.REGULAR)
        ])

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 1. 生存確率
    row_survival: Dict[str, Any] = {
        "pension_start_age": pension_start,
        "spend_multiplier": spend_mult,
        "spending_rule": rule,
        "use_dynamic_spending": 1 if use_dyn_spend else 0,
        "initial_money": init_money,
        "initial_annual_cost": initial_annual_cost,
        "value_type": "survival"
    }
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row_survival[str(year)] = survival_rate
    results.append(row_survival)

    # 2. 支出額のパーセンタイル
    if res.annual_spends is not None:
      p25 = np.percentile(res.annual_spends, 25, axis=0)
      p50 = np.percentile(res.annual_spends, 50, axis=0)
      p75 = np.percentile(res.annual_spends, 75, axis=0)

      for name, p_values in [("spend25p", p25), ("spend50p", p50),
                             ("spend75p", p75)]:
        row_p: Dict[str, Any] = {
            "pension_start_age": pension_start,
            "spend_multiplier": spend_mult,
            "spending_rule": rule,
            "use_dynamic_spending": 1 if use_dyn_spend else 0,
            "initial_money": init_money,
            "initial_annual_cost": initial_annual_cost,
            "value_type": name
        }
        for year in range(1, YEARS + 1):
          row_p[str(year)] = p_values[year - 1]
        results.append(row_p)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {CSV_PATH} に保存しました。")


if __name__ == "__main__":
  main()
