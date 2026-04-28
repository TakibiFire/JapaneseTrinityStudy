"""
40歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。

実験設定:
- 期間: 55年 (40歳〜95歳)
- 試行回数: 1,000回~2,000回
- 資産構成:
    - オルカン (ファットテール考慮・S&P500補完モデル, 信託報酬 0.05775%)
    - ゼロリスク資産 (利回り 4.0%)
- ダイナミックリバランス: 毎年実施 (資産寿命を最大化する最適比率)
- 為替: USDJPY (期待リターン 0%, リスク 10.53%)
- インフレ: AR(12) 粘着性モデル
- 初年度支出ベースライン: 464.3万/年 (二人以上世帯の平均)
  - 年金保険代込支出: 510万 (42.5万/月*12)
  - 40歳で止めるので厚生年金保険料率を引く = - 457,500円
- 税率: 20.315%
- 年金保険料:
  - 40-60歳まで国民年金保険料を支払い: 20.4万/年。別途 cashflow として考える
- 年金受給開始年齢: 60

可変条件:
- 世帯人数 (1, 2)
- 初年度支出倍率
- ダイナミックスペンディング
  * なし: 出費のトレンドを家計調査報告のデータに基づき推移させる。 
  * あり: 年出費率がX%に近づくように、上限+3%, 下限+0%（絶対に額面は減らさない）で支出を毎年決定。
- 支出率のルール (資産額に対する比率)
"""

import argparse
import os
from itertools import product
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core import (DynamicSpending, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowConfig,
                                        CashflowRule, CashflowType,
                                        PensionConfig, generate_cashflows)
from src.lib.dynamic_rebalance import (calculate_optimal_strategy,
                                       calculate_safe_target_ratio)
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.world_setup import create_standard_world


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="40歳リタイア開始・95歳までの生存確率を分析するグリッドサーチスクリプト。")
  parser.add_argument("--exp_type",
                      type=str,
                      default="P60-D1",
                      help="実験設定 (P-D-RANGE or P60-D1)")
  args = parser.parse_args()

  # 設定
  exp_type = args.exp_type
  assert exp_type in (
      # 生存確率を上げるために年金受け取りの受給タイミングとDynamicSpendingを
      # するかどうかの最適組み合わせを求める。
      "P-D-RANGE",
      # 年金受け取りの受給タイミング=60, DynamicSpending=ON が確定した。
      # 細かい数字を見ていく。
      "P60-D1",
  ), f"Unsupported exp_type: {exp_type}"

  data_dir = "data/all_40yr/"
  csv_path = os.path.join(data_dir, f"{exp_type}.csv")

  # 共通設定
  YEARS = 55  # 40歳から95歳まで
  START_AGE = 40
  SEED = 42
  CPI_NAME = "Japan_CPI"
  PENSION_CPI_NAME = "Pension_CPI"
  FX_NAME = "USDJPY_0_10.53"
  ZERO_RISK_NAME = "ゼロリスク資産"
  ORUKAN_NAME = "オルカン"

  TRUST_FEE = 0.0005775
  ZERO_RISK_YIELD = 0.04
  TAX_RATE = 0.20315
  CURRENT_YEAR = 2026
  MACRO_ECONOMIC_SLIDE_END_YEAR = 2057

  if exp_type == "P-D-RANGE":
    spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.5, 3.0]
    spending_rules = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    household_sizes = [1]
    N_SIM = 1000
    pension_start_ages = [60, 65]
    use_dynamic_spending_list = [False, True]
  elif exp_type == "P60-D1":
    spend_multipliers = [
        0.36, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0
    ]
    spending_rules = [2.8, 3.0, 3.33, 3.66, 4.0, 4.33, 4.66, 5.0, 5.5, 6.0, 7.0]
    N_SIM = 2000
    household_sizes = [1]
    pension_start_ages = [60]
    use_dynamic_spending_list = [True]
  else:
    raise KeyError(f"Unsupported {exp_type}")

  os.makedirs(data_dir, exist_ok=True)

  # 1. アセット生成
  world = create_standard_world(
      n_sim=N_SIM,
      start_age=START_AGE,
      end_age=START_AGE + YEARS - 1,
      retirement_age=60,  # 国民年金は60歳まで
      pension_start_age=65,  # Dummy, will be overridden in the loop
      seed=SEED,
      trust_fee=TRUST_FEE,
      zero_risk_yield=ZERO_RISK_YIELD)

  monthly_prices = world.monthly_prices
  zr_asset_obj = world.zr_asset_obj
  ORUKAN_NAME = world.ORUKAN_NAME
  ZERO_RISK_NAME = world.ZERO_RISK_NAME
  CPI_NAME = world.CPI_NAME
  PENSION_CPI_NAME = world.PENSION_CPI_NAME

  # 2. グリッドパラメータ

  all_combinations = list(
      product(household_sizes, pension_start_ages, spend_multipliers,
              use_dynamic_spending_list, spending_rules))

  # 年金設定の定義 (世帯人数, 受給開始年齢) -> (保険料/年, 受給額/年)
  # 基礎年金満額: 81.6万, 厚生年金相当: 2.736 * (40 - 22) = 49.248
  KISO_FULL_ANNUAL = 81.6
  KOUSEI_UNIT_ANNUAL = 49.2
  pension_map = {
      (1, 60): (-20.4, 99.4),
      (1, 65): (-20.4, 130.8),
      (2, 60): (-40.7, 161.4),
      (2, 65): (-40.7, 212.4),
  }

  results: List[Dict[str, Any]] = []

  # 年齢による支出倍率の取得 (40歳から55年間)
  # 案A: 年金保険料を除外したトレンドを使用
  spending_multipliers_by_age = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=START_AGE,
      num_years=YEARS)

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")

  # 初年度支出ベースライン (二人以上世帯の平均)
  # 510万 (42.5万/月*12) - 40歳退職による厚生年金保険料率引分 (457,500円) = 464.3万
  BASE_SPEND_ANNUAL_WO_PENSION = 464.3

  # ダイナミックリバランスの関数
  def dynamic_rebalance_fn(total_net, annual_spend, rem_years, post_tax_net):
    s_rate = annual_spend / np.maximum(total_net, 1.0)
    orukan_ratio = calculate_optimal_strategy(s_rate=s_rate,
                                              remaining_years=rem_years,
                                              base_yield=ZERO_RISK_YIELD,
                                              tax_rate=TAX_RATE,
                                              inflation_rate=0.0177)
    return {ORUKAN_NAME: orukan_ratio, ZERO_RISK_NAME: 1.0 - orukan_ratio}

  # セーフティな支出率 (DynamicSpending用)
  target_ratio = calculate_safe_target_ratio(YEARS)

  for i, (household_size, pension_start, spend_mult, use_dyn_spend,
          rule) in enumerate(all_combinations):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    # 国民年金保険料を正の値にする。支払い量は household_sizeに依存。
    pension_cost = -pension_map[(household_size, pension_start)][0]
    # base_spend_annual は2人世帯の40歳の平均的支出。国民年金保険料を含む。
    base_spend_annual = (BASE_SPEND_ANNUAL_WO_PENSION + pension_cost)
    # 初年度支出 (国民年金保険料含む) と初期資産
    initial_annual_cost = base_spend_annual * spend_mult
    init_money = initial_annual_cost / (rule / 100.0)
    # 初年度支出, 国民年金保険料含まない。これを initial_cost にする。
    initial_annual_cost_wo_pension = initial_annual_cost - pension_cost

    # 支出とキャッシュフローの設定
    cf_configs: List[CashflowConfig] = []
    cf_rules: List[CashflowRule] = []

    if use_dyn_spend:
      # ダイナミックスペンディング
      ds_handler = DynamicSpending(
          initial_annual_spend=initial_annual_cost_wo_pension,
          target_ratio=target_ratio,
          upper_limit=0.03,
          lower_limit=0.0)
      cf_configs.append(
          BaseSpendConfig(name="base_spend",
                          amount=initial_annual_cost_wo_pension,
                          cpi_name=None))
      cf_rules.append(
          CashflowRule(source_name="base_spend",
                       cashflow_type=CashflowType.REGULAR,
                       dynamic_handler=ds_handler))
    else:
      # 年齢による支出トレンドを適用
      annual_cost_list = [
          initial_annual_cost_wo_pension * m
          for m in spending_multipliers_by_age
      ]
      cf_configs.append(
          BaseSpendConfig(name="base_spend",
                          amount=annual_cost_list,
                          cpi_name=CPI_NAME))
      cf_rules.append(
          CashflowRule(source_name="base_spend",
                       cashflow_type=CashflowType.REGULAR))

    # キャッシュフロー (年金保険料と受給)
    premium_annual, _ = pension_map[(household_size, pension_start)]

    # 40歳から60歳までの保険料支払い (20年間 = 240ヶ月)
    cf_configs.append(
        PensionConfig(name="Pension_Premium",
                      amount=premium_annual / 12.0,
                      start_month=0,
                      end_month=240,
                      cpi_name=CPI_NAME))
    cf_rules.append(
        CashflowRule(source_name="Pension_Premium",
                     cashflow_type=CashflowType.REGULAR))

    # 受給開始年齢に基づく受給
    receipt_start_month = (pension_start - START_AGE) * 12
    reduction_rate = 0.76 if pension_start == 60 else 1.0

    # 本人厚生年金
    kousei_annual = KOUSEI_UNIT_ANNUAL * reduction_rate
    cf_configs.append(
        PensionConfig(name="Pension_Receipt_Kousei",
                      amount=kousei_annual / 12.0,
                      start_month=receipt_start_month,
                      cpi_name=CPI_NAME))
    cf_rules.append(
        CashflowRule(source_name="Pension_Receipt_Kousei",
                     cashflow_type=CashflowType.REGULAR))

    # 本人基礎年金 (マクロ経済スライド適用)
    kiso_annual = KISO_FULL_ANNUAL * reduction_rate
    cf_configs.append(
        PensionConfig(name="Pension_Receipt_Kiso",
                      amount=kiso_annual / 12.0,
                      start_month=receipt_start_month,
                      cpi_name=PENSION_CPI_NAME))
    cf_rules.append(
        CashflowRule(source_name="Pension_Receipt_Kiso",
                     cashflow_type=CashflowType.REGULAR))

    # 配偶者基礎年金 (2人世帯の場合)
    if household_size == 2:
      spouse_kiso_annual = KISO_FULL_ANNUAL * reduction_rate
      cf_configs.append(
          PensionConfig(name="Pension_Receipt_Spouse_Kiso",
                        amount=spouse_kiso_annual / 12.0,
                        start_month=receipt_start_month,
                        cpi_name=PENSION_CPI_NAME))
      cf_rules.append(
          CashflowRule(source_name="Pension_Receipt_Spouse_Kiso",
                       cashflow_type=CashflowType.REGULAR))

    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略
    strategy = Strategy(
        name=
        f"H{household_size}_P{pension_start}_Mult{spend_mult}_Dyn{use_dyn_spend}_Rule{rule}",
        initial_money=float(init_money),
        initial_loan=0.0,
        yearly_loan_interest=0.0,
        initial_asset_ratio={
            ORUKAN_NAME: 1.0,
            zr_asset_obj: 0.0
        },
        tax_rate=TAX_RATE,
        rebalance_interval=12,
        dynamic_rebalance_fn=dynamic_rebalance_fn,
        selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
        record_annual_spend=True,
        cashflow_rules=cf_rules)

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 結果の記録
    base_row: Dict[str, Any] = {
        "household_size": household_size,
        "pension_start_age": pension_start,
        "spend_multiplier": spend_mult,
        "use_dynamic_spending": 1 if use_dyn_spend else 0,
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

    # 2. 支出額のパーセンタイル
    if res.annual_spends is not None:
      # res.annual_spends shape: (n_sim, n_years)
      p25 = np.percentile(res.annual_spends, 25, axis=0)
      p50 = np.percentile(res.annual_spends, 50, axis=0)
      p75 = np.percentile(res.annual_spends, 75, axis=0)

      for name, p_values in [("spend25p", p25), ("spend50p", p50),
                             ("spend75p", p75)]:
        row_p = base_row.copy()
        row_p["value_type"] = name
        for year in range(1, YEARS + 1):
          # year は 1-indexed, p_values も 0番目が 1年目
          row_p[str(year)] = p_values[year - 1]
        results.append(row_p)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
