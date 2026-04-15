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
- 年金: 60歳から繰り上げ受給 (141万円/年)

可変条件:
- ダイナミックスペンディング
  - on の時: (上限3%, 下限0%)
  - off の時: 支出トレンド: 家計調査報告のデータに基づき、年齢とともに変化
- 何％ルールにするか
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
DATA_DIR = "data/all_60yr/"
CSV_PATH = os.path.join(DATA_DIR, "all_60yr_grid.csv")


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 35  # 60歳から95歳まで
  SEED = 42
  CPI_NAME = "Japan_CPI"
  FX_NAME = "USDJPY_0_10.53"
  ZERO_RISK_NAME = "ゼロリスク資産"
  ORUKAN_NAME = "オルカン"

  TRUST_FEE = 0.0005775
  PENSION_ANNUAL = 141.0  # 万円 (Step 1で決定)
  ZERO_RISK_YIELD = 0.04
  TAX_RATE = 0.20315

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

  configs: List[AssetConfigType] = [
      fx_asset, base_sp500, base_acwi, orukan, base_cpi
  ]

  print(f"価格推移を生成中... (試行回数: {N_SIM}, 期間: {YEARS}年)")
  monthly_prices = generate_monthly_asset_prices(configs,
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # 2. グリッドパラメータ
  BASE_SPEND_ANNUAL = 540.0  # 初年度支出ベースライン (45万 * 12ヶ月)
  spend_multipliers = [0.36, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
  # 支出率のルール:
  spending_rules = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
  use_dynamic_spending_list = [False, True]

  all_combinations = list(
      product(spend_multipliers, spending_rules, use_dynamic_spending_list))

  results: List[Dict[str, Any]] = []

  # 年齢による支出倍率の取得 (60歳から35年間)
  spending_multipliers_by_age = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION], start_age=60,
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

  for i, (spend_mult, rule, use_dyn_spend) in enumerate(all_combinations):
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
    cf_configs: List[CashflowConfig] = [
        PensionConfig(
            name="Pension",
            amount=PENSION_ANNUAL / 12.0,
            start_month=0,
            # ダイナミックスペンディング時は支出が名目なので、年金も名目（CPI連動なし）として扱う
            cpi_name=CPI_NAME if not use_dyn_spend else None)
    ]
    # ダイナミックスペンディングは名目ベース。
    # core.pyにおいて、inflation_rateがNoneの場合は名目値をそのまま使用し、
    # PensionConfigのcpi_nameがNoneの場合も名目値が生成される。
    # これにより、ダイナミックスペンディングONの時はすべて名目空間で計算される。

    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略
    strategy = Strategy(
        name=f"Rule_{rule}%_Mult_{spend_mult}_Dyn_{use_dyn_spend}",
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
        cashflow_rules=[
            CashflowRule(source_name="Pension",
                         cashflow_type=CashflowType.EXTRAORDINARY)
        ])

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 結果の記録
    row = {
        "spend_multiplier": spend_mult,
        "spending_rule": rule,
        "use_dynamic_spending": 1 if use_dyn_spend else 0,
        "initial_money": init_money,
        "initial_annual_cost": initial_annual_cost
    }
    # 各年時点の生存確率を記録
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
