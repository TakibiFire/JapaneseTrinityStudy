"""
年金受給が資産寿命に与える影響をグリッドサーチで分析するスクリプト。

実験設定:
- 期間: 60年
- 試行回数: 5,000回
- 資産構成: オルカン 100% (7%, 15%, 信託報酬 0.05775%)
- CPI: AR(12) 粘着性モデル (1970年〜)
- 税率: 20.315%

シナリオ設定 (A~G):
- A (年金なし): 年金制度が存在しない世界。支出 = S。
- B (Sanity check): 保険料を払うが受給しない。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- C (継続・60歳): 保険料を60歳まで納付、60歳繰り上げ受給。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- D (継続・65歳): 保険料を60歳まで納付、65歳受給。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- E (免除・60歳): N歳(リタイア開始)から全額免除、60歳繰り上げ受給。支出 = S-21.5、保険料 = 0。
- F (免除・65歳): N歳から全額免除、65歳受給。支出 = S-21.5、保険料 = 0。
- G (未納・65歳): N歳から未納(放置)、65歳受給。支出 = S-21.5、保険料 = 0。

計算上の前提:
- 厚生年金: 22歳からリタイア開始(N歳)まで加入。年収500万を想定し、年額 = 2.736 * (N-22) 万。
- 基礎年金 (満額): 年額 81.6万。
- 免除時の基礎年金: 免除期間(N~60歳)の受給額は 1/2 として計算。
- 繰り上げ受給 (60歳): 受給額を 76% (0.4% * 60ヶ月減額) とする。
- マクロ経済スライド: 基礎年金にのみ適用し、2057年度に終了すると想定。厚生年金はCPI連動のみ。
- 国民年金保険料: 年額 21.5万 (CPI連動) とし、60歳の誕生日前まで支払う。
"""

import argparse
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
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowConfig,
                                        CashflowRule, CashflowType,
                                        PensionConfig, generate_cashflows)
from src.lib.simulation_defaults import get_cpi_ar12_config

# 設定
DATA_DIR = "data/pension"

def main():
  # 引数の処理
  parser = argparse.ArgumentParser(description="年金受給のグリッドシミュレーションを実行する")
  parser.add_argument("--exp_name",
                      type=str,
                      default="exp1",
                      help="実験名 (exp1)。カンマ区切りで複数指定可能")
  args = parser.parse_args()

  exp_names = [name.strip() for name in args.exp_name.split(",")]

  for exp_name in exp_names:
    run_experiment(exp_name)


def run_experiment(exp_name: str):
  # 共通設定
  CURRENT_YEAR = 2026
  MACRO_ECONOMIC_SLIDE_END_YEAR = 2057
  N_SIM = 5000
  YEARS = 60
  SEED = 42
  CPI_NAME = "Japan_CPI"
  PENSION_CPI_NAME = "Pension_CPI"

  os.makedirs(DATA_DIR, exist_ok=True)
  csv_path = os.path.join(DATA_DIR, f"{exp_name}.csv")

  # 1. アセット生成
  # オルカン (7%, 15%)
  orukan = Asset(name="オルカン",
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15))
  
  # AR(12) CPI (1970年〜)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)
  
  # 年金用CPI (マクロ経済スライド 0.5% 抑制)
  # 2057年度に終了予定 (2026年から31年 = 372ヶ月)
  pension_cpi = SlideAdjustedCpiAsset(name=PENSION_CPI_NAME,
                                      base_cpi=CPI_NAME,
                                      slide_rate=0.005,
                                      slide_end_month=(MACRO_ECONOMIC_SLIDE_END_YEAR - CURRENT_YEAR)*12)

  configs: List[AssetConfigType] = [orukan, base_cpi, pension_cpi]
  
  print(f"価格推移を生成中... (実験: {exp_name}, 試行回数: {N_SIM}, 期間: {YEARS}年)")
  monthly_prices = generate_monthly_asset_prices(configs,
                                                 n_paths=N_SIM,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # グリッドパラメータ
  initial_money_annual_cost_list = [(5000, 200), (10000, 400), (20000, 800)]
  initial_age_list = [30, 40, 50, 60]
  
  if exp_name == "exp1":
    scenarios = [
        "NoPensionWorld", "PayNoReceive", "Pay_60", "Pay_65", "Pay_70", "Pay_75",
        "Exempt_60", "Exempt_65", "Unpaid_65"
    ]
  else:
    print(f"Skipping unknown exp_name: {exp_name}")
    return

  all_combinations = list(product(
      initial_money_annual_cost_list,
      initial_age_list,
      scenarios
  ))

  results: List[Dict[str, Any]] = []

  PREMIUM_ANNUAL = 21.5
  KISO_FULL_ANNUAL = 81.6
  KOUSEI_UNIT_ANNUAL = 2.736  # 年収500万想定、1年勤務あたりの年金増額

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")
  for i, ((init_money, annual_cost), init_age, scenario) in enumerate(all_combinations):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    # 基本支出とキャッシュフローの初期化
    current_annual_cost = float(annual_cost)
    cf_configs: List[CashflowConfig] = []
    extra_cf_names = []
    
    # 記録用データ
    pension_start_age = 0
    kousei_annual_nominal = 0.0
    kiso_annual_nominal = 0.0

    # シナリオ別の設定
    if scenario == "NoPensionWorld":
      # 年金制度なし
      pass
    else:
      # 他のシナリオは生活費支出を 21.5万(保険料分) 減らす
      current_annual_cost -= PREMIUM_ANNUAL

      # 国民年金保険料 (Pay シナリオのみ支払い)
      if "Pay" in scenario:
        months_to_60 = max(0, (60 - init_age) * 12)
        if months_to_60 > 0:
          cf_name = f"Premium_{i}"
          cf_configs.append(PensionConfig(
              name=cf_name,
              amount=-(PREMIUM_ANNUAL / 12.0), # 負のキャッシュフロー
              start_month=0,
              end_month=months_to_60,
              cpi_name=CPI_NAME
          ))
          extra_cf_names.append(cf_name)

      # 年金受給 (PayNoReceive 以外)
      if scenario != "PayNoReceive":
        # 受給開始年齢の判定
        if "_60" in scenario:
          pension_start_age = 60
        elif "_65" in scenario:
          pension_start_age = 65
        elif "_70" in scenario:
          pension_start_age = 70
        elif "_75" in scenario:
          pension_start_age = 75
        else:
          # ここには来ないはずだがデフォルト
          pension_start_age = 65

        start_month = max(0, (pension_start_age - init_age) * 12)

        # 受給額倍率の計算
        if pension_start_age < 65:
          # 繰り上げ (0.4% / 月 減額)
          reduction_rate = 1.0 - 0.004 * (65 - pension_start_age) * 12
        else:
          # 繰り下げ (0.7% / 月 増額)
          reduction_rate = 1.0 + 0.007 * (pension_start_age - 65) * 12
        
        # 厚生年金 (22歳からリタイア開始年齢 N まで加入と想定)
        kousei_annual_nominal = KOUSEI_UNIT_ANNUAL * (init_age - 22) * reduction_rate
        if kousei_annual_nominal > 0:
          cf_name = f"Kousei_{i}"
          cf_configs.append(PensionConfig(
              name=cf_name,
              amount=kousei_annual_nominal / 12.0,
              start_month=start_month,
              cpi_name=CPI_NAME
          ))
          extra_cf_names.append(cf_name)

        # 基礎年金
        if "Pay" in scenario:
          # 満額受給
          kiso_annual_nominal = KISO_FULL_ANNUAL * reduction_rate
        elif "Exempt" in scenario:
          # 全額免除期間あり (Nから60歳まで免除)
          kiso_annual_nominal = (KISO_FULL_ANNUAL * (init_age - 22) / 40.0 +
                                 KISO_FULL_ANNUAL * (60 - init_age) / 40.0 * 0.5) * reduction_rate
        elif "Unpaid" in scenario:
          # 未納 (Nから60歳まで未納)
          kiso_annual_nominal = (KISO_FULL_ANNUAL * (init_age - 22) / 40.0) * reduction_rate
        else:
          raise ValueError(f"Unknown scenario for Kiso calculation: {scenario}")

        if kiso_annual_nominal > 0:
          cf_name = f"Kiso_{i}"
          cf_configs.append(PensionConfig(
              name=cf_name,
              amount=kiso_annual_nominal / 12.0,
              start_month=start_month,
              cpi_name=PENSION_CPI_NAME
          ))
          extra_cf_names.append(cf_name)

    # 1. キャッシュフロールールの定義 (基本支出)
    spend_config = BaseSpendConfig(name="生活費",
                                   amount=current_annual_cost,
                                   cpi_name=CPI_NAME)
    cf_configs.append(spend_config)

    # キャッシュフロー生成
    monthly_cashflows = generate_cashflows(cf_configs,
                                           monthly_prices,
                                           n_sim=N_SIM,
                                           n_months=YEARS * 12)

    # 戦略
    # TODO: INCLUDE_IN_ANNUAL_SPEND を使うべきだが、backward-compatibility のために ISOLATED を今は使う
    # 生活費は REGULAR に設定
    cf_rules = [
        CashflowRule(source_name=spend_config.name,
                     cashflow_type=CashflowType.REGULAR)
    ]
    cf_rules.extend([
        CashflowRule(source_name=name, cashflow_type=CashflowType.EXTRAORDINARY)
        for name in extra_cf_names
    ])

    strategy = Strategy(name=f"Pattern_{i}",
                        initial_money=float(init_money),
                        initial_loan=0.0,
                        yearly_loan_interest=0.0,
                        initial_asset_ratio={"オルカン": 1.0},
                        selling_priority=["オルカン"],
                        cashflow_rules=cf_rules)

    # シミュレーション
    res = simulate_strategy(strategy,
                            monthly_prices,
                            monthly_cashflows=monthly_cashflows)

    # 生存確率の記録
    row = {
        "initial_money": init_money,
        "initial_annual_cost": annual_cost,
        "initial_age": init_age,
        "scenario": scenario,
        "pension_start_age": pension_start_age,
        "initial_pension_nominal_annual": kousei_annual_nominal + kiso_annual_nominal
    }
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row[str(year)] = survival_rate
    
    results.append(row)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {csv_path} に保存しました。")

if __name__ == "__main__":
  main()
