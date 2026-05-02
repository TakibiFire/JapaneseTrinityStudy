"""
40歳からの55年間の資産運用・取り崩しシミュレーションを行い、
新旧のリバランス戦略（V1 vs V2/DP）を比較するグリッドサーチスクリプト。

実験設定:
- 期間: 55年 (40歳〜95歳)
- 試行回数: 5,000回
- 資産構成: FX, ACWI (fat tail), CPI, Pension CPI (slide_rate=0.005)
- 世帯設定: 1人世帯, 年金受給開始60歳 (H1_P60)
- 比較戦略:
    1. オルカン100%
    2. 無リスク100%
    3. 固定最適比率 (Fixed Optimal Ratio)
    4. 一般的な最適リバランス (Previous V1)
    5. 支出に合わせた最適リバランス (Previous V2/DP)
"""

import argparse
import os
from dataclasses import replace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.scenario_builder import (DynamicV1Rebalance, FixedRebalance,
                                      PredefinedStock, PredefinedZeroRisk,
                                      SpendAwareDPRebalance, StrategySpec,
                                      create_experiment_setup)
from src.lib.world_setup import create_standard_world


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

  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. セットアップの構築
  setup = create_standard_world(n_sim=N_SIM,
                                start_age=START_AGE,
                                end_age=START_AGE + YEARS - 1,
                                retirement_start_age=40,
                                pension_start_age=60,
                                seed=SEED)

  # 2. グリッドループ
  spending_rules = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
  strategies_to_compare = [
      "オルカン100%", "無リスク100%", "固定最適比率", "一般的な最適リバランス", "支出に合わせた最適リバランス"
  ]

  # 基本支出（統計値）を取得するために、一度コンパイルして情報を抽出する
  # (モデルフィッティングとの整合性を保つためのリファレンスとして使用)
  base_exp = create_experiment_setup(setup)[0]
  # annual_cost_real is in man-yen/year. [0] is the start_age value.
  initial_annual_cost_man_yen = base_exp.annual_cost_real[0]

  # dump_withdraw モードの処理
  if EXP_NAME == "dump_withdraw":
    print("dump_withdraw モード: キャッシュフローを解析して支出額をダンプします。")
    total_months = YEARS * 12
    monthly_cashflows = base_exp.monthly_cashflows

    # キャッシュフローの合算 (名目、万円)
    # monthly_cashflows には負の値（支出）と正の値（収入）の両方が含まれている
    total_cf_m = np.zeros((N_SIM, total_months), dtype=np.float64)
    for cf_arr in monthly_cashflows.values():
      total_cf_m += cf_arr

    # 取り崩し額 = - 純キャッシュフロー (正の値が引き出しを表す)
    withdraw_m = -total_cf_m

    # 年次集計 (万円/年)
    withdraw_y = np.zeros((N_SIM, YEARS), dtype=np.float64)
    for y in range(YEARS):
      withdraw_y[:, y] = withdraw_m[:, y * 12:(y + 1) * 12].sum(axis=1)

    # パーセンタイル
    p25 = np.percentile(withdraw_y, 25, axis=0)
    p50 = np.percentile(withdraw_y, 50, axis=0)
    p75 = np.percentile(withdraw_y, 75, axis=0)

    results_dump = []
    for name, p_values in [("spend25p", p25), ("spend50p", p50),
                           ("spend75p", p75)]:
      row = {
          "spend_multiplier": 1.0,
          "strategy": "dump_withdraw",
          "spending_rule": 0.0,
          "initial_money": 0.0,
          "initial_annual_cost": 0.0,
          "value_type": name
      }
      for y in range(YEARS):
        row[str(y + 1)] = p_values[y]
      results_dump.append(row)

    df_dump = pd.DataFrame(results_dump)
    df_dump.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"完了。結果を {CSV_PATH} に保存しました。")
    return

  # グリッドシミュレーション用の実験を Setup に追加
  for rule in spending_rules:
    # 初期資産の計算
    init_money = initial_annual_cost_man_yen / (rule / 100.0)

    for strat_name in strategies_to_compare:
      spec = StrategySpec(
          initial_money=init_money,
          initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),
                               (PredefinedZeroRisk.ZERO_RISK_4PCT, 0.0)),
          selling_priority=(PredefinedStock.ORUKAN_155,
                            PredefinedZeroRisk.ZERO_RISK_4PCT))

      if strat_name == "オルカン100%":
        spec = replace(spec, rebalance=FixedRebalance())
      elif strat_name == "無リスク100%":
        spec = replace(spec,
                       initial_asset_ratio=((PredefinedStock.ORUKAN_155, 0.0),
                                            (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                             1.0)),
                       rebalance=FixedRebalance())
      elif strat_name == "固定最適比率":
        # FixedRebalance で初期配分を維持。
        # 比率自体は内部で calculate_optimal_strategy を呼ぶ V1Rebalance か、
        # あるいはここで計算して initial_asset_ratio に設定する。
        # 以前のスクリプトの挙動（rule に基づく固定比率）を再現するため、ここで計算。
        from src.lib.dynamic_rebalance import calculate_optimal_strategy
        fixed_ratio = calculate_optimal_strategy(s_rate=np.array([rule / 100.0
                                                                 ]),
                                                 remaining_years=YEARS,
                                                 base_yield=0.04,
                                                 tax_rate=0.20315,
                                                 inflation_rate=0.0177)[0]
        spec = replace(spec,
                       initial_asset_ratio=((PredefinedStock.ORUKAN_155,
                                             fixed_ratio),
                                            (PredefinedZeroRisk.ZERO_RISK_4PCT,
                                             1.0 - fixed_ratio)),
                       rebalance=FixedRebalance())
      elif strat_name == "一般的な最適リバランス":
        spec = replace(spec,
                       rebalance=DynamicV1Rebalance(
                           risky_asset=PredefinedStock.ORUKAN_155,
                           zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT))
      elif strat_name == "支出に合わせた最適リバランス":
        spec = replace(spec,
                       rebalance=SpendAwareDPRebalance(
                           risky_asset=PredefinedStock.ORUKAN_155,
                           zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                           model_name=MODELS_PATH))

      setup.add_experiment(name=f"{strat_name}_Rule{rule}",
                           overwrite_strategy=spec)

  # 実験の実行
  print(f"全 {len(setup.experiments)} パターンのシミュレーションをコンパイル中...")
  compiled_exps = create_experiment_setup(setup, record_annual_spend=True)

  results: List[Dict[str, Any]] = []
  total_its = len(compiled_exps) - 1  # ベースラインを除く
  print(f"シミュレーション実行中...")

  for i, exp in enumerate(compiled_exps):
    if i == 0:
      continue  # ベースライン (standard_world) はスキップ

    if i % 10 == 0:
      print(f"Progress: {i}/{total_its}")

    # 実験名から戦略名とルールを復元 (パース)
    # 形式: "{strat_name}_Rule{rule}"
    name_parts = exp.name.split("_Rule")
    strat_name = name_parts[0]
    rule = float(name_parts[1])

    res = simulate_strategy(exp.strategy, exp.monthly_prices,
                            exp.monthly_cashflows)

    # 結果の記録 (共通項目)
    base_row = {
        "spend_multiplier": 1.0,
        "strategy": strat_name,
        "spending_rule": rule,
        "initial_money": exp.strategy.initial_money,
        "initial_annual_cost": initial_annual_cost_man_yen,
    }

    # 1. 生存確率
    row_survival = base_row.copy()
    row_survival["value_type"] = "survival"
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row_survival[str(year)] = survival_rate
    results.append(row_survival)

    # 2. 支出額の統計 (特定の条件のみ記録)
    if res.annual_spends is not None and rule == 4.0 and strat_name == "支出に合わせた最適リバランス":
      p25 = np.percentile(res.annual_spends, 25, axis=0)
      p50 = np.percentile(res.annual_spends, 50, axis=0)
      p75 = np.percentile(res.annual_spends, 75, axis=0)

      for stat_name, p_values in [("spend25p", p25), ("spend50p", p50),
                                  ("spend75p", p75)]:
        row = base_row.copy()
        row["value_type"] = stat_name
        for year in range(1, YEARS + 1):
          row[str(year)] = p_values[year - 1]
        results.append(row)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {CSV_PATH} に保存しました。")


if __name__ == "__main__":
  main()
