"""
このスクリプトは、異なる支出率と目標年数の組み合わせに対して、
固定最適比率、ダイナミック最適比率、および(110-年齢)ルールの生存確率を計算します。
"""

import os
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from core import (Asset, Forex, Strategy, ZeroRiskAsset, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)
from dynamic_rebalance import calculate_optimal_strategy


def main():
  # 為替の定義
  forexes = [Forex(name="USDJPY", mu=0.0, sigma=0.1053)]
  forex_paths = generate_forex_paths(forexes)

  # 資産の定義 (オルカン)
  asset_name = "オルカン"
  assets_def = [
      Asset(name=asset_name,
            trust_fee=0.0005775,
            mu=0.07,
            sigma=0.15,
            leverage=1,
            forex="USDJPY"),
  ]

  print("月次価格推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets_def,
                                                       forex_paths=forex_paths)

  # 無リスク資産の定義 (4%)
  rf_name = "無リスク資産(4%)"
  rf = ZeroRiskAsset(name=rf_name, yield_rate=0.04)

  spending_rates = [
      (0.0666666, "6.67% (x15)"),
      (0.05, "5.0% (x20)"),
      (0.04, "4.0% (x25)"),
      (0.035714, "3.57% (x28)"),
      (0.033333, "3.33% (x30)"),
      (0.03, "3.0% (x33)"),
      (0.028571, "2.86% (x35)"),
      (0.025, "2.5% (x40)"),
      (0.022222, "2.22% (x45)"),
      (0.02, "2.0% (x50)"),
  ]
  target_years = [10, 20, 30, 40, 50]

  # spending_rates = [(0.05, "5.0% (x20)")]
  # target_years = [50]

  initial_money = 10000.0

  results: list[dict[str, Any]] = []

  print("各戦略のシミュレーションを実行中...")

  # "110 - 年齢" ルール用のコールバック関数を生成
  def make_age_rule_fn(start_age: int, target_N: int):

    def fn(net_value: np.ndarray, annual_spend: np.ndarray,
           remaining_years: float) -> Dict[str, np.ndarray]:
      elapsed_years = target_N - remaining_years
      current_age = start_age + elapsed_years
      ratio = max(0.0, min(1.0, (110 - current_age) / 100.0))
      ratio_array = np.full_like(net_value, ratio)
      return {asset_name: ratio_array, rf_name: 1.0 - ratio_array}

    return fn

  # ダイナミック最適比率用のコールバック関数を生成
  def make_dynamic_optimal_fn():

    def fn(net_value: np.ndarray, annual_spend: np.ndarray,
           remaining_years: float) -> Dict[str, np.ndarray]:
      # 純資産が0以下になる場合のゼロ除算を防ぐ
      safe_net_value = np.maximum(net_value, 1e-10)
      S = annual_spend / safe_net_value

      ratio_array = calculate_optimal_strategy(S, remaining_years)
      return {asset_name: ratio_array, rf_name: 1.0 - ratio_array}

    return fn

  for target_N in target_years:
    for spending_rate, spending_label in spending_rates:
      annual_cost = initial_money * spending_rate

      # 固定最適比率 (初期の S と N で計算)
      initial_S = np.array([spending_rate])
      fixed_ratio = calculate_optimal_strategy(initial_S, float(target_N))[0]

      strategies_to_test = [
          ("固定最適比率", None, fixed_ratio),
          ("ダイナミック最適比率", make_dynamic_optimal_fn(), None),
          ("110-年齢 (30歳開始)", make_age_rule_fn(30, target_N), None),
          ("110-年齢 (40歳開始)", make_age_rule_fn(40, target_N), None),
          ("110-年齢 (50歳開始)", make_age_rule_fn(50, target_N), None),
          ("110-年齢 (60歳開始)", make_age_rule_fn(60, target_N), None),
      ]

      for strategy_name, dynamic_fn, fixed_ratio_val in strategies_to_test:

        initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float] = {}

        if fixed_ratio_val is not None:
          initial_asset_ratio[asset_name] = float(fixed_ratio_val)
          initial_asset_ratio[rf] = float(1.0 - fixed_ratio_val)
        else:
          # ダイナミックな場合、初期比率は時間0で計算されるべきである
          # 年齢ルールの場合は、開始時の年齢を使用する
          if "年齢" in strategy_name:
            start_age = int(strategy_name.split("(")[1].split("歳")[0])
            r = max(0.0, min(1.0, (110 - start_age) / 100.0))
            initial_asset_ratio[asset_name] = float(r)
            initial_asset_ratio[rf] = float(1.0 - r)
          else:
            initial_asset_ratio[asset_name] = float(
                calculate_optimal_strategy(np.array([spending_rate]),
                                           float(target_N))[0])
            initial_asset_ratio[rf] = float(1.0 -
                                            initial_asset_ratio[asset_name])

        strategy = Strategy(name=strategy_name,
                            initial_money=initial_money,
                            initial_loan=0.0,
                            yearly_loan_interest=0.0,
                            initial_asset_ratio=initial_asset_ratio,
                            annual_cost=annual_cost,
                            inflation_rate=0.02,
                            tax_rate=0.20315,
                            selling_priority=[rf.name, asset_name],
                            rebalance_interval=12,
                            dynamic_rebalance_fn=dynamic_fn)

        res = simulate_strategy(strategy, monthly_asset_prices)

        row: dict[str, Any] = {
            "target_years": target_N,
            "spend_ratio": spending_rate,
            "strategy": strategy_name
        }
        for year in range(1, 51):
          bankrupt_count = (res.sustained_months < year * 12).sum()
          survival_rate = 1.0 - (bankrupt_count / len(res.sustained_months))
          row[str(year)] = survival_rate

        results.append(row)

  df = pd.DataFrame(results)
  os.makedirs("data", exist_ok=True)
  csv_path = "data/dynamic_rebalance_comp.csv"
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"✅ {csv_path} に保存しました。")
  print(df.head())


if __name__ == "__main__":
  main()
