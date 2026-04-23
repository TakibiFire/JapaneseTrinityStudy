"""
支出率と経過年数に応じた生存確率の比較シミュレーション（新エンジン版）。

このスクリプトは、異なる支出率と目標年数の組み合わせに対して、
以下の戦略による生存確率を計算し、比較結果を CSV で出力します。

1. 固定最適比率 (Fixed Optimal Ratio):
   シミュレーション開始時の支出率(S)と目標年数(N)から導き出された
   「その時点での最適比率」を全期間維持する戦略。
2. ダイナミック最適比率 (Dynamic Optimal Ratio):
   毎年のリバランス時に、その時点の残り年数(N')と現在の支出率(S')を再計算し、
   それに基づきポートフォリオの比率を動的に変更する戦略。
3. (110 - 年齢) ルール:
   リスク資産（オルカン）の比率を (110 - 現在の年齢)% とする一般的なルール。

主な設定:
- インフレ率: 年率 1.77% (固定)
- 無リスク資産: 利回り 4.0%
- 税率: 20.315%
- リバランス: 毎年
- 売却順序: 無リスク資産を優先的に売却し、不足分をオルカンから補う。

出力ファイル:
- data/dynamic_rebalance_summary.csv: 各条件における各戦略の生存確率をまとめた表。

計算ロジックの詳細は `src/lib/dynamic_rebalance.py` および `docs/dynamic_rebalance.md` を参照してください。
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.core import Strategy, ZeroRiskAsset, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.dynamic_rebalance import calculate_optimal_strategy


def main():
  # シミュレーション設定
  n_sim = 5000
  max_years = 50
  seed = 42
  initial_money = 10000.0  # 万円
  tax_rate = 0.20315
  inflation_rate_val = 0.0177
  fee_acwi = 0.0005775
  zero_risk_yield = 0.04

  # 共通アセット名
  cpi_name = "Japan_CPI_1.77pct"
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  zr_name = "無リスク資産(4%)"

  # 1. 価格推移の生成
  # オルカン: 期待リターン 7%, リスク 15%
  # 為替: 期待リターン 0%, リスク 10.53%
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name, dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)),
      Asset(name=acwi_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=fee_acwi,
            forex=fx_name),
      CpiAsset(name=cpi_name,
               dist=YearlyLogNormalArithmetic(mu=inflation_rate_val, sigma=0.0))
  ]

  print(f"月次価格推移を生成中 (n_sim={n_sim}, years={max_years})...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=max_years * 12,
                                                       seed=seed)

  zr_asset = ZeroRiskAsset(name=zr_name, yield_rate=zero_risk_yield)

  # 比較対象の支出率と目標年数
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

  results: List[Dict[str, Any]] = []

  # --- 戦略別のコールバック関数生成 ---

  # ダイナミック最適比率用のコールバック
  def dynamic_optimal_fn(net_value: np.ndarray, annual_spend: np.ndarray,
                         remaining_years: float,
                         post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    # 純資産が0以下になる場合のゼロ除算を防ぐ
    safe_net_value = np.maximum(net_value, 1e-10)
    s_rate = annual_spend / safe_net_value
    ratio_array = calculate_optimal_strategy(s_rate, remaining_years,
                                             base_yield=zero_risk_yield,
                                             tax_rate=tax_rate,
                                             inflation_rate=inflation_rate_val)
    return {acwi_name: ratio_array, zr_name: 1.0 - ratio_array}

  # (110 - 年齢) ルール用のコールバック
  def make_age_rule_fn(start_age: int, total_years: int):
    def fn(net_value: np.ndarray, annual_spend: np.ndarray,
           remaining_years: float,
           post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
      elapsed_years = total_years - remaining_years
      current_age = start_age + elapsed_years
      ratio = max(0.0, min(1.0, (110 - current_age) / 100.0))
      return {acwi_name: ratio, zr_name: 1.0 - ratio}
    return fn

  print("各支出率と目標年数の組み合わせでシミュレーションを実行中...")
  for target_n in target_years:
    print(f"目標年数: {target_n}年")
    for spending_rate, spending_label in spending_rates:
      annual_cost = initial_money * spending_rate

      # 初期時点での最適比率を計算 (固定最適比率用)
      initial_s_arr = np.array([spending_rate])
      fixed_ratio_val = calculate_optimal_strategy(initial_s_arr, float(target_n),
                                                   base_yield=zero_risk_yield,
                                                   tax_rate=tax_rate,
                                                   inflation_rate=inflation_rate_val)[0]

      # テストする戦略のリスト
      test_cases = [
          ("固定最適比率", None, fixed_ratio_val),
          ("ダイナミック最適比率", dynamic_optimal_fn, None),
          ("110-年齢 (30歳開始)", make_age_rule_fn(30, target_n), None),
          ("110-年齢 (40歳開始)", make_age_rule_fn(40, target_n), None),
          ("110-年齢 (50歳開始)", make_age_rule_fn(50, target_n), None),
          ("110-年齢 (60歳開始)", make_age_rule_fn(60, target_n), None),
      ]

      print(f"  - 支出率: {spending_label}")
      for strategy_name, dynamic_fn, fixed_ratio in test_cases:
        # 初期資産配分の設定
        initial_ratio: Dict[Union[str, ZeroRiskAsset], float] = {}
        if fixed_ratio is not None:
          # 固定比率戦略
          initial_ratio[acwi_name] = float(fixed_ratio)
          initial_ratio[zr_asset] = 1.0 - float(fixed_ratio)
        else:
          # ダイナミック戦略の初期値を決定
          if "年齢" in strategy_name:
            # 年齢ルールの初期年齢から比率を計算
            start_age_str = strategy_name.split("(")[1].split("歳")[0]
            start_age = int(start_age_str)
            r = max(0.0, min(1.0, (110 - start_age) / 100.0))
            initial_ratio[acwi_name] = r
            initial_ratio[zr_asset] = 1.0 - r
          else:
            # ダイナミック最適比率の初期時点(T=0)での比率
            r = calculate_optimal_strategy(np.array([spending_rate]), float(target_n),
                                           base_yield=zero_risk_yield,
                                           tax_rate=tax_rate,
                                           inflation_rate=inflation_rate_val)[0]
            initial_ratio[acwi_name] = float(r)
            initial_ratio[zr_asset] = 1.0 - float(r)

        # 戦略の構築
        strategy = Strategy(
            name=strategy_name,
            initial_money=initial_money,
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio=initial_ratio,
            annual_cost=annual_cost,
            inflation_rate=cpi_name,
            tax_rate=tax_rate,
            selling_priority=[zr_name, acwi_name],
            rebalance_interval=12,
            dynamic_rebalance_fn=dynamic_fn
        )

        # target_n 年分だけスライスして渡す
        # 注意: monthly_asset_prices をそのまま渡すと、simulate_strategy 内で
        # total_months が 50 年 (600ヶ月) と解釈されてしまい、
        # ダイナミックリバランスの残り年数計算 (rem_years = (total_months - m) / 12) が
        # 狂ってしまう (30年目標なのに 50年残っていると判定されて過剰にリスクを取る)。
        # これを防ぐため、シミュレーション対象の target_n 年分だけをスライスして渡す。
        sliced_prices = {k: v[:, :target_n * 12 + 1] for k, v in monthly_asset_prices.items()}

        # シミュレーション実行
        res = simulate_strategy(strategy, sliced_prices)
        
        # target_n 年時点での生存確率を計算
        bankrupt_count = (res.sustained_months < target_n * 12).sum()
        survival_rate = 1.0 - (bankrupt_count / n_sim)

        results.append({
            "target_years": target_n,
            "spend_ratio": spending_rate,
            "strategy": strategy_name,
            "survival_probability": survival_rate
        })

  # 全結果をデータフレームにまとめ、CSV 出力
  df = pd.DataFrame(results)
  os.makedirs("data", exist_ok=True)
  output_path = "data/dynamic_rebalance_summary.csv"
  df.to_csv(output_path, index=False, encoding="utf-8-sig")
  print(f"✅ {output_path} に結果を保存しました。")


if __name__ == "__main__":
  main()
