"""
このスクリプトは、特定のDynamic Spending設定（lower=-1.5%, upper=5%）と
ダイナミックリバランスを組み合わせた場合の、名目支出額の分布を計算し、
ヒストグラムを作成します。

設定詳細:
- 初期資産: 1億円 (10,000万円)
- 初期支出: 400万円 (4%)
- インフレ率: 1.77%
- Dynamic Spending: 下限 -1.5%, 上限 5%
- ダイナミックリバランス: 有効
"""

import os
from typing import Dict, List, Union

import altair as alt
import numpy as np
import pandas as pd

from src.core import (DynamicSpending, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (Asset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.dynamic_rebalance import calculate_optimal_strategy


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  seed = 42
  initial_money = 10000
  tax_rate = 0.20315
  fee_acwi = 0.0005775
  zero_risk_yield = 0.04
  inflation_rate = 0.0177  # 実質価値の計算用
  
  # Dynamic Spending 設定 (Vanguard style)
  upper = 0.05
  lower = -0.015
  target_ratio = 0.04
  initial_spend = initial_money * target_ratio

  # 共通アセット名
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  zr_name = "無リスク資産(4%)"

  # 1. 価格推移の生成
  assets: List[Union[Asset, ForexAsset]] = [
      ForexAsset(name=fx_name,
                 dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)),
      Asset(name=acwi_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=fee_acwi,
            forex=fx_name)
  ]

  print(f"月次価格の推移を生成中... (試行回数: {n_sim}, 期間: {years}年)")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=years * 12,
                                                       seed=seed)

  zr_asset = ZeroRiskAsset(name=zr_name, yield_rate=zero_risk_yield)

  # ダイナミック最適比率用のコールバック
  def dynamic_optimal_fn(net_value: np.ndarray, annual_spend: np.ndarray,
                         remaining_years: float,
                         post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    safe_net_value = np.maximum(net_value, 1e-10)
    s_rate = annual_spend / safe_net_value
    ratio_array = calculate_optimal_strategy(s_rate,
                                             remaining_years,
                                             base_yield=zero_risk_yield,
                                             tax_rate=tax_rate,
                                             inflation_rate=inflation_rate)
    return {acwi_name: ratio_array, zr_name: 1.0 - ratio_array}

  # 初期比率の計算
  initial_r = 1.0
  initial_ratio: Dict[Union[str, ZeroRiskAsset], float] = {acwi_name: float(initial_r), zr_asset: 1.0 - float(initial_r)}
  print(f"Initial Optimal ACWI Ratio: {initial_r:.2%}")
  print(f"Initial Zero Risk Ratio: {1.0 - float(initial_r):.2%}")

  strategy = Strategy(
      name=f"DynamicSpending_U{upper}_L{lower}",
      initial_money=initial_money,
      initial_loan=0,
      yearly_loan_interest=0,
      initial_asset_ratio=initial_ratio,
      annual_cost=DynamicSpending(initial_annual_spend=initial_spend,
                                  target_ratio=target_ratio,
                                  upper_limit=upper,
                                  lower_limit=lower),
      inflation_rate=0.0,
      tax_rate=tax_rate,
      selling_priority=[zr_name, acwi_name],
      rebalance_interval=12,
      dynamic_rebalance_fn=dynamic_optimal_fn,
      record_annual_spend=True)

  print("シミュレーションを実行中...")
  res = simulate_strategy(strategy, monthly_asset_prices)

  # 生存確率の確認
  bankrupt_count = (res.sustained_months < years * 12).sum()
  survival_rate = 1.0 - (bankrupt_count / n_sim)
  print(f"Success Rate: {survival_rate:.2%}")
  if abs(survival_rate - 0.753) > 0.01:
      print(f"WARNING: Expected Success Rate is around 75.3%, but got {survival_rate:.2%}")

  # 年次支出額の取得 (n_sim, years)
  nominal_spends = res.annual_spends
  if nominal_spends is None:
      raise ValueError("annual_spends is None. Check if record_annual_spend=True is working.")

  # 実質支出額の計算 (Real Purchasing Power)
  real_spends = np.zeros_like(nominal_spends)
  for y in range(years):
      # Year y+1 (index y)
      real_spends[:, y] = nominal_spends[:, y] / ((1.0 + inflation_rate) ** y)

  # 指定年のパーセンタイル計算 (生存パスのみ)
  target_years = [30, 40, 50]
  percentiles_to_print = [1, 5, 10, 25, 50, 75, 90, 95, 99]
  print("\n--- Spending Percentiles (万円) - Survivors Only ---")
  for y in target_years:
      vals_real = real_spends[:, y-1]
      vals_nom = nominal_spends[:, y-1]
      survivors_real = vals_real[vals_real > 0]
      survivors_nom = vals_nom[vals_nom > 0]
      bankrupt_rate = (vals_real == 0).mean()
      
      print(f"\nYear {y} ({bankrupt_rate:.1%} bankrupt paths excluded):")
      if len(survivors_real) > 0:
          for p in percentiles_to_print:
              p_real = np.percentile(survivors_real, p)
              p_nom = np.percentile(survivors_nom, p)
              print(f"  {p}th: {p_nom:.2f} (Nominal) / {p_real:.2f} (Real)")
      else:
          print("  No survivors.")

  # 可視化 (名目支出額の分布)
  os.makedirs("docs/imgs/dynamic_spending", exist_ok=True)
  
  for y in target_years:
      vals_nom = nominal_spends[:, y-1]
      bankrupt_rate = (vals_nom == 0).mean()
      survivors_nom = vals_nom[vals_nom > 0]
      valid_data = pd.DataFrame({'nominal_spend': survivors_nom})
      
      # 1.77%インフレ時の期待名目支出額
      expected_nominal = initial_spend * ((1.0 + inflation_rate) ** (y - 1))
      
      # パーセンタイル計算 (生存パス)
      p25 = np.percentile(survivors_nom, 25)
      p50 = np.percentile(survivors_nom, 50)
      p75 = np.percentile(survivors_nom, 75)
      
      # ヒストグラム
      hist = alt.Chart(valid_data).mark_bar(opacity=0.6).encode(
          alt.X("nominal_spend:Q", bin=alt.Bin(maxbins=100), title="年間支出額 (名目) (万円)"),
          alt.Y("count()", title="試行回数")
      )
      
      # 垂直線のデータ
      rules_data = pd.DataFrame([
          {'x': expected_nominal, 'label': '期待支出額 (1.77%インフレ考慮)'},
          {'x': p25, 'label': '25パーセンタイル (生存パス)'},
          {'x': p50, 'label': '50パーセンタイル (中央値)'},
          {'x': p75, 'label': '75パーセンタイル (生存パス)'}
      ])
      
      rules = alt.Chart(rules_data).mark_rule(size=2).encode(
          x='x:Q',
          color=alt.Color('label:N', title='', scale=alt.Scale(
              domain=['期待支出額 (1.77%インフレ考慮)', '25パーセンタイル (生存パス)', '50パーセンタイル (中央値)', '75パーセンタイル (生存パス)'],
              range=['red', 'blue', 'green', 'purple']
          )),
          strokeDash=alt.condition(
              alt.datum.label == '期待支出額 (1.77%インフレ考慮)',
              alt.value([5, 5]),
              alt.value([0])
          )
      )

      final_chart = (hist + rules).properties(
          title=f"名目支出額の分布: {y}年目 (上限{upper:.1%}, 下限{lower:.1%}) - 破産ケース {bankrupt_rate:.1%} を除外",
          width=600,
          height=300
      ).configure_legend(
          orient='top',
          titleOrient='left'
      )
      
      svg_path = f"docs/imgs/dynamic_spending/nominal_spend_hist_year_{y}.svg"
      try:
          import vl_convert as vlc
          svg_str = vlc.vegalite_to_svg(final_chart.to_json())
          with open(svg_path, "w") as f:
              f.write(svg_str)
          print(f"Saved {svg_path}")
      except (ImportError, Exception) as e:
          print(f"Could not save as SVG via vl-convert: {e}. Saving as HTML.")
          final_chart.save(svg_path.replace(".svg", ".html"))

if __name__ == "__main__":
  main()
