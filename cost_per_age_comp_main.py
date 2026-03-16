"""
年齢ごとの支出変動を考慮した場合の4%ルールの生存確率へ与える影響を比較するシミュレーション。
"""

import os
import re
import sys

import numpy as np
from scipy.interpolate import CubicSpline

from core import (Asset, Forex, Strategy, ZeroRiskAsset, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def get_cost_multiplier(start_age, num_years=50):
  """
  analyze_cost_main.py のロジックを用いて、指定された開始年齢からの生活費の乗数リストを返す。
  乗数は開始年齢の時の生活費を 1.0 とした相対的な値。
  """
  ages = np.array([34.4, 44.8, 54.2, 64.6, 77.6, 85.2])
  costs = np.array([280451, 331134, 356946, 311392, 252781, 221056])

  # 3次スプライン補間 (自然スプライン)
  cs = CubicSpline(ages, costs, bc_type='natural')

  # 推計対象の年齢 (開始年齢からnum_years年分)
  # たとえば start_age=60 で num_years=50 の場合、年齢は 60 から 109 となる。
  target_ages = np.arange(start_age, start_age + num_years)
  target_costs = cs(target_ages)

  # 85.2歳付近の221,056円で保守的に下限クリップする。
  # これにより、85.2歳を超えた年齢（例：105歳など）に対しても、
  # スプライン曲線の不自然な下降を防ぎ、221,056円の平坦な消費支出が維持される。
  target_costs = np.maximum(target_costs, 221056)

  # 開始年齢のコストで割って乗数にする
  multipliers = target_costs / target_costs[0]

  return multipliers.tolist()


def main():
  # ---------------------------------------------------------------------------
  # 1. 為替レートと資産の定義
  # ---------------------------------------------------------------------------
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
  rf = ZeroRiskAsset(name="無リスク資産(4%)", yield_rate=0.04)

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  ratio = 0.8
  base_annual_cost = 400

  strategies = []

  strategies.append(
      Strategy(name="1. 支出一定 (400万円)",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={
                   asset_name: ratio,
                   rf: (1.0 - ratio)
               },
               annual_cost=base_annual_cost,
               inflation_rate=0.02,
               tax_rate=0.20315,
               selling_priority=[rf.name, asset_name],
               rebalance_interval=12))

  start_ages = [30, 35, 40, 45, 50, 55, 60]

  for i, start_age in enumerate(start_ages):
    multipliers = get_cost_multiplier(start_age, 50)
    print(f"[{start_age}歳からの50年間] 出費乗数: {[round(m, 2) for m in multipliers[:5]]} ... {[round(m, 2) for m in multipliers[-5:]]}")
    annual_costs = [base_annual_cost * m for m in multipliers]

    strategies.append(
        Strategy(name=f"{i+2}. {start_age}歳からの50年間",
                 initial_money=10000,
                 initial_loan=0,
                 yearly_loan_interest=0,
                 initial_asset_ratio={
                     asset_name: ratio,
                     rf: (1.0 - ratio)
                 },
                 annual_cost=annual_costs,
                 inflation_rate=0.02,
                 tax_rate=0.20315,
                 selling_priority=[rf.name, asset_name],
                 rebalance_interval=12))

  # ---------------------------------------------------------------------------
  # 3. シミュレーションの実行
  # ---------------------------------------------------------------------------
  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 4. 可視化と保存
  # ---------------------------------------------------------------------------
  survival_image_file = 'imgs/cost_per_age_comp_survival.svg'
  distribution_image_file = 'imgs/cost_per_age_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/cost_per_age_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='年齢別支出の影響',
                     summary_title='年齢別支出の影響サマリー（1,000回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50])

  # ---------------------------------------------------------------------------
  # 5. Markdown レポートの更新
  # ---------------------------------------------------------------------------
  bankruptcy_years = [10, 20, 30, 40, 50]
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=bankruptcy_years)

  md_text = formatted_df.to_markdown(colalign=("left",) +
                                     ("right",) * len(formatted_df.columns))

  report_file = 'docs/report.md'
  try:
    with open(report_file, 'r', encoding='utf-8') as f:
      content = f.read()

    pattern = r'(<!--<cost_per_age_comp_main\.py>-->).*?(<!--</cost_per_age_comp_main\.py>-->)'
    if re.search(pattern, content, re.DOTALL):
      insert_text = f'\n\n{md_text.strip()}\n\n'
      new_content = re.sub(pattern,
                           rf'\g<1>{insert_text}\g<2>',
                           content,
                           flags=re.DOTALL)
      with open(report_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
      print(f"✅ {report_file} を更新しました。")
    else:
      print(
          f"\033[91mWarning: Placeholder <cost_per_age_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
