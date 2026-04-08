"""
ダイナミック・スペンディング（動的支出）戦略の効果を比較するシミュレーション。
"""

import os
import re
import sys

from core import (Asset, DynamicSpending, Forex, Strategy, ZeroRiskAsset,
                  generate_forex_paths, generate_monthly_asset_prices,
                  simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


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

  strategies = []

  # ベースライン (支出一定 400万円, インフレ 2%)
  strategies.append(
      Strategy(name="1. 支出一定 (400万円, インフレ2%)",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={
                   asset_name: ratio,
                   rf: (1.0 - ratio)
               },
               annual_cost=400,
               inflation_rate=0.02,
               tax_rate=0.20315,
               selling_priority=[rf.name, asset_name],
               rebalance_interval=12))

  # DynamicSpending設定
  spending_configs = [
      (0.05, -0.015, "2. 動的支出 (上限+5%, 下限-1.5%)"),
      (0.02, -0.02, "3. 動的支出 (上限+2%, 下限-2%)"),
      (0.02, 0.0, "4. 動的支出 (上限+2%, 下限0%)"),
  ]

  for upper, lower, label in spending_configs:
    strategies.append(
        Strategy(
            name=label,
            initial_money=10000,
            initial_loan=0,
            yearly_loan_interest=0,
            initial_asset_ratio={
                asset_name: ratio,
                rf: (1.0 - ratio)
            },
            annual_cost=DynamicSpending(target_ratio=0.04,
                                        upper_limit=upper,
                                        lower_limit=lower),
            inflation_rate=0.0,  # DynamicSpending requires 0.0 or None
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
  survival_image_file = 'docs/imgs/dynamic_spending_comp_survival.svg'
  distribution_image_file = 'docs/imgs/dynamic_spending_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/dynamic_spending_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='ダイナミック・スペンディングの効果',
                     summary_title='ダイナミック・スペンディングのサマリー（1,000回試行）',
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

    pattern = r'(<!--<dynamic_spending_comp_main\.py>-->).*?(<!--</dynamic_spending_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <dynamic_spending_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
