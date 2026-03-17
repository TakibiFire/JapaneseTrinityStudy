"""
資産を現金化する順番（オルカン優先 vs 無リスク資産優先）による4%ルールの生存確率への影響を比較するシミュレーション。
"""

import os
import re
import sys

from core import (Asset, Forex, Strategy, ZeroRiskAsset, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)
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
      Asset(name=asset_name, trust_fee=0.0005775, mu=0.07, sigma=0.15, leverage=1, forex="USDJPY"),
  ]

  print("月次価格推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets_def, forex_paths=forex_paths)

  # 無リスク資産の定義 (4%)
  rf = ZeroRiskAsset(name="無リスク資産(4%)", yield_rate=0.04)

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  stock_ratios = [0.8, 0.7, 0.5]
  
  strategies = []
  
  # Baseline: 100% Stock
  strategies.append(
      Strategy(
          name="オルカン 100% (基準)",
          initial_money=10000,
          initial_loan=0,
          yearly_loan_interest=0,
          initial_asset_ratio={asset_name: 1.0},
          annual_cost=400,
          inflation_rate=0.02,
          tax_rate=0.20315,
          selling_priority=[asset_name],
          rebalance_interval=0
      )
  )

  for ratio in stock_ratios:
    # 1. Sell Stock then RF
    strategies.append(
        Strategy(
            name=f"オルカン {round(ratio*100)}% (売却順: 1.株, 2.無リスク)",
            initial_money=10000,
            initial_loan=0,
            yearly_loan_interest=0,
            initial_asset_ratio={asset_name: ratio, rf: (1.0 - ratio)},
            annual_cost=400,
            inflation_rate=0.02,
            tax_rate=0.20315,
            selling_priority=[asset_name, rf.name],
            rebalance_interval=0
        )
    )
    # 2. Sell RF then Stock
    strategies.append(
        Strategy(
            name=f"オルカン {round(ratio*100)}% (売却順: 1.無リスク, 2.株)",
            initial_money=10000,
            initial_loan=0,
            yearly_loan_interest=0,
            initial_asset_ratio={asset_name: ratio, rf: (1.0 - ratio)},
            annual_cost=400,
            inflation_rate=0.02,
            tax_rate=0.20315,
            selling_priority=[rf.name, asset_name],
            rebalance_interval=0
        )
    )

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
  survival_image_file = 'imgs/zero_risk_priority_comp_survival.svg'
  distribution_image_file = 'imgs/zero_risk_priority_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/zero_risk_priority_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='資産売却の優先順位による生存確率の比較',
                     summary_title='売却順序の影響サマリー（1,000回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     survival_height=400)

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

    pattern = r'(<!--<zero_risk_priority_comp_main\.py>-->).*?(<!--</zero_risk_priority_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <zero_risk_priority_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
