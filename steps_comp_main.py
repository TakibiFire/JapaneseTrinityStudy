"""
4%ルールの生存確率に対する各要因の影響を段階的に比較するシミュレーション。
"""

import os
import re
import sys

from core import (Asset, Forex, Strategy, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def main():
  # ---------------------------------------------------------------------------
  # 1. 為替レートと資産の定義
  # ---------------------------------------------------------------------------
  # 為替の定義 (Step 6 用)
  forexes = [Forex(name="USDJPY", mu=0.0, sigma=0.1053)]
  forex_paths = generate_forex_paths(forexes)

  # 資産の定義
  assets_def = [
      # Step 1: 7%固定 (ボラ0)
      Asset(name="Asset_Step1", trust_fee=0.0, mu=0.07, sigma=0.0, leverage=1),
      # Step 2-4: ボラ15%
      Asset(name="Asset_Step2-4",
            trust_fee=0.0,
            mu=0.07,
            sigma=0.15,
            leverage=1),
      # Step 5: 信託報酬あり
      Asset(name="Asset_Step5",
            trust_fee=0.0005775,
            mu=0.07,
            sigma=0.15,
            leverage=1),
      # Step 6: 為替リスクあり (Step 5 の設定を引き継ぐ)
      Asset(name="Asset_Step6",
            trust_fee=0.0005775,
            mu=0.07,
            sigma=0.15,
            leverage=1,
            forex="USDJPY"),
  ]

  print("月次価格推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets_def,
                                                       forex_paths=forex_paths)

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      Strategy(name="1. 7%固定運用 (リスク0)",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step1": 1.0},
               annual_cost=400,
               inflation_rate=0.0,
               tax_rate=0.0,
               selling_priority=["Asset_Step1"]),
      Strategy(name="2. ボラティリティ 15% を設定",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step2-4": 1.0},
               annual_cost=400,
               inflation_rate=0.0,
               tax_rate=0.0,
               selling_priority=["Asset_Step2-4"]),
      Strategy(name="3. 物価上昇率 2% を設定",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step2-4": 1.0},
               annual_cost=400,
               inflation_rate=0.02,
               tax_rate=0.0,
               selling_priority=["Asset_Step2-4"]),
      Strategy(name="4. 譲渡所得税 20.315% を設定",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step2-4": 1.0},
               annual_cost=400,
               inflation_rate=0.02,
               tax_rate=0.20315,
               selling_priority=["Asset_Step2-4"]),
      Strategy(name="5. 信託報酬 0.05775% を設定",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step5": 1.0},
               annual_cost=400,
               inflation_rate=0.02,
               tax_rate=0.20315,
               selling_priority=["Asset_Step5"]),
      Strategy(name="6. 為替リスク 10.5% を設定",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=0,
               initial_asset_ratio={"Asset_Step6": 1.0},
               annual_cost=400,
               inflation_rate=0.02,
               tax_rate=0.20315,
               selling_priority=["Asset_Step6"]),
  ]

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
  survival_image_file = 'docs/imgs/steps_comp_survival.svg'
  distribution_image_file = 'docs/imgs/steps_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/steps_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='各要因による生存確率への影響（段階的比較）',
                     summary_title='各要因の段階的影響サマリー（1,000回試行）',
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

    pattern = r'(<!--<steps_comp_main\.py>-->).*?(<!--</steps_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <steps_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
