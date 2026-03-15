"""
シンプルな4%ルールに基づく取り崩しシミュレーション。
オルカンと現金の3つのセットアップを比較し、生存確率などを計算・可視化する。
"""

import re
import sys

from core import (Asset, Strategy, generate_monthly_asset_prices,
                  simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  assets = [
      Asset(name="オルカン", yearly_cost=0, mu=0.07, sigma=0.15, leverage=1),
      Asset(name="定率7%商品", yearly_cost=0, mu=0.07, sigma=0.0, leverage=1)
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      Strategy(name="1. オルカン100% / 取り崩しなし",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=0,
               inflation_rate=0,
               selling_priority=["オルカン"]),
      Strategy(name="2. 現金100% / 400万円取り崩し",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={},
               annual_cost=400,
               inflation_rate=0,
               selling_priority=[]),
      Strategy(name="3. 定率7%商品100% / 400万円取り崩し",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"定率7%商品": 1.0},
               annual_cost=400,
               inflation_rate=0,
               selling_priority=["定率7%商品"]),
      Strategy(name="4. オルカン100% / 400万円取り崩し",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=400,
               inflation_rate=0,
               selling_priority=["オルカン"]),
  ]

  # ---------------------------------------------------------------------------
  # 3. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 4. 可視化と保存
  # ---------------------------------------------------------------------------
  image_file = 'imgs/simple_4p_survival.svg'
  visualize_and_save(results=results,
                     html_file='temp/simple_4p_result.html',
                     survival_image_file=image_file,
                     title='シンプルな4%ルールの最終評価額のパーセンタイル分布',
                     summary_title='シンプルな4%ルール比較サマリー（1,000回試行）',
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

    pattern = r'(<!--<simple_4p_main\.py>-->).*?(<!--</simple_4p_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <simple_4p_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
