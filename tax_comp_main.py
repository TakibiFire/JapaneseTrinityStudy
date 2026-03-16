"""
譲渡所得税の影響のシミュレーション。
オルカンを用いた4%ルールの取り崩しにおいて、譲渡所得税の有無による影響を比較する。
"""

import re
import sys

from core import (Asset, Strategy, generate_monthly_asset_prices,
                  simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def create_strategy(name: str, annual_cost: float, tax_rate: float) -> Strategy:
  """
  共通設定を持つ Strategy インスタンスを作成する。

  Args:
    name: 戦略の名前
    annual_cost: 年出費
    tax_rate: 譲渡所得税率

  Returns:
    Strategy: 初期設定済みの戦略オブジェクト
  """
  return Strategy(name=name,
                  initial_money=10000,
                  initial_loan=0,
                  yearly_loan_interest=2.125 / 100,
                  initial_asset_ratio={"オルカン": 1.0},
                  annual_cost=annual_cost,
                  inflation_rate=0.02,
                  tax_rate=tax_rate,
                  selling_priority=["オルカン"])


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  assets = [
      Asset(name="オルカン", yearly_cost=0, mu=0.07, sigma=0.15, leverage=1),
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      create_strategy(name="1. 譲渡所得税を考慮しない", annual_cost=400, tax_rate=0.0),
      create_strategy(name="2. 譲渡所得税が 20.315%",
                      annual_cost=400,
                      tax_rate=0.20315),
      create_strategy(name="3. 譲渡所得税は0%、出費を20.315%増やす",
                      annual_cost=400 * (1.0 + 0.20315),
                      tax_rate=0),
      create_strategy(name="4. 譲渡所得税は0%、出費を11.5%増やす",
                      annual_cost=400 * (1.0 + 0.115),
                      tax_rate=0),
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
  survival_image_file = 'imgs/tax_comp_survival.svg'
  distribution_image_file = 'imgs/tax_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/tax_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='譲渡所得税の有無のシミュレーション比較',
                     summary_title='譲渡所得税ルールの比較サマリー（1,000回試行）',
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

    pattern = r'(<!--<tax_comp_main\.py>-->).*?(<!--</tax_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <tax_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
