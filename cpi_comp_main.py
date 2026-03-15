"""
インフレ率（物価上昇率）のシミュレーション。
オルカンを用いた4%ルールの取り崩しにおいて、様々な物価上昇率の影響を比較する。
"""

import re
import sys
from typing import Union

from core import (Asset, Cpi, Strategy, generate_cpi_paths,
                  generate_monthly_asset_prices, simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def create_strategy(name: str, inflation_rate: Union[float, str]) -> Strategy:
  """
  共通設定を持つ Strategy インスタンスを作成する。

  Args:
    name: 戦略の名前
    inflation_rate: インフレ率（数値またはCPIパス名）

  Returns:
    Strategy: 初期設定済みの戦略オブジェクト
  """
  return Strategy(name=name,
                  initial_money=10000,
                  initial_loan=0,
                  yearly_loan_interest=2.125 / 100,
                  initial_asset_ratio={"オルカン": 1.0},
                  annual_cost=400,
                  inflation_rate=inflation_rate,
                  tax_rate=0,
                  selling_priority=["オルカン"])


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  assets = [
      Asset(name="オルカン", yearly_cost=0, mu=0.07, sigma=0.15, leverage=1),
  ]

  # ---------------------------------------------------------------------------
  # 2. CPIの定義
  # ---------------------------------------------------------------------------
  cpis = [
      Cpi(name="日本のCPI", mu=0.0244, sigma=0.0413),
  ]

  # ---------------------------------------------------------------------------
  # 3. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      create_strategy(name="1. 物価上昇なし", inflation_rate=0.0),
      create_strategy(name="2. インフレ1%", inflation_rate=0.01),
      create_strategy(name="3. インフレ1.5%", inflation_rate=0.015),
      create_strategy(name="4. インフレ2%", inflation_rate=0.02),
      create_strategy(name="5. インフレ2.44%", inflation_rate=0.0244),
      create_strategy(name="6. インフレ2.44% (標準偏差 4.13%)",
                      inflation_rate="日本のCPI"),
  ]

  # ---------------------------------------------------------------------------
  # 4. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  print("CPIの推移を生成中...")
  cpi_paths = generate_cpi_paths(cpis)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices, cpi_paths=cpi_paths)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 5. 可視化と保存
  # ---------------------------------------------------------------------------
  survival_image_file = 'imgs/cpi_comp_survival.svg'
  distribution_image_file = 'imgs/cpi_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/cpi_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='物価上昇率のシミュレーション比較の生存確率',
                     summary_title='物価上昇率ルールの比較サマリー（1,000回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50])

  # ---------------------------------------------------------------------------
  # 6. Markdown レポートの更新
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

    pattern = r'(<!--<cpi_comp_main\.py>-->).*?(<!--</cpi_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <cpi_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
