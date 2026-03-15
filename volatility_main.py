"""
ボラティリティ比較のシミュレーションを実行し、HTMLレポートを生成するスクリプト。

オルカンの期待リターンは固定とし、ボラティリティ（シグマ）のみを変化させた場合の
複数のセットアップを比較します。
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
  sigmas = [0, 11, 13, 15, 17]

  assets = [
      Asset(name=f"オルカン v{v}%",
            yearly_cost=0,
            mu=0.07,
            sigma=v / 100.0,
            leverage=1) for v in sigmas
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  strategies = [
      Strategy(name=f"オルカン, ボラ={v}%",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={f"オルカン v{v}%": 1.0},
               annual_cost=0,
               inflation_rate=0,
               selling_priority=[f"オルカン v{v}%"]) for v in sigmas
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
  visualize_and_save(results=results,
                     html_file='temp/volatility_result.html',
                     image_file='imgs/volatility_result.svg',
                     title='ボラティリティ違いによる50年後の最終評価額',
                     summary_title='ボラティリティ比較サマリー（1,000回試行）',
                     bankruptcy_years=[])

  # ---------------------------------------------------------------------------
  # 5. Markdown レポートの更新
  # ---------------------------------------------------------------------------
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[])
  md_text = formatted_df.to_markdown(colalign=("left",) +
                                     ("right",) * len(formatted_df.columns))

  report_file = 'docs/report.md'
  try:
    with open(report_file, 'r', encoding='utf-8') as f:
      content = f.read()

    pattern = r'(<!--<volatility_main\.py>-->).*?(<!--</volatility_main\.py>-->)'
    if re.search(pattern, content, re.DOTALL):
      new_content = re.sub(pattern,
                           rf'\g<1>\n\n{md_text.strip()}\n\n\g<2>',
                           content,
                           flags=re.DOTALL)
      with open(report_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
      print(f"✅ {report_file} を更新しました。")
    else:
      print(
          f"\033[91mWarning: Placeholder <volatility_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
