"""
為替リスクの影響のシミュレーション。
オルカンを用いた4%ルールの取り崩しにおいて、為替変動による影響を比較する。
"""

import re
import sys

from core import (Asset, Forex, Strategy, generate_forex_paths,
                  generate_monthly_asset_prices, simulate_strategy)
from visualize import create_styled_summary, visualize_and_save


def create_strategy(name: str, asset_name: str) -> Strategy:
  """
  共通設定を持つ Strategy インスタンスを作成する。

  Args:
    name: 戦略の名前
    asset_name: 使用する資産の名前

  Returns:
    Strategy: 初期設定済みの戦略オブジェクト
  """
  return Strategy(name=name,
                  initial_money=10000,
                  initial_loan=0,
                  yearly_loan_interest=2.125 / 100,
                  initial_asset_ratio={asset_name: 1.0},
                  annual_cost=400,
                  inflation_rate=0.02,
                  tax_rate=0.20315,
                  selling_priority=[asset_name])


def main():
  # ---------------------------------------------------------------------------
  # 1. 為替レートと資産の定義
  # ---------------------------------------------------------------------------
  fx_params = [
      ("為替リスクなし (= ドル円固定)", 0.0, 0.0),
      ("ドル円 0%, 10.53%", 0.0, 0.1053),
      ("ドル円 0.03%, 10.53%", 0.0003, 0.1053),
      ("ドル円 0%, 9.18%", 0.0, 0.0918),
  ]

  forexes = []
  assets = []
  strategies = []

  for i, (name, mu, sigma) in enumerate(fx_params):
    fx_name = name
    forexes.append(Forex(name=fx_name, mu=mu, sigma=sigma))

    asset_name = f"オルカン({fx_name})"
    assets.append(
        Asset(name=asset_name,
              trust_fee=0.0005775,
              mu=0.07,
              sigma=0.15,
              leverage=1,
              forex=fx_name))

    strategy_name = f"{i + 1}. {name}"
    strategies.append(create_strategy(name=strategy_name,
                                      asset_name=asset_name))

  # 5th.
  fifth_asset_name = f"為替リスクなし, オルカンリスク18.3%"
  assets.append(
      Asset(name=fifth_asset_name,
            trust_fee=0.0005775,
            mu=0.07,
            sigma=0.183,
            leverage=1,
            forex=fx_params[0][0]))
  fifth_strategy_name = f"5. {fifth_asset_name}"
  strategies.append(
      create_strategy(name=fifth_strategy_name, asset_name=fifth_asset_name))

  print("為替の月次推移を生成中...")
  forex_paths = generate_forex_paths(forexes)

  # ---------------------------------------------------------------------------
  # 2. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("オルカンの月次価格推移を生成中...")
  # 同じ関数内で複数の Asset を渡すと、共通の乱数 Z が使用されるため、
  # 為替変動以外の値動き（オルカン自体の価格推移）は全く同じになる
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       forex_paths=forex_paths)

  results = {}
  print("各戦略のシミュレーションを実行中...")

  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 3. 可視化と保存
  # ---------------------------------------------------------------------------
  survival_image_file = 'imgs/fx_comp_survival.svg'
  distribution_image_file = 'imgs/fx_comp_distribution.svg'
  visualize_and_save(results=results,
                     html_file='temp/fx_comp_result.html',
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='為替リスクのシミュレーション比較',
                     summary_title='為替リスクの比較サマリー（1,000回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50])

  # ---------------------------------------------------------------------------
  # 4. Markdown レポートの更新
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

    pattern = r'(<!--<fx_comp_main\.py>-->).*?(<!--</fx_comp_main\.py>-->)'
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
          f"\033[91mWarning: Placeholder <fx_comp_main.py> not found in {report_file}\033[0m",
          file=sys.stderr)
  except FileNotFoundError:
    print(f"\033[91mWarning: {report_file} not found\033[0m", file=sys.stderr)


if __name__ == "__main__":
  main()
