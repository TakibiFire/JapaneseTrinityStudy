"""
資産配分（オルカン vs 現金）と取り崩し順序による生存確率への影響を比較するシミュレーション。

実験1: 現金を先に使う (Spend Cash First)
実験2: オルカンを先に使う (Spend Stock First)

設定:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 取り崩し額: 毎年400万円 (物価連動)
- インフレ率: 年率 1.77% (固定)
- 譲渡所得税: 20.315%
- 信託報酬: 0.05775%
"""

import os
from typing import Dict, List, Union

import numpy as np

from src.core import Strategy, ZeroRiskAsset, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, ForexAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  seed = 42
  initial_money = 10000  # 1億円 (単位: 万円)
  annual_cost_base = 400  # 400万円 (単位: 万円)
  tax_rate = 0.20315
  inflation_rate_std = 0.0177
  fee_acwi = 0.0005775

  # 共通アセット名
  cpi_name = "Japan_CPI_1.77pct"
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  cash_asset_name = "現金資産"

  # 資産モデル設定
  # オルカン: 算術平均 7%, 算術標準偏差 15% (Yearly Log-Normal Arithmetic)
  ork_dist = YearlyLogNormalArithmetic(mu=0.07, sigma=0.15)
  # 為替: 算術平均 0%, 算術標準偏差 10.53% (Yearly Log-Normal Arithmetic)
  fx_dist = YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)

  # 1. 価格推移の生成 (全実験で共通)
  assets: List[Union[Asset, ForexAsset, CpiAsset]] = [
      ForexAsset(name=fx_name, dist=fx_dist),
      Asset(name=acwi_name, dist=ork_dist, trust_fee=fee_acwi, forex=fx_name),
      CpiAsset(name=cpi_name,
               dist=YearlyLogNormalArithmetic(mu=inflation_rate_std, sigma=0.0))
  ]

  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=years * 12,
                                                       seed=seed)

  def run_experiment(exp_name: str, exp_title: str, ratios: List[float],
                     selling_priority_func):
    """
    指定された比率と売却順序でシミュレーションを実行し、結果を保存する。

    Args:
        exp_name: 実験の識別名（ファイル名に使用）
        exp_title: 実験のタイトル（グラフやサマリーに使用）
        ratios: オルカンの初期保有比率のリスト
        selling_priority_func: 売却順序を決定する関数
    """

    cash_asset = ZeroRiskAsset(name=cash_asset_name, yield_rate=0.0)

    # 1. 戦略(Plan)の定義
    strategies = []
    for ratio in ratios:
      stock_ratio = ratio
      cash_ratio = 1.0 - ratio

      name = f"オル:現={round(stock_ratio*100)}:{round(cash_ratio*100)}"

      # 資産比率の設定
      initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float] = {
          acwi_name: stock_ratio,
          cash_asset: cash_ratio
      }

      strategies.append(
          Strategy(name=name,
                   initial_money=initial_money,
                   initial_loan=0,
                   yearly_loan_interest=0.0,
                   initial_asset_ratio=initial_asset_ratio,
                   annual_cost=annual_cost_base,
                   inflation_rate=cpi_name,
                   tax_rate=tax_rate,
                   selling_priority=selling_priority_func(
                       acwi_name, cash_asset_name)))

    # 2. シミュレーションの実行
    results = {}
    print(f"[{exp_title}] 各戦略のシミュレーションを実行中...")
    for strategy in strategies:
      res = simulate_strategy(strategy, monthly_asset_prices)
      results[strategy.name] = res

    # 3. 可視化と保存
    img_dir = "docs/imgs/cache_ratio"
    data_dir = "docs/data/cache_ratio"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    survival_image_file = os.path.join(img_dir, f'{exp_name}_survival.svg')
    distribution_image_file = os.path.join(img_dir,
                                           f'{exp_name}_distribution.svg')
    html_file = f'temp/{exp_name}_result.html'

    visualize_and_save(results=results,
                       html_file=html_file,
                       survival_image_file=survival_image_file,
                       distribution_image_file=distribution_image_file,
                       title=f'資産配分による生存確率の比較 ({exp_title})',
                       summary_title=f'{exp_title} サマリー（{n_sim:,}回試行）',
                       bankruptcy_years=[10, 20, 30, 40, 50],
                       open_browser=False)

    # 4. Markdownデータの出力
    formatted_df, _ = create_styled_summary(
        results,
        quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
        bankruptcy_years=[10, 20, 30, 40, 50])

    md_table = formatted_df.to_markdown(colalign=("left",) +
                                        ("right",) * len(formatted_df.columns))

    md_file = os.path.join(data_dir, f'{exp_name}_result.md')
    with open(md_file, 'w', encoding='utf-8') as f:
      f.write(md_table)

    print(f"✅ {md_file} を作成しました。")
    print(f"✅ {survival_image_file} を作成しました。")
    print(f"✅ {distribution_image_file} を作成しました。")

  ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

  # 実験1: 現金を先に使う
  run_experiment(exp_name="exp1",
                 exp_title="実験1: 現金を先に使う",
                 ratios=ratios,
                 selling_priority_func=lambda s, c: [c, s])

  # 実験2: オルカンを先に使う
  run_experiment(exp_name="exp2",
                 exp_title="実験2: オルカンを先に使う",
                 ratios=ratios,
                 selling_priority_func=lambda s, c: [s, c])


if __name__ == "__main__":
  main()
