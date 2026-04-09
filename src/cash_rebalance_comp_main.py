"""
リバランスの有無および頻度による生存確率への影響を比較するシミュレーション。

実験1: リバランスの有無と売却順序による影響 (オルカン 100%, 80%)
実験2: リバランスの頻度による影響 (オルカン 80% 固定)

設定:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 現金資産: 利回り 0%
- 取り崩し額: 毎年400万円 (物価連動)
- インフレ率: 年率 1.77% (固定)
"""

import os
from typing import Dict, List, Union

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
  initial_money = 10000
  annual_cost_base = 400
  tax_rate = 0.20315
  inflation_rate_std = 0.0177
  fee_acwi = 0.0005775

  # 共通アセット名
  cpi_name = "Japan_CPI_1.77pct"
  fx_name = "USDJPY_0_10.53"
  acwi_name = "オルカン"
  cash_asset_name = "現金資産"

  # 資産モデル設定
  ork_dist = YearlyLogNormalArithmetic(mu=0.07, sigma=0.15)
  fx_dist = YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053)

  # 1. 価格推移の生成
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

  cash_asset = ZeroRiskAsset(name=cash_asset_name, yield_rate=0.0)

  def run_experiment(exp_name: str, exp_title: str, test_cases: List[tuple]):
    """
    指定されたテストケースでシミュレーションを実行し、結果を保存する。
    
    Args:
        exp_name: 実験の識別名
        exp_title: 実験のタイトル
        test_cases: (オルカン比率, リバランス間隔, 売却順序, ラベル) のリスト
    """
    strategies = []
    for stock_ratio, interval, selling_priority, label in test_cases:
      cash_ratio = 1.0 - stock_ratio
      
      initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float] = {
          acwi_name: stock_ratio
      }
      if cash_ratio > 0:
        initial_asset_ratio[cash_asset] = cash_ratio

      strategies.append(
          Strategy(name=label,
                   initial_money=initial_money,
                   initial_loan=0,
                   yearly_loan_interest=0.0,
                   initial_asset_ratio=initial_asset_ratio,
                   annual_cost=annual_cost_base,
                   inflation_rate=cpi_name,
                   tax_rate=tax_rate,
                   selling_priority=selling_priority,
                   rebalance_interval=interval))

    # シミュレーションの実行
    results = {}
    print(f"[{exp_title}] 各戦略のシミュレーションを実行中...")
    for strategy in strategies:
      res = simulate_strategy(strategy, monthly_asset_prices)
      results[strategy.name] = res

    # 可視化と保存
    img_dir = "docs/imgs/cash_rebalance"
    data_dir = "docs/data/cash_rebalance"
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
                       title=f'生存確率の比較 ({exp_title})',
                       summary_title=f'{exp_title} サマリー（{n_sim:,}回試行）',
                       bankruptcy_years=[10, 20, 30, 40, 50],
                       open_browser=False)

    # Markdownデータの出力
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

  # 実験1: リバランスの有無と売却順序
  # (オルカン比率, リバランス間隔, 売却順序, ラベル)
  priority_c_s = [cash_asset_name, acwi_name]
  priority_s_c = [acwi_name, cash_asset_name]
  
  exp1_cases = [
      (1.0, 0, [acwi_name], "オルカン 100%"),
      (0.6, 0, priority_c_s, "60% リバなし (現→オ)"),
      (0.6, 12, priority_c_s, "60% リバ毎年 (現→オ)"),
      (0.6, 0, priority_s_c, "60% リバなし (オ→現)"),
      (0.6, 12, priority_s_c, "60% リバ毎年 (オ→現)"),
  ]
  run_experiment("rebalance_effect", "実験1: リバランスの有無と売却順序", exp1_cases)

  # 実験2: リバランスの頻度による影響
  # 売却順序は「現金を先に使う」で固定
  exp2_cases = [
      (0.6, 1, priority_c_s, "リバ毎月"),
      (0.6, 3, priority_c_s, "リバ3ヶ月"),
      (0.6, 6, priority_c_s, "リバ半年"),
      (0.6, 12, priority_c_s, "リバ1年"),
      (0.6, 24, priority_c_s, "リバ2年"),
      (0.6, 60, priority_c_s, "リバ5年"),
      (0.6, 0, priority_c_s, "リバなし"),
  ]
  run_experiment("rebalance_freq", "実験2: リバランスの頻度", exp2_cases)


if __name__ == "__main__":
  main()
