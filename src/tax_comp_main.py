"""
譲渡所得税の影響のシミュレーション。

オルカンを用いた4%ルールの取り崩しにおいて、譲渡所得税の有無による影響を比較する。
以下の4つの戦略を比較する：
1. 譲渡所得税を考慮しない (0%)
2. 譲渡所得税を考慮する (20.315%)
3. 譲渡所得税は0%だが、出費を20.315%増やす
4. 譲渡所得税は0%だが、出費を11.5%増やす

出力ファイル:
- docs/data/tax/result.md: シミュレーション結果のサマリーテーブル
- docs/imgs/tax/tax_comp_survival.svg: 生存確率の推移グラフ
- docs/imgs/tax/tax_comp_distribution.svg: 50年後の資産分布グラフ
"""

import os
from typing import List, Union

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  n_months = years * 12
  seed = 42
  initial_money = 10000
  annual_cost_base = 400
  tax_rate_std = 0.20315
  inflation_rate_std = 0.0177

  # 1. 資産の定義
  # 全ての戦略で同じ市場環境（価格推移）を共有するために DerivedAsset を使用する
  base_asset_name = "オルカン_ベース"
  cpi_name = "Japan_CPI_1.77pct"

  assets = [
      Asset(name=base_asset_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=0,
            leverage=1),
      DerivedAsset(name="オルカン", base=base_asset_name, trust_fee=0, leverage=1),
  ]
  cpi_asset = CpiAsset(name=cpi_name,
                       dist=YearlyLogNormalArithmetic(mu=inflation_rate_std,
                                                      sigma=0.0))

  # 2. 戦略(Plan)の定義
  strategies = [
      Strategy(name="1. 税を考慮しない",
               initial_money=initial_money,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=annual_cost_base,
               inflation_rate=cpi_name,
               tax_rate=0.0,
               selling_priority=["オルカン"]),
      Strategy(name="2. 税 20.315%",
               initial_money=initial_money,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=annual_cost_base,
               inflation_rate=cpi_name,
               tax_rate=tax_rate_std,
               selling_priority=["オルカン"]),
      Strategy(name="3. 税 0%、出費を 20.315% 増やす",
               initial_money=initial_money,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=annual_cost_base * (1.0 + tax_rate_std),
               inflation_rate=cpi_name,
               tax_rate=0.0,
               selling_priority=["オルカン"]),
      Strategy(name="4. 税 0%、出費を 11.5% 増やす",
               initial_money=initial_money,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={"オルカン": 1.0},
               annual_cost=annual_cost_base * (1.0 + 0.115),
               inflation_rate=cpi_name,
               tax_rate=0.0,
               selling_priority=["オルカン"]),
  ]

  # 3. シミュレーションの実行
  print(f"月次価格の推移を生成中 (パス数: {n_sim})...")
  all_configs: List[Union[Asset, DerivedAsset,
                          CpiAsset]] = assets + [cpi_asset]  # type: ignore
  monthly_asset_prices = generate_monthly_asset_prices(all_configs,
                                                       n_paths=n_sim,
                                                       n_months=n_months,
                                                       seed=seed)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # 4. 可視化と保存
  img_dir = "docs/imgs/tax"
  data_dir = "docs/data/tax"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'tax_comp_survival.svg')
  distribution_image_file = os.path.join(img_dir, 'tax_comp_distribution.svg')
  html_file = 'temp/tax_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='譲渡所得税の有無のシミュレーション比較',
                     summary_title=f'譲渡所得税ルールの比較サマリー（{n_sim:,}回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     open_browser=False)

  # 5. Markdownデータの出力
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[10, 20, 30, 40, 50])

  md_table = formatted_df.to_markdown(colalign=("left",) +
                                      ("right",) * len(formatted_df.columns))

  md_file = os.path.join(data_dir, 'result.md')
  with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_table)

  print(f"✅ {md_file} を作成しました。")
  print(f"✅ {survival_image_file} を作成しました。")
  print(f"✅ {distribution_image_file} を作成しました。")
  print(f"詳細な結果は {html_file} で確認できます。")
  print(f"open {html_file}")


if __name__ == "__main__":
  main()
