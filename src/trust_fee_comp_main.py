"""
信託報酬の影響のシミュレーション。

オルカンを用いた4%ルールの取り崩しにおいて、信託報酬の違いによる影響を比較する。
以下の5つの信託報酬設定を比較する：
1. 0%
2. 0.05775% (現在の eMAXIS Slim 全世界株式（オール・カントリー）)
3. 0.1133% (以前の eMAXIS Slim 全世界株式（オール・カントリー）)
4. 1%
5. 2%

出力ファイル:
- docs/data/trust_fee/result.md: シミュレーション結果のサマリーテーブル
- docs/imgs/trust_fee/trust_fee_comp_survival.svg: 生存確率の推移グラフ
- docs/imgs/trust_fee/trust_fee_comp_distribution.svg: 50年後の資産分布グラフ
"""

import os
from typing import List, Union

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  n_months = years * 12
  seed = 42
  initial_money = 10000  # 1億円 (単位: 万円)
  annual_cost_base = 400  # 400万円
  tax_rate_std = 0.20315
  inflation_rate_std = 0.0177

  # 1. 資産の定義
  # 全ての戦略で同じ市場環境（価格推移）を共有するために DerivedAsset を使用する
  base_asset_name = "オルカン_ベース"
  cpi_name = "Japan_CPI_1.77pct"

  # ベースとなる資産（信託報酬 0%）
  assets = [
      Asset(name=base_asset_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=0,
            leverage=1),
  ]

  # 比較対象の信託報酬設定
  trust_fees = [
      ("1. 信託報酬=0%", 0.0),
      ("2. 信託報酬=0.05775%", 0.0005775),
      ("3. 信託報酬=0.1133%", 0.001133),
      ("4. 信託報酬=1%", 0.01),
      ("5. 信託報酬=2%", 0.02),
  ]

  # 各信託報酬に対応する資産を定義
  comparison_assets = []
  for name, fee in trust_fees:
    comparison_assets.append(
        DerivedAsset(name=name, base=base_asset_name, trust_fee=fee, leverage=1))

  cpi_asset = CpiAsset(name=cpi_name,
                       dist=YearlyLogNormalArithmetic(mu=inflation_rate_std,
                                                       sigma=0.0))

  # 2. 戦略(Plan)の定義
  # 1. キャッシュフロールールの定義
  spend_config = BaseSpendConfig(name="生活費",
                                 amount=annual_cost_base,
                                 cpi_name=cpi_name)
  cashflow_rules = [
      CashflowRule(source_name=spend_config.name,
                   cashflow_type=CashflowType.REGULAR)
  ]

  strategies = []
  for name, _ in trust_fees:
    strategies.append(
        Strategy(name=name,
                 initial_money=initial_money,
                 initial_loan=0,
                 yearly_loan_interest=2.125 / 100,
                 initial_asset_ratio={name: 1.0},
                 cashflow_rules=cashflow_rules,
                 tax_rate=tax_rate_std,
                 selling_priority=[name]))

  # 3. シミュレーションの実行
  print(f"月次価格の推移を生成中 (パス数: {n_sim})...")
  all_configs: List[Union[Asset, DerivedAsset,
                          CpiAsset]] = assets + comparison_assets + [cpi_asset]  # type: ignore
  monthly_asset_prices = generate_monthly_asset_prices(all_configs,
                                                       n_paths=n_sim,
                                                       n_months=n_months,
                                                       seed=seed)
  monthly_cashflows = generate_cashflows([spend_config], monthly_asset_prices,
                                         n_sim, n_months)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy,
                            monthly_asset_prices,
                            monthly_cashflows=monthly_cashflows)
    results[strategy.name] = res

  # 4. 可視化と保存
  img_dir = "docs/imgs/trust_fee"
  data_dir = "docs/data/trust_fee"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'trust_fee_comp_survival.svg')
  distribution_image_file = os.path.join(img_dir, 'trust_fee_comp_distribution.svg')
  html_file = 'temp/trust_fee_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='信託報酬の違いのシミュレーション比較',
                     summary_title=f'信託報酬ルールの比較サマリー（{n_sim:,}回試行）',
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
