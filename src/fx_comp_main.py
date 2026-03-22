"""
為替リスクの影響のシミュレーション。

オルカンを用いた4%ルールの取り崩しにおいて、為替変動による影響を比較する。
以下の5つのシナリオを比較する：
1. 為替リスクなし (= ドル円固定)
2. ドル円のリスク・リターン=0%, 10.53%
3. ドル円のリスク・リターン=0.03%, 10.53%
4. ドル円のリスク・リターン=0%, 9.18%
5. 為替リスクなし, オルカンのリスクを15%→18.3%に変更 (合成リスクの検証)

出力ファイル:
- docs/data/forex/result.md: シミュレーション結果のサマリーテーブル
- docs/imgs/forex/fx_comp_survival.svg: 生存確率の推移グラフ
- docs/imgs/forex/fx_comp_distribution.svg: 50年後の資産分布グラフ
"""

import os
from typing import List, Union

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset, ForexAsset,
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
  inflation_rate_std = 0.02
  trust_fee_std = 0.0005775

  # 共通のCPI資産
  cpi_name = "Japan_CPI_2pct"
  cpi_asset = CpiAsset(name=cpi_name,
                       dist=YearlyLogNormalArithmetic(mu=inflation_rate_std,
                                                      sigma=0.0))

  # 1. 資産の定義
  # 為替と資産のパラメータ設定
  fx_params = [
      ("為替リスクなし", 0.0, 0.0),
      ("ドル円_0_10.53", 0.0, 0.1053),
      ("ドル円_0.03_10.53", 0.0003, 0.1053),
      ("ドル円_0_9.18", 0.0, 0.0918),
  ]

  configs: List[Union[Asset, DerivedAsset, ForexAsset, CpiAsset]] = [cpi_asset]

  # ベースとなる「オルカン(生)」を定義（為替なし、手数料なし）
  # これを元に DerivedAsset で為替や手数料を適用する
  base_stock_name = "BaseStock"
  configs.append(
      Asset(name=base_stock_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=0,
            leverage=1))

  # 各為替設定に対する資産と戦略
  strategies = []

  for i, (fx_label, mu, sigma) in enumerate(fx_params):
    fx_name = f"Forex_{i}"
    configs.append(
        ForexAsset(name=fx_name,
                   dist=YearlyLogNormalArithmetic(mu=mu, sigma=sigma)))

    asset_name = f"オルカン_{fx_label}"
    configs.append(
        DerivedAsset(name=asset_name,
                     base=base_stock_name,
                     trust_fee=trust_fee_std,
                     forex=fx_name))

    strategy_name = f"{i + 1}. {fx_label}"
    if fx_label == "為替リスクなし":
      strategy_name = f"{i + 1}. {fx_label} (= ドル円固定)"
    elif fx_label == "ドル円_0_10.53":
      strategy_name = f"{i + 1}. ドル円 0%, 10.53%"
    elif fx_label == "ドル円_0.03_10.53":
      strategy_name = f"{i + 1}. ドル円 0.03%, 10.53%"
    elif fx_label == "ドル円_0_9.18":
      strategy_name = f"{i + 1}. ドル円 0%, 9.18%"

    strategies.append(
        Strategy(name=strategy_name,
                 initial_money=initial_money,
                 initial_loan=0,
                 yearly_loan_interest=2.125 / 100,
                 initial_asset_ratio={asset_name: 1.0},
                 annual_cost=annual_cost_base,
                 inflation_rate=cpi_name,
                 tax_rate=tax_rate_std,
                 selling_priority=[asset_name]))

  # 5. 合成リスクの検証用 (為替リスクなし, オルカンリスク18.3%)
  synth_asset_name = "オルカン_合成リスク18.3%"
  configs.append(
      Asset(name=synth_asset_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.183),
            trust_fee=trust_fee_std,
            leverage=1))

  strategies.append(
      Strategy(name="5. 為替リスクなし, オルカンリスク18.3%",
               initial_money=initial_money,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={synth_asset_name: 1.0},
               annual_cost=annual_cost_base,
               inflation_rate=cpi_name,
               tax_rate=tax_rate_std,
               selling_priority=[synth_asset_name]))

  # 2. シミュレーションの実行
  print(f"月次価格の推移を生成中 (パス数: {n_sim})...")
  monthly_asset_prices = generate_monthly_asset_prices(configs,
                                                       n_paths=n_sim,
                                                       n_months=n_months,
                                                       seed=seed)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # 3. 可視化と保存
  img_dir = "docs/imgs/forex"
  data_dir = "docs/data/forex"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'fx_comp_survival.svg')
  distribution_image_file = os.path.join(img_dir, 'fx_comp_distribution.svg')
  html_file = 'temp/fx_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='為替リスクのシミュレーション比較',
                     summary_title=f'為替リスクの比較サマリー（{n_sim:,}回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     open_browser=False)

  # 4. Markdownデータの出力
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
