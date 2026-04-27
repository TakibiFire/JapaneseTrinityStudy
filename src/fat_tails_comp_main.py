"""
異なる確率分布（対数正規分布、Johnson SU分布）を用いたリタイアメントシミュレーションの比較。

- ACWI（全世界株式）を対象とする
- モデル1: 対数正規分布 (Log-Normal)
- モデル2: Johnson SU分布 (Fixed Mean)
- パラメータ引用元: data/model_fitting_results_v3.txt (ACWI [Monthly])
"""

import os
from typing import Dict, List, Union

from scipy import stats

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset, ForexAsset,
                                     MonthlyLogDist, MonthlyLogNormal,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main() -> None:
  # シミュレーション設定
  n_sim = 5000
  years = 50
  n_months = years * 12
  seed = 42
  initial_money = 10000  # 1億円 (単位: 万円)
  annual_cost_base = 400  # 400万円
  tax_rate_std = 0.20315
  inflation_rate_std = 0.0177
  trust_fee_std = 0.0005775  # 0.05775%

  # 1. データのパラメータ設定 (data/model_fitting_results_v3.txt より ACWI [Monthly])
  # Model B (Log Normal): mu=0.006393, std=0.048285
  mu_log_monthly = 0.006393
  sigma_log_monthly = 0.048285

  # Model C (Best Asymmetric Dist Fixed Mean): dist=johnsonsu,
  # params=(0.5985794609028992, 1.5979947040822444, 0.033828733503047, 0.05883263137905962)
  jsu_params_monthly = (0.5985794609028992, 1.5979947040822444,
                        0.033828733503047, 0.05883263137905962)

  # 2. 資産の定義
  cpi_name = "Japan_CPI_1.77pct"

  # 為替リスクの定義 (ドル円 0%, 10.53%)
  fx_name = "USDJPY_0_10.53"
  fx_asset = ForexAsset(name=fx_name,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))

  # 比較対象のベースモデル (為替・コスト適用前)
  base_configs = [
      ("Base_ACWI_LogNormal",
       MonthlyLogDist(stats.norm, params=(mu_log_monthly, sigma_log_monthly))),
      ("Base_ACWI_JSU",
       MonthlyLogDist(stats.johnsonsu, params=jsu_params_monthly)),
  ]

  assets: List[Union[Asset, DerivedAsset, ForexAsset, CpiAsset]] = []
  assets.append(fx_asset)

  # ベース資産を登録
  for base_name, dist in base_configs:
    assets.append(Asset(name=base_name, dist=dist, trust_fee=0.0, leverage=1))

  # 為替と信託報酬を適用した資産を定義
  model_names = []
  for base_name, _ in base_configs:
    if "LogNormal" in base_name:
      final_name = "ACWI-1: 対数正規分布 (為替あり)"
    else:
      final_name = "ACWI-2: Johnson SU分布 (為替あり)"

    assets.append(
        DerivedAsset(name=final_name,
                     base=base_name,
                     trust_fee=trust_fee_std,
                     forex=fx_name))
    model_names.append(final_name)

  cpi_asset = CpiAsset(name=cpi_name,
                       dist=YearlyLogNormalArithmetic(mu=inflation_rate_std,
                                                      sigma=0.0))

  # 3. キャッシュフロールールの定義
  spend_config = BaseSpendConfig(name="生活費",
                                 amount=annual_cost_base,
                                 cpi_name=cpi_name)
  cashflow_rules = [
      CashflowRule(source_name=spend_config.name,
                   cashflow_type=CashflowType.REGULAR)
  ]

  # 4. 戦略(Plan)の定義
  strategies = []
  for name in model_names:
    strategies.append(
        Strategy(name=name,
                 initial_money=initial_money,
                 initial_loan=0,
                 yearly_loan_interest=0.0,
                 initial_asset_ratio={name: 1.0},
                 cashflow_rules=cashflow_rules,
                 tax_rate=tax_rate_std,
                 selling_priority=[name]))

  # 5. シミュレーションの実行
  print(f"月次価格の推移を生成中 (パス数: {n_sim})...")
  all_configs: List[Union[Asset,
                          CpiAsset]] = assets + [cpi_asset]  # type: ignore
  monthly_asset_prices = generate_monthly_asset_prices(all_configs,
                                                       n_paths=n_sim,
                                                       n_months=n_months,
                                                       seed=seed)
  monthly_cashflows = generate_cashflows([spend_config], monthly_asset_prices,
                                         n_sim, n_months)

  results = {}
  print("各モデルのシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy,
                            monthly_asset_prices,
                            monthly_cashflows=monthly_cashflows)
    results[strategy.name] = res

  # 5. 可視化と保存
  img_dir = "docs/imgs/fat_tails"
  data_dir = "docs/data/fat_tails"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'fat_tails_comp_survival.svg')
  distribution_image_file = os.path.join(img_dir,
                                         'fat_tails_comp_distribution.svg')
  html_file = 'temp/fat_tails_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='確率分布モデルの違いによるシミュレーション比較',
                     summary_title=f'分布モデルの比較サマリー（{n_sim:,}回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     open_browser=False)

  # 6. Markdownデータの出力
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[10, 20, 30, 40, 50])

  md_table = formatted_df.to_markdown(colalign=("left",) +
                                      ("right",) * len(formatted_df.columns))

  md_file = os.path.join(data_dir, 'simulation_result.md')
  with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_table)

  print(f"✅ {md_file} を作成しました。")
  print(f"✅ {survival_image_file} を作成しました。")
  print(f"✅ {distribution_image_file} を作成しました。")


if __name__ == "__main__":
  main()
