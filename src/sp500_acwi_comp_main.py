"""
S&P500 と ACWI の取り崩しシミュレーション比較。

- 1. S&P500 (155 years, Monthly, Model C: genlogistic)
- 2. S&P500 (30 years, Monthly, Model C: genlogistic)
- 3. ACWI (18 years, Monthly, Model C: johnsonsu)
- 4. ACWI Approx (S&P500 155yr * 1.0269 - 0.002907 + dweibull noise)

為替リスク（ドル円 0%, 10.53%）を全モデルに合成。
信託報酬: S&P500=0.0814%, ACWI=0.05775%
"""

import os
from typing import Dict, List, Union

from scipy import stats

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset, ForexAsset,
                                     MonthlyDist, MonthlyLogDist,
                                     MonthlyLogNormal,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.simulation_defaults import AcwiModelKey, get_acwi_fat_tail_config
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
  
  # 信託報酬
  fee_sp500 = 0.000814   # 0.0814%
  fee_acwi = 0.0005775   # 0.05775%

  # 1. データのパラメータ設定 (data/model_fitting_results_v3.txt より)
  # S&P500 155y Model C (genlogistic)
  # Ann(mu=10.33%, sig=15.03%)
  sp500_155y_params = (0.5983257553837089, 0.024055922548623175, 0.017141333060447166)
  # S&P500 30y Model C (genlogistic)
  # Ann(mu=11.64%, sig=17.14%)
  sp500_30y_params = (0.4879653982267047, 0.033214317138593324, 0.017280587830235443)
  # ACWI 18y Model C (johnsonsu)
  # Ann(mu=9.51%, sig=18.30%)
  acwi_18y_params = (0.5985794609028992, 1.5979947040822444, 0.033828733503047, 0.05883263137905962)
  
  # ACWI Approx
  # ACWI = 1.0269 * SP500 + -0.002907 + noise
  acwi_approx_mult = 1.0269
  acwi_approx_intercept = -0.002907
  # Noise (dweibull): c, loc (intercept as mean shift), scale
  # Ann(mu=7.05%, sig=15.77%)
  acwi_approx_noise_params = (1.2199932203810953, acwi_approx_intercept, 0.010652296731100462)

  # 2. 資産の定義
  cpi_name = "Japan_CPI_1.77pct"
  
  # 為替リスクの定義 (ドル円 0%, 10.53%)
  fx_name = "USDJPY_0_10.53"
  fx_asset = ForexAsset(name=fx_name,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  
  assets: List[Union[Asset, DerivedAsset, ForexAsset, CpiAsset]] = []
  assets.append(fx_asset)

  # ベース資産の登録
  # 1. S&P500 155yr (共通モデルから取得)
  assets.append(get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y))
  
  # 2. S&P500 30yr
  assets.append(
      Asset(name="Base_SP500_30y",
            dist=MonthlyLogDist(stats.genlogistic, params=sp500_30y_params),
            trust_fee=0.0,
            leverage=1)
  )
  # 3. ACWI 18yr
  assets.append(
      Asset(name="Base_ACWI_18y",
            dist=MonthlyLogDist(stats.johnsonsu, params=acwi_18y_params),
            trust_fee=0.0,
            leverage=1)
  )
  # 4. ACWI Approx (共通モデルから取得)
  assets.append(get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX))

  # 為替と各信託報酬を適用した投資用DerivedAssetを定義
  model_names = [
      ("1. S&P500 (155年)", "Base_SP500_155y", fee_sp500),
      ("2. S&P500 (30年)", "Base_SP500_30y", fee_sp500),
      ("3. オルカン (18年)", "Base_ACWI_18y", fee_acwi),
      ("4. オルカン (S&P近似)", "Base_ACWI_Approx", fee_acwi),
  ]

  strategy_names = []
  for final_name, base_name, fee in model_names:
    assets.append(
        DerivedAsset(name=final_name,
                     base=base_name,
                     trust_fee=fee,
                     forex=fx_name)
    )
    strategy_names.append(final_name)

  cpi_asset = CpiAsset(name=cpi_name,
                       dist=YearlyLogNormalArithmetic(mu=inflation_rate_std,
                                                      sigma=0.0))
  assets.append(cpi_asset)

  # 3. 戦略(Plan)の定義
  strategies = []
  for name in strategy_names:
    strategies.append(
        Strategy(name=name,
                 initial_money=initial_money,
                 initial_loan=0,
                 yearly_loan_interest=0.0,
                 initial_asset_ratio={name: 1.0},
                 annual_cost=annual_cost_base,
                 inflation_rate=cpi_name,
                 tax_rate=tax_rate_std,
                 selling_priority=[name]))

  # 4. シミュレーションの実行
  print(f"月次価格の推移を生成中 (パス数: {n_sim})...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=n_sim,
                                                       n_months=n_months,
                                                       seed=seed)

  results = {}
  print("各モデルのシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # 5. 可視化と保存
  img_dir = "docs/imgs/sp500_acwi"
  data_dir = "docs/data/sp500_acwi"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'sp500_vs_acwi_comp_survival.svg')
  distribution_image_file = os.path.join(img_dir, 'sp500_vs_acwi_comp_distribution.svg')
  html_file = 'temp/sp500_vs_acwi_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='S&P500 vs ACWI シミュレーション比較',
                     summary_title=f'比較サマリー（{n_sim:,}回試行）',
                     bankruptcy_years=[10, 20, 30, 40, 50],
                     open_browser=False)

  # 6. Markdownデータの出力
  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[10, 20, 30, 40, 50])

  md_table = formatted_df.to_markdown(colalign=("left",) +
                                      ("right",) * len(formatted_df.columns))

  # Let's save to a specific markdown file
  md_file = os.path.join(data_dir, 'sp500_vs_acwi_result.md')
  with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_table)

  print(f"✅ {md_file} を作成しました。")
  print(f"✅ {survival_image_file} を作成しました。")
  print(f"✅ {distribution_image_file} を作成しました。")


if __name__ == "__main__":
  main()
