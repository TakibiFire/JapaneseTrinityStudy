"""
年齢ごとの支出変動を考慮した場合の生存確率への影響を比較するシミュレーション。
"""

import os
from dataclasses import replace

from src.core import simulate_strategy
from src.lib.retired_spending import SpendingType
from src.lib.scenario_builder import (ConstantSpend, CpiType, CurveSpend,
                                      FxType, Lifeplan, PensionStatus,
                                      PredefinedStock, Setup, StrategySpec,
                                      WorldConfig, create_experiment_setup)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  max_years = 50
  seed = 42
  initial_money = 10000.0  # 万円
  base_annual_cost = 400.0

  # 1. 宣言型APIによるシナリオ構築
  baseline_world = WorldConfig(
      n_sim=n_sim,
      n_years=max_years,
      start_age=30,  # ベースラインの開始年齢
      seed=seed,
      cpi_type=CpiType.FIXED_1_77,
      fx_type=FxType.USDJPY)

  baseline_lifeplan = Lifeplan(
      retirement_start_age=30,
      base_spend=ConstantSpend(annual_amount=base_annual_cost),
      pension_status=PensionStatus.NONE)  # retired_spending_comp_main.py では年金なし

  baseline_strategy = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),),
      selling_priority=(PredefinedStock.ORUKAN_155,),
      rebalance=None,
      spend_adjustment=None)

  exp_setup = Setup(name="1. 支出一定 (400万円)",
                    world=baseline_world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  start_ages = [30, 35, 40, 45, 50, 55, 60]
  for i, age in enumerate(start_ages):
    exp_setup.add_experiment(
        name=f"{i+2}. {age}歳",
        overwrite_world=replace(baseline_world, start_age=age),
        overwrite_lifeplan=replace(
            baseline_lifeplan,
            retirement_start_age=age,
            base_spend=CurveSpend(
                first_year_annual_amount=base_annual_cost,
                spending_types=(SpendingType.CONSUMPTION,
                                SpendingType.NON_CONSUMPTION))))

  # 2. コンパイルと実行
  compiled_experiments = create_experiment_setup(exp_setup)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for exp in compiled_experiments:
    results[exp.name] = simulate_strategy(
        exp.strategy,
        exp.monthly_prices,
        monthly_cashflows=exp.monthly_cashflows)

  # 3. 可視化と保存
  img_dir = "docs/imgs/retired_spending"
  data_dir = "docs/data/retired_spending"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  survival_image_file = os.path.join(img_dir, 'survival.svg')
  distribution_image_file = os.path.join(img_dir, 'distribution.svg')
  html_file = 'temp/cost_per_age_comp_result.html'

  print("結果を保存中...")
  visualize_and_save(results=results,
                     html_file=html_file,
                     survival_image_file=survival_image_file,
                     distribution_image_file=distribution_image_file,
                     title='年齢別支出の影響',
                     summary_title='年齢別支出の影響サマリー',
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


if __name__ == "__main__":
  main()
