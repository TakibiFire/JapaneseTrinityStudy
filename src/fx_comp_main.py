"""
為替リスクの影響のシミュレーション。

オルカンを用いた4%ルールの取り崩しにおいて、為替変動による影響を比較する。
以下の5つのシナリオを比較する：
1. 為替リスクなし (= ドル円固定)
2. ドル円のリスク・リターン=0%, 10.53%
3. ドル円のリスク・リターン=0.03%, 10.53%
4. ドル円のリスク・リターン=0%, 9.18%
5. 為替リスクなし, オルカンのリスクを15%→18.3%に変更 (合成リスクの検証)
"""

import os
from dataclasses import replace

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType, FxType, Lifeplan,
                                      PensionStatus, PredefinedStock, Setup,
                                      StrategySpec, WorldConfig,
                                      create_experiment_setup)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  start_age = 50
  initial_money = 10000
  annual_cost_base = 400

  # 1. ビルダーの準備
  world = WorldConfig(n_sim=n_sim,
                      n_years=years,
                      start_age=start_age,
                      cpi_type=CpiType.FIXED_1_77,
                      fx_type=FxType.NONE)
  baseline_lifeplan = Lifeplan(
      base_spend=ConstantSpend(annual_amount=annual_cost_base),
      retirement_start_age=start_age,
      pension_status=PensionStatus.NONE)
  baseline_strategy = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN_FX, 1.0),),
      selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN_FX,))

  exp_setup = Setup(name="baseline",
                    world=world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  # 為替と資産のパラメータ設定 (実験1-4)
  fx_scenarios = [
      ("1. 為替リスクなし (= ドル円固定)", FxType.NONE),
      ("2. ドル円 0%, 10.53%", FxType.USDJPY),
      ("3. ドル円 0.03%, 10.53%", FxType.USDJPY_MU_003_SIGMA_1053),
      ("4. ドル円 0%, 9.18%", FxType.USDJPY_SIGMA_918),
  ]

  for label, fx in fx_scenarios:
    new_world = replace(world, fx_type=fx)
    exp_setup.add_experiment(name=label, overwrite_world=new_world)

  # 実験5: 合成リスク
  strategy_5 = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_18_3_ORUKAN_WITH_FEE,
                            1.0),),
      selling_priority=(PredefinedStock.SIMPLE_7_18_3_ORUKAN_WITH_FEE,))
  exp_setup.add_experiment(name="5. 為替リスクなし, オルカンリスク18.3%",
                           overwrite_strategy=strategy_5)

  # コンパイル
  compiled_experiments = create_experiment_setup(exp_setup)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for exp in compiled_experiments:
    if exp.name == "baseline":
      continue
    results[exp.name] = simulate_strategy(exp.strategy, exp.monthly_prices,
                                          exp.monthly_cashflows)

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


if __name__ == "__main__":
  main()
