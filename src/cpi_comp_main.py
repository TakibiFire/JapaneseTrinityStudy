"""
CPI（消費者物価指数）の変動が資産寿命に与える影響を分析するスクリプト。

このスクリプトは、以下の3つの実験を行います。
1. インフレ率（平均値）の影響: ボラティリティを0に固定し、インフレ率の違いによる資産寿命の変化を分析。
2. インフレ率のボラティリティの影響: インフレ率の平均を固定し、ボラティリティ（物価変動の激しさ）による影響を分析。
3. インフレの粘着性（自己相関）の影響: 月次データから推計したAR(12)モデルを用い、粘着性が資産寿命に与える影響を分析。

資産設定:
- オルカン: 期待リターン 7%, リスク 15% (YearlyLogNormalArithmetic)
- 初期資産: 1億円 (10,000万円)
- 初期取り崩し額: 400万円/年 (物価連動)
- シミュレーション期間: 50年
- 試行回数: 5,000回
"""

import os

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType, FxType, Lifeplan,
                                      PensionStatus, PredefinedStock, Setup,
                                      StrategySpec, WorldConfig,
                                      create_experiment_setup)
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)


def main():
  # 共通設定
  n_sim = 5000
  years = 50
  seed = 42
  initial_money = 10000.0  # 1億円
  annual_cost = 400.0  # 400万円

  img_dir = "docs/imgs/cpi/"
  data_dir = "docs/data/cpi/"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  # ベースラインの定義 (実験1, 2, 3 で共有)
  baseline_world = WorldConfig(
      n_sim=n_sim,
      n_years=years,
      start_age=50,
      seed=seed,
      tax_rate=0,  # オリジナルスクリプトに合わせる
      fx_type=FxType.NONE  # オリジナルスクリプトに合わせる
  )

  baseline_lifeplan = Lifeplan(
      retirement_start_age=50,
      base_spend=ConstantSpend(annual_amount=annual_cost),
      pension_status=PensionStatus.NONE)

  baseline_strategy = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),),
      selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN,),
      rebalance=None,
      spend_adjustment=None)

  # --- 実験1: インフレ率（平均）の影響 ---
  print("実験1を実行中: インフレ率の影響 (ボラティリティ=0)...")
  exp1_setup = Setup(name="dummy",
                     world=baseline_world,
                     lifeplan=baseline_lifeplan,
                     strategy=baseline_strategy)

  exp1_configs = [
      ('インフレ率 0.0%', CpiType.FIXED_0),
      ('インフレ率 1.0%', CpiType.FIXED_1_0),
      ('インフレ率 1.5%', CpiType.FIXED_1_5),
      ('インフレ率 2.0%', CpiType.FIXED_2_0),
      ('インフレ率 2.44% (歴史的平均)', CpiType.FIXED_2_44),
  ]

  for label, cpi in exp1_configs:
    exp1_setup.add_experiment(name=label,
                              overwrite_world=replace(baseline_world,
                                                      cpi_type=cpi))

  results1 = {
      exp.name:
      simulate_strategy(exp.strategy,
                        exp.monthly_prices,
                        monthly_cashflows=exp.monthly_cashflows)
      for exp in create_experiment_setup(exp1_setup)[1:]
  }

  # 保存と可視化
  formatted_df1, _ = create_styled_summary(
      results1, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(data_dir, "experiment1.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df1.to_markdown())

  _, chart1 = create_survival_probability_chart(results1, max_years=years)
  chart1.save(os.path.join(img_dir, "experiment1_result.svg"))

  # --- 実験2: インフレ・ボラティリティの影響 ---
  print("実験2を実行中: インフレ・ボラティリティの影響 (平均=2.0%)...")
  exp2_setup = Setup(name="dummy",
                     world=baseline_world,
                     lifeplan=baseline_lifeplan,
                     strategy=baseline_strategy)

  exp2_configs = [
      ('ボラティリティ 0.0%', CpiType.FIXED_2_0),
      ('ボラティリティ 2.0%', CpiType.FIXED_2_0_VOL_2_0),
      ('ボラティリティ 4.13% (歴史的標準偏差)', CpiType.FIXED_2_0_VOL_4_13),
  ]

  for label, cpi in exp2_configs:
    exp2_setup.add_experiment(name=label,
                              overwrite_world=replace(baseline_world,
                                                      cpi_type=cpi))

  results2 = {
      exp.name:
      simulate_strategy(exp.strategy,
                        exp.monthly_prices,
                        monthly_cashflows=exp.monthly_cashflows)
      for exp in create_experiment_setup(exp2_setup)[1:]
  }

  # 保存と可視化
  formatted_df2, _ = create_styled_summary(
      results2, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(data_dir, "experiment2.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df2.to_markdown())

  _, chart2 = create_survival_probability_chart(results2, max_years=years)
  chart2.save(os.path.join(img_dir, "experiment2_result.svg"))

  # --- 実験3: インフレの粘着性 (AR12) の影響 ---
  print("実験3を実行中: インフレの粘着性 (AR12) の影響...")
  exp3_setup = Setup(name="dummy",
                     world=baseline_world,
                     lifeplan=baseline_lifeplan,
                     strategy=baseline_strategy)

  exp3_configs = [
      ('独立 (歴史的 2.44%, 4.13%)', CpiType.FIXED_2_44_VOL_4_13),
      ('AR(12) 粘着性モデル (1970年〜)', CpiType.JAPAN_AR12),
      ('AR(12) 粘着性モデル (1981年〜)', CpiType.JAPAN_AR12_1981),
      ('比較: 独立 (1.77%, 0%)', CpiType.FIXED_1_77),
  ]

  for label, cpi in exp3_configs:
    exp3_setup.add_experiment(name=label,
                              overwrite_world=replace(baseline_world,
                                                      cpi_type=cpi))

  results3 = {
      exp.name:
      simulate_strategy(exp.strategy,
                        exp.monthly_prices,
                        monthly_cashflows=exp.monthly_cashflows)
      for exp in create_experiment_setup(exp3_setup)[1:]
  }

  formatted_df3, _ = create_styled_summary(
      results3, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(data_dir, "experiment3.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df3.to_markdown())

  _, chart3 = create_survival_probability_chart(results3, max_years=years)
  chart3.save(os.path.join(img_dir, "experiment3_result.svg"))

  print("\nシミュレーション完了。")
  print(f"実験1サマリー: {os.path.join(data_dir, 'experiment1.md')}")
  print(f"実験2サマリー: {os.path.join(data_dir, 'experiment2.md')}")
  print(f"実験3サマリー: {os.path.join(data_dir, 'experiment3.md')}")


if __name__ == "__main__":
  from dataclasses import replace
  main()
