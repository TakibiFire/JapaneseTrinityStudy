"""
無リスク資産（4%利回り）とのリバランスが生存確率へ与える影響を比較するシミュレーション。

実験1: リバランスの有無と売却順序による影響 (オルカン 100%, 60%)
実験2: リバランスの頻度による影響 (オルカン 60% 固定)

設定:
- 初期資産: 1億円 (10,000万円)
- 投資先: オルカン (年率 7%, リスク 15%) + 為替リスク (0%, 10.53%)
- 無リスク資産: 利回り 4%
- 取り崩し額: 毎年400万円 (物価連動)
- インフレ率: 年率 1.77% (固定)
"""

import os
from dataclasses import replace
from typing import Dict, List, Tuple, Union

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType,
                                      DynamicV1Rebalance, FixedRebalance,
                                      Lifeplan, PensionStatus, PredefinedAsset,
                                      PredefinedStock, PredefinedZeroRisk,
                                      Setup, StrategySpec, WorldConfig,
                                      create_experiment_setup)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  # シミュレーション設定
  n_sim = 5000
  years = 50
  start_age = 50
  initial_money = 10000
  annual_cost_base = 400

  # 1. シナリオビルダーの準備
  world = WorldConfig(n_sim=n_sim,
                      n_years=years,
                      start_age=start_age,
                      cpi_type=CpiType.FIXED_1_77)
  baseline_lifeplan = Lifeplan(
      base_spend=ConstantSpend(annual_amount=annual_cost_base),
      retirement_start_age=start_age,
      pension_status=PensionStatus.NONE)
  # baseline_strategy は実験ごとに大きく変わるので、ここではプレースホルダ
  baseline_strategy = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),),
      selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN,))

  def run_experiment(exp_name: str,
                     exp_title: str,
                     test_cases: List[tuple],
                     is_dynamic: bool = False):
    """
    指定されたテストケースでシミュレーションを実行し、結果を保存する。
    """
    exp_setup = Setup(name="baseline",
                      world=world,
                      lifeplan=baseline_lifeplan,
                      strategy=baseline_strategy)

    for stock_ratio, interval, selling_priority_enums, label in test_cases:
      zr_ratio = 1.0 - stock_ratio
      ratios: List[Tuple[PredefinedAsset, float]] = []
      if stock_ratio > 0:
        ratios.append((PredefinedStock.SIMPLE_7_15_ORUKAN, stock_ratio))
      if zr_ratio > 0:
        ratios.append((PredefinedZeroRisk.ZERO_RISK_4PCT, zr_ratio))

      rebalance: Union[FixedRebalance, DynamicV1Rebalance, None] = None
      if interval > 0:
        if is_dynamic:
          rebalance = DynamicV1Rebalance(
              risky_asset=PredefinedStock.SIMPLE_7_15_ORUKAN,
              zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
              interval_months=interval)
        else:
          rebalance = FixedRebalance(interval_months=interval)

      new_strategy = StrategySpec(
          initial_money=initial_money,
          initial_asset_ratio=tuple(ratios),
          selling_priority=tuple(selling_priority_enums),
          rebalance=rebalance)
      exp_setup.add_experiment(name=label, overwrite_strategy=new_strategy)

    # コンパイル
    compiled_experiments = create_experiment_setup(exp_setup)

    results = {}
    print(f"[{exp_title}] 各戦略のシミュレーションを実行中...")
    for exp in compiled_experiments:
      if exp.name == "baseline":
        continue
      results[exp.name] = simulate_strategy(exp.strategy, exp.monthly_prices,
                                            exp.monthly_cashflows)

    # 可視化と保存
    img_dir = "docs/imgs/zero_risk_rebalance"
    data_dir = "docs/data/zero_risk_rebalance"
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

  # 実験1: リバランスの有無と資産比率
  # (オルカン比率, リバランス間隔, 売却順序Enums, ラベル)
  priority_zr_s = [
      PredefinedZeroRisk.ZERO_RISK_4PCT, PredefinedStock.SIMPLE_7_15_ORUKAN
  ]

  # 実験1の「リバ毎年」は FixedRebalance
  exp1_cases = [
      (1.0, 0, [PredefinedStock.SIMPLE_7_15_ORUKAN], "オルカン 100%"),
      (0.8, 0, priority_zr_s, "80% リバなし"),
      (0.8, 12, priority_zr_s, "80% リバ毎年"),
      (0.6, 0, priority_zr_s, "60% リバなし"),
      (0.6, 12, priority_zr_s, "60% リバ毎年"),
  ]
  run_experiment("rebalance_effect",
                 "実験1: リバランスの有無と資産比率",
                 exp1_cases,
                 is_dynamic=False)

  # 実験2: リバランスの頻度による影響
  # 売却順序は「無リスク資産を先に使う」で固定
  exp2_cases = [
      (0.6, 1, priority_zr_s, "リバ毎月"),
      (0.6, 3, priority_zr_s, "リバ3ヶ月"),
      (0.6, 6, priority_zr_s, "リバ半年"),
      (0.6, 12, priority_zr_s, "リバ1年"),
      (0.6, 24, priority_zr_s, "リバ2年"),
      (0.6, 60, priority_zr_s, "リバ5年"),
      (0.6, 0, priority_zr_s, "リバなし"),
  ]
  run_experiment("rebalance_freq",
                 "実験2: リバランスの頻度",
                 exp2_cases,
                 is_dynamic=False)


if __name__ == "__main__":
  main()
