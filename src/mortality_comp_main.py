"""
死亡率を考慮した資産寿命（生存中の破産確率）を多角的に分析するスクリプト。

実験設定:
- 固定パラメータ:
  - 試行回数: 5,000回
  - 期間: 50年間
  - 初期資産: 1億円
  - 初期年間支出: 400万円 (支出率4%)
  - 投資先: オルカン100%（期待リターン7%、リスク15%、信託報酬 0.05775%）
  - 為替リスク: USDJPY（期待リターン0%、リスク10.53%）
  - インフレ率: 1.77% (固定)
  - 税率: 20.315%

- 変動パラメータ:
  - 開始年齢: 40, 50, 60, 70
  - 性別: 男性, 女性
  - 死亡率考慮: あり/なし

成功判定の定義:
1. 破産せずにシミュレーション期間（50年間）を終える
2. シミュレーション期間中に死亡する (死亡＝成功)
"""

import os
from dataclasses import replace
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, CpiType, FxType, Gender,
                                      Lifeplan, PensionStatus, PredefinedStock,
                                      Setup, StrategySpec, WorldConfig,
                                      create_experiment_setup)
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)

# 出力先ディレクトリ
IMG_DIR = "docs/imgs/mortality/"
DATA_DIR = "docs/data/mortality/"


def main():
  # 共通設定
  n_sim = 5000
  years = 50
  seed = 42
  initial_money = 10000.0
  annual_cost = 400.0

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. 宣言型APIによるシナリオ構築
  baseline_world = WorldConfig(n_sim=n_sim,
                               n_years=years,
                               start_age=40,
                               seed=seed,
                               cpi_type=CpiType.FIXED_1_77,
                               fx_type=FxType.USDJPY)

  baseline_lifeplan = Lifeplan(
      retirement_start_age=40,
      base_spend=ConstantSpend(annual_amount=annual_cost),
      pension_status=PensionStatus.NONE,
      mortality_gender=None)

  baseline_strategy = StrategySpec(
      initial_money=initial_money,
      initial_asset_ratio=((PredefinedStock.ORUKAN_155, 1.0),),
      selling_priority=(PredefinedStock.ORUKAN_155,),
      rebalance=None,
      spend_adjustment=None)

  exp_setup = Setup(name="baseline",
                    world=baseline_world,
                    lifeplan=baseline_lifeplan,
                    strategy=baseline_strategy)

  start_ages = [40, 50, 60, 70]
  for age in start_ages:
    # 男性
    # 死亡率考慮なし
    exp_setup.add_experiment(name=f"Male_Age{age}_OFF",
                             overwrite_world=replace(baseline_world,
                                                     start_age=age),
                             overwrite_lifeplan=replace(
                                 baseline_lifeplan,
                                 retirement_start_age=age,
                                 mortality_gender=None))
    # 死亡率考慮あり
    exp_setup.add_experiment(name=f"Male_Age{age}_ON",
                             overwrite_world=replace(baseline_world,
                                                     start_age=age),
                             overwrite_lifeplan=replace(
                                 baseline_lifeplan,
                                 retirement_start_age=age,
                                 mortality_gender=Gender.MALE))

    # 女性
    # 死亡率考慮なし
    exp_setup.add_experiment(name=f"Female_Age{age}_OFF",
                             overwrite_world=replace(baseline_world,
                                                     start_age=age),
                             overwrite_lifeplan=replace(
                                 baseline_lifeplan,
                                 retirement_start_age=age,
                                 mortality_gender=None))
    # 死亡率考慮あり
    exp_setup.add_experiment(name=f"Female_Age{age}_ON",
                             overwrite_world=replace(baseline_world,
                                                     start_age=age),
                             overwrite_lifeplan=replace(
                                 baseline_lifeplan,
                                 retirement_start_age=age,
                                 mortality_gender=Gender.FEMALE))

  # 2. コンパイルと実行
  compiled_experiments = create_experiment_setup(exp_setup)

  results = {}
  print("シミュレーションを実行中...")
  # ベースラインは結果に含めない
  for exp in compiled_experiments[1:]:
    results[exp.name] = simulate_strategy(
        exp.strategy,
        exp.monthly_prices,
        monthly_cashflows=exp.monthly_cashflows)

  # 3. 可視化とデータ保存

  # 全結果のサマリー
  formatted_df, _ = create_styled_summary(results,
                                          bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(DATA_DIR, "result.md"), "w", encoding="utf-8") as f:
    f.write(formatted_df.to_markdown())

  # 比較チャート作成
  male_results = {k: v for k, v in results.items() if k.startswith("Male")}
  female_results = {k: v for k, v in results.items() if k.startswith("Female")}

  create_comparison_chart(male_results, "male", years)
  create_comparison_chart(female_results, "female", years)

  # ヒートマップ作成
  create_combined_heatmaps(male_results, female_results, start_ages)

  print(f"完了。結果を {DATA_DIR} と {IMG_DIR} に保存しました。")


def create_comparison_chart(results: Dict, gender_label: str, years: int):
  target_strategies = {
      f"{gender_label.capitalize()}_Age40_OFF": "死亡率を考慮しない",
      f"{gender_label.capitalize()}_Age40_ON": "40歳からの  死亡率を考慮",
      f"{gender_label.capitalize()}_Age50_ON": "50歳からの  死亡率を考慮",
      f"{gender_label.capitalize()}_Age60_ON": "60歳からの  死亡率を考慮",
      f"{gender_label.capitalize()}_Age70_ON": "70歳からの  死亡率を考慮",
  }

  plot_data = []
  for original_name, japanese_name in target_strategies.items():
    if original_name not in results:
      continue
    res = results[original_name]
    sustained = res.sustained_months
    for y in range(years + 1):
      survival_rate = np.mean(sustained >= y * 12) * 100.0
      plot_data.append({
          'Year': y,
          'Strategy': japanese_name,
          'Survival Probability (%)': survival_rate
      })

  if not plot_data:
    return

  df_plot = pd.DataFrame(plot_data)

  strategy_order = list(target_strategies.values())

  # y軸の下限を調整
  min_val = df_plot['Survival Probability (%)'].min()
  y_min = (min_val // 10) * 10
  y_max = 100

  ja_map = {
      "male": "男性",
      "female": "女性",
  }
  display_survival_title = f'経過年数と成功確率の推移 ({ja_map[gender_label]})'
  if y_min > 0:
    display_survival_title += f"（成功確率 {y_min:.0f}%以下は描画を省略）"

  chart = alt.Chart(df_plot).mark_line(point=True).encode(
      x=alt.X('Year:Q', title='経過年数 (年)'),
      y=alt.Y('Survival Probability (%):Q',
              title='成功確率 (%)',
              scale=alt.Scale(domain=[y_min, y_max])),
      color=alt.Color('Strategy:N',
                      title='戦略',
                      sort=strategy_order,
                      legend=alt.Legend(orient='top',
                                        labelExpr="split(datum.label, '  ')")),
      tooltip=[
          'Year', 'Strategy',
          alt.Tooltip('Survival Probability (%):Q', format='.1f')
      ]).properties(title=display_survival_title, width=600, height=300)

  chart.save(os.path.join(IMG_DIR, f"survival_comparison_{gender_label}.svg"))


def create_combined_heatmaps(male_results: Dict, female_results: Dict,
                             start_ages: List[int]):
  """
  80歳と90歳時点での生存確率を、性別を統合したヒートマップで可視化する。
  """
  heatmap_data = []

  for age in start_ages:
    # 考慮しない (Male_AgeX_OFF と Female_AgeX_OFF は同じ結果なので片方から取得)
    res_off = male_results[f"Male_Age{age}_OFF"]
    sustained_off = res_off.sustained_months

    # 男性死亡率を考慮
    res_male = male_results[f"Male_Age{age}_ON"]
    sustained_male = res_male.sustained_months

    # 女性死亡率を考慮
    res_female = female_results[f"Female_Age{age}_ON"]
    sustained_female = res_female.sustained_months

    for target_age in [80, 90]:
      years_to_target = target_age - age
      if years_to_target >= 0:
        # 共通のターゲット年齢に対する計算
        heatmap_data.append({
            'StartAge':
                age,
            'Setting':
                '考慮しない',
            'TargetAge':
                target_age,
            'SurvivalProbability':
                np.mean(sustained_off >= years_to_target * 12) * 100.0
        })
        heatmap_data.append({
            'StartAge':
                age,
            'Setting':
                '男性死亡率を考慮',
            'TargetAge':
                target_age,
            'SurvivalProbability':
                np.mean(sustained_male >= years_to_target * 12) * 100.0
        })
        heatmap_data.append({
            'StartAge':
                age,
            'Setting':
                '女性死亡率を考慮',
            'TargetAge':
                target_age,
            'SurvivalProbability':
                np.mean(sustained_female >= years_to_target * 12) * 100.0
        })

  df_heatmap = pd.DataFrame(heatmap_data)

  y_order = ['男性死亡率を考慮', '女性死亡率を考慮', '考慮しない']

  for target_age in [80, 90]:
    df_sub = df_heatmap[df_heatmap['TargetAge'] == target_age]

    chart = alt.Chart(df_sub).mark_rect().encode(
        x=alt.X('StartAge:O', title='開始年齢'),
        y=alt.Y('Setting:N', title='設定', sort=y_order),
        color=alt.Color('SurvivalProbability:Q',
                        title='成功確率 (%)',
                        scale=alt.Scale(scheme='redyellowgreen',
                                        domain=[0, 100])),
        tooltip=[
            'StartAge', 'Setting',
            alt.Tooltip('SurvivalProbability:Q', format='.1f')
        ]).properties(title=f"{target_age}歳時点での成功確率 (%)", width=300, height=200)

    text = chart.mark_text(baseline='middle').encode(
        text=alt.Text('SurvivalProbability:Q', format='.1f'),
        color=alt.condition(alt.datum.SurvivalProbability > 30,
                            alt.value('black'), alt.value('white')))

    final_chart = (chart + text)
    final_chart.save(
        os.path.join(IMG_DIR, f"heatmap_combined_age_{target_age}.svg"))


if __name__ == "__main__":
  main()
