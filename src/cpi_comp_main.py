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
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.core import SimulationResult, Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, CpiAsset, Distribution,
                                     MonthlyARLogNormal,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.simulation_defaults import get_cpi_ar12_config
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)


@dataclass(frozen=True)
class CpiParam:
  """
  CPIシミュレーションのパラメータ。

  Attributes:
    label: グラフや表での表示名
    dist: CPI生成に使用する分布オブジェクト
  """
  label: str
  dist: Distribution


def run_experiment(name_prefix: str, asset_configs: List[Asset],
                   cpi_params: List[CpiParam], n_sim: int, n_years: int,
                   initial_money: float, annual_cost: float,
                   seed: int) -> Dict[str, SimulationResult]:
  """
  指定されたCPIパラメータ群に対してシミュレーションを実行する。
  """
  n_months = n_years * 12
  results: Dict[str, SimulationResult] = {}

  for p in cpi_params:
    label = p.label
    cpi_name = f"CPI_{label}"

    # CPI資産の定義
    cpi_asset = CpiAsset(name=cpi_name, dist=p.dist)

    # 価格推移の生成
    all_configs: List[Union[Asset, CpiAsset]] = asset_configs + [cpi_asset]
    monthly_prices = generate_monthly_asset_prices(configs=all_configs,
                                                   n_paths=n_sim,
                                                   n_months=n_months,
                                                   seed=seed)

    # 戦略の定義
    strategy = Strategy(name=label,
                        initial_money=initial_money,
                        initial_loan=0,
                        yearly_loan_interest=0,
                        initial_asset_ratio={asset_configs[0].name: 1.0},
                        annual_cost=annual_cost,
                        inflation_rate=cpi_name,
                        tax_rate=0,
                        selling_priority=[asset_configs[0].name])

    # シミュレーション実行
    res = simulate_strategy(strategy, monthly_prices)
    results[label] = res

  return results


def main():
  # 共通設定
  N_SIM = 5000
  YEARS = 50
  SEED = 42
  INITIAL_MONEY = 10000.0  # 1億円
  ANNUAL_COST = 400.0  # 400万円

  IMG_DIR = "docs/imgs/cpi/"
  DATA_DIR = "docs/data/cpi/"
  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)

  # 資産定義 (7%, 15%)
  orukan = Asset(name="オルカン",
                 dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
                 trust_fee=0,
                 leverage=1)
  assets = [orukan]

  # --- 実験1: インフレ率（平均）の影響 ---
  print("実験1を実行中: インフレ率の影響 (ボラティリティ=0)...")
  exp1_params = [
      CpiParam(label='インフレ率 0.0%',
               dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.0)),
      CpiParam(label='インフレ率 1.0%',
               dist=YearlyLogNormalArithmetic(mu=0.01, sigma=0.0)),
      CpiParam(label='インフレ率 1.5%',
               dist=YearlyLogNormalArithmetic(mu=0.015, sigma=0.0)),
      CpiParam(label='インフレ率 2.0%',
               dist=YearlyLogNormalArithmetic(mu=0.02, sigma=0.0)),
      CpiParam(label='インフレ率 2.44% (歴史的平均)',
               dist=YearlyLogNormalArithmetic(mu=0.0244, sigma=0.0)),
  ]

  results1 = run_experiment("Exp1", assets, exp1_params, N_SIM, YEARS,
                            INITIAL_MONEY, ANNUAL_COST, SEED)

  # 保存と可視化
  formatted_df1, _ = create_styled_summary(
      results1, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(DATA_DIR, "experiment1.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df1.to_markdown())

  _, chart1 = create_survival_probability_chart(results1, max_years=YEARS)
  chart1.save(os.path.join(IMG_DIR, "experiment1_result.svg"))

  # --- 実験2: インフレ・ボラティリティの影響 ---
  print("実験2を実行中: インフレ・ボラティリティの影響 (平均=2.0%)...")
  exp2_params = [
      CpiParam(label='ボラティリティ 0.0%',
               dist=YearlyLogNormalArithmetic(mu=0.02, sigma=0.0)),
      CpiParam(label='ボラティリティ 2.0%',
               dist=YearlyLogNormalArithmetic(mu=0.02, sigma=0.02)),
      CpiParam(label='ボラティリティ 4.13% (歴史的標準偏差)',
               dist=YearlyLogNormalArithmetic(mu=0.02, sigma=0.0413)),
  ]

  results2 = run_experiment("Exp2", assets, exp2_params, N_SIM, YEARS,
                            INITIAL_MONEY, ANNUAL_COST, SEED)

  # 保存と可視化
  formatted_df2, _ = create_styled_summary(
      results2, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(DATA_DIR, "experiment2.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df2.to_markdown())

  _, chart2 = create_survival_probability_chart(results2, max_years=YEARS)
  chart2.save(os.path.join(IMG_DIR, "experiment2_result.svg"))

  # --- 実験3: インフレの粘着性 (AR12) の影響 ---
  print("実験3を実行中: インフレの粘着性 (AR12) の影響...")
  # 直近12ヶ月の対数リターン (古い順). 2025/03〜2026/02
  initial_y = [
      0.0027039223324009146, 0.0035938942545892623, 0.0026869698208253877,
      -0.0008948546458437107, 0.0017889092427246362, 0.0017857147602345312,
      -0.0008924587830196112, 0.007117467768863955, 0.003539826705123987,
      -0.0017683470567420034, -0.0008853475567242145, -0.00621947806702042
  ]

  # AR(12) 1981年〜
  phis_1981 = [0.07456263570805544, -0.14442648233062177, -0.0693542287128989, -0.006265287407105956, 0.06328448135944292, 0.0493508550156997, 0.09362194504911231, 0.03832889494972861, 0.03269694292183145, -0.06762784737529454, 0.07140939573134378, 0.41951806303046024]

  exp3_params = [
      CpiParam(label='独立 (歴史的 2.44%, 4.13%)',
               dist=YearlyLogNormalArithmetic(mu=0.0244, sigma=0.0413)),
      CpiParam(label='AR(12) 粘着性モデル (1970年〜)',
               dist=get_cpi_ar12_config().dist),
      CpiParam(label='AR(12) 粘着性モデル (1981年〜)',
               dist=MonthlyARLogNormal(c=0.00028810,
                                       phis=phis_1981,
                                       sigma_e=0.00317681,
                                       initial_y=initial_y)),
      CpiParam(label='比較: 独立 (1.77%, 0%)',
               dist=YearlyLogNormalArithmetic(mu=0.0177, sigma=0.0)),
  ]

  results3 = run_experiment("Exp3", assets, exp3_params, N_SIM, YEARS,
                            INITIAL_MONEY, ANNUAL_COST, SEED)

  formatted_df3, _ = create_styled_summary(
      results3, bankruptcy_years=[10, 20, 30, 40, 50])
  with open(os.path.join(DATA_DIR, "experiment3.md"), "w",
            encoding="utf-8") as f:
    f.write(formatted_df3.to_markdown())

  _, chart3 = create_survival_probability_chart(results3, max_years=YEARS)
  chart3.save(os.path.join(IMG_DIR, "experiment3_result.svg"))

  print("\nシミュレーション完了。")
  print(f"実験1サマリー: {os.path.join(DATA_DIR, 'experiment1.md')}")
  print(f"実験2サマリー: {os.path.join(DATA_DIR, 'experiment2.md')}")
  print(f"実験3サマリー: {os.path.join(DATA_DIR, 'experiment3.md')}")


if __name__ == "__main__":
  main()
