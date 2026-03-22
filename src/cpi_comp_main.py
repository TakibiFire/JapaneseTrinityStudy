"""
CPI（消費者物価指数）の変動が資産寿命に与える影響を分析するスクリプト。

このスクリプトは、以下の2つの実験を行います。
1. インフレ率（平均値）の影響: ボラティリティを0に固定し、インフレ率の違いによる資産寿命の変化を分析。
2. インフレ率のボラティリティの影響: インフレ率の平均を固定し、ボラティリティ（物価変動の激しさ）による影響を分析。

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
from src.lib.asset_generator import (Asset, CpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)


@dataclass(frozen=True)
class CpiParam:
  """
  CPIシミュレーションのパラメータ。

  Attributes:
    label: グラフや表での表示名
    mu: 年率平均インフレ率
    sigma: 年率インフレ率の標準偏差
  """
  label: str
  mu: float
  sigma: float


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
    mu = p.mu
    sigma = p.sigma
    label = p.label
    cpi_name = f"CPI_mu{mu:.4f}_sigma{sigma:.4f}"

    # CPI資産の定義
    cpi_asset = CpiAsset(name=cpi_name,
                         dist=YearlyLogNormalArithmetic(mu=mu, sigma=sigma))

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
      CpiParam(label='インフレ率 0.0%', mu=0.0000, sigma=0.0),
      CpiParam(label='インフレ率 1.0%', mu=0.0100, sigma=0.0),
      CpiParam(label='インフレ率 1.5%', mu=0.0150, sigma=0.0),
      CpiParam(label='インフレ率 2.0%', mu=0.0200, sigma=0.0),
      CpiParam(label='インフレ率 2.44% (歴史的平均)', mu=0.0244, sigma=0.0),
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
      CpiParam(label='ボラティリティ 0.0%', mu=0.0200, sigma=0.0000),
      CpiParam(label='ボラティリティ 2.0%', mu=0.0200, sigma=0.0200),
      CpiParam(label='ボラティリティ 4.13% (歴史的標準偏差)', mu=0.0200, sigma=0.0413),
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

  print("\nシミュレーション完了。")
  print(f"実験1サマリー: {os.path.join(DATA_DIR, 'experiment1.md')}")
  print(f"実験1グラフ: {os.path.join(IMG_DIR, 'experiment1_result.svg')}")
  print(f"実験2サマリー: {os.path.join(DATA_DIR, 'experiment2.md')}")
  print(f"実験2グラフ: {os.path.join(IMG_DIR, 'experiment2_result.svg')}")

  # openコマンドの表示
  print("\n以下のコマンドで結果を確認できます:")
  print(f"open {os.path.join(IMG_DIR, 'experiment1_result.svg')}")
  print(f"open {os.path.join(IMG_DIR, 'experiment2_result.svg')}")


if __name__ == "__main__":
  main()
