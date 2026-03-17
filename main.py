"""
日本版トリニティ・スタディのシミュレーションを実行し、HTMLレポートを生成するスクリプト。

`core.py` の機能を利用して複数の投資戦略（オルカン100%やレバレッジ活用など）の
シミュレーションを比較し、最終純資産の分布や破産確率などを視覚化して `new_result.html` に出力する。
"""

from core import (Asset, Strategy, ZeroRiskAsset, generate_monthly_asset_prices,
                  simulate_strategy)
from visualize import visualize_and_save


def main():
  # ---------------------------------------------------------------------------
  # 1. 資産の定義
  # ---------------------------------------------------------------------------
  assets = [
      Asset(name="オルカン", trust_fee=0.05775 / 100, leverage=1),
      Asset(name="レバカン", trust_fee=0.422 / 100, leverage=2)
  ]

  # ---------------------------------------------------------------------------
  # 2. 戦略(Plan)の定義
  # ---------------------------------------------------------------------------
  plan_zero = Strategy(name="ZERO",
                       initial_money=10000,
                       initial_loan=0,
                       yearly_loan_interest=2.125 / 100,
                       initial_asset_ratio={},
                       annual_cost=400,
                       inflation_rate=0.015,
                       selling_priority=[])

  plan_a = Strategy(name="A: オルカン100%",
                    initial_money=10000,
                    initial_loan=0,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={"オルカン": 1.0},
                    annual_cost=0,
                    inflation_rate=0,
                    selling_priority=["オルカン"])

  plan_a_cost_4p = Strategy(name="オルカン100%, cost4%, inf1.5%",
                            initial_money=10000,
                            initial_loan=0,
                            yearly_loan_interest=2.125 / 100,
                            initial_asset_ratio={"オルカン": 1.0},
                            annual_cost=400,
                            inflation_rate=0.015,
                            selling_priority=["オルカン"])

  plan_a_80p_cost_4p = Strategy(name="オルカン80%, cost4%, inf1.5%",
                                initial_money=10000,
                                initial_loan=0,
                                yearly_loan_interest=2.125 / 100,
                                initial_asset_ratio={"オルカン": 0.8},
                                annual_cost=400,
                                inflation_rate=0.015,
                                selling_priority=["オルカン"])

  plan_a_50p_cost_4p = Strategy(name="オルカン50%, cost4%, inf1.5%",
                                initial_money=10000,
                                initial_loan=0,
                                yearly_loan_interest=2.125 / 100,
                                initial_asset_ratio={"オルカン": 0.5},
                                annual_cost=400,
                                inflation_rate=0.015,
                                selling_priority=["オルカン"])

  plan_opt = Strategy(name="Opt",
                      initial_money=10000,
                      initial_loan=3000,
                      yearly_loan_interest=2.125 / 100,
                      initial_asset_ratio={
                          "オルカン": 0.9,
                          "レバカン": 0.0
                      },
                      annual_cost=400,
                      inflation_rate=0.015,
                      selling_priority=["オルカン", "レバカン"])

  plan_opt_reb1 = Strategy(name="Opt Rebalance 1",
                           initial_money=10000,
                           initial_loan=3000,
                           yearly_loan_interest=2.125 / 100,
                           initial_asset_ratio={
                               "オルカン": 0.9,
                               "レバカン": 0.0
                           },
                           annual_cost=400,
                           inflation_rate=0.015,
                           selling_priority=["オルカン", "レバカン"],
                           rebalance_interval=1)

  plan_opt_reb12 = Strategy(name="Opt Rebalance 12",
                            initial_money=10000,
                            initial_loan=3000,
                            yearly_loan_interest=2.125 / 100,
                            initial_asset_ratio={
                                "オルカン": 0.9,
                                "レバカン": 0.0
                            },
                            annual_cost=400,
                            inflation_rate=0.015,
                            selling_priority=["オルカン", "レバカン"],
                            rebalance_interval=12)

  plan_opt_reb180 = Strategy(name="Opt Rebalance 180",
                             initial_money=10000,
                             initial_loan=3000,
                             yearly_loan_interest=2.125 / 100,
                             initial_asset_ratio={
                                 "オルカン": 0.9,
                                 "レバカン": 0.0
                             },
                             annual_cost=400,
                             inflation_rate=0.015,
                             selling_priority=["オルカン", "レバカン"],
                             rebalance_interval=180)

  plan_b = Strategy(name="B: オルカン50%+レバカン50%, cost4%, inf1.5%",
                    initial_money=10000,
                    initial_loan=0,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={
                        "オルカン": 0.5,
                        "レバカン": 0.5
                    },
                    annual_cost=400,
                    inflation_rate=0.015,
                    selling_priority=["オルカン", "レバカン"])

  plan_b_4_4 = Strategy(name="B: オルカン40%+レバカン40%, cost4%, inf1.5%",
                        initial_money=10000,
                        initial_loan=0,
                        yearly_loan_interest=2.125 / 100,
                        initial_asset_ratio={
                            "オルカン": 0.4,
                            "レバカン": 0.4
                        },
                        annual_cost=400,
                        inflation_rate=0.015,
                        selling_priority=["オルカン", "レバカン"])

  plan_c = Strategy(name="C: 証券担保ローン1.5倍, cost4%, inf1.5%",
                    initial_money=10000,
                    initial_loan=5000,
                    yearly_loan_interest=2.125 / 100,
                    initial_asset_ratio={"オルカン": 1.0},
                    annual_cost=400,
                    inflation_rate=0.015,
                    selling_priority=["オルカン"])

  plan_zero_risk_test = Strategy(
      name="ZeroRiskAssetテスト (BIL 20%)",
      initial_money=10000,
      initial_loan=0,
      yearly_loan_interest=2.125 / 100,
      initial_asset_ratio={
          "オルカン": 0.8,
          ZeroRiskAsset(name="BIL", yield_rate=0.04): 0.2
      },
      annual_cost=400,
      inflation_rate=0.015,
      selling_priority=["オルカン"])

  opt50 = Strategy(name="Opt50",
                   initial_money=10000,
                   initial_loan=5000,
                   yearly_loan_interest=2.125 / 100,
                   initial_asset_ratio={
                       "オルカン": 0.0,
                       "レバカン": 0.9
                   },
                   annual_cost=400,
                   inflation_rate=0.015,
                   selling_priority=["レバカン", "オルカン"],
                   rebalance_interval=0)

  strategies = [
      #plan_zero,
      #plan_a,
      plan_a_cost_4p,
      plan_a_80p_cost_4p,
      plan_a_50p_cost_4p,
      plan_opt,
      plan_opt_reb1,
      plan_opt_reb12,
      plan_opt_reb180,
      opt50,
      #plan_b,
      #plan_b_4_4,
      #plan_c,
      plan_zero_risk_test,
  ]

  # ---------------------------------------------------------------------------
  # 3. シミュレーションの実行
  # ---------------------------------------------------------------------------
  print("月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets)

  results = {}
  print("各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # ---------------------------------------------------------------------------
  # 4. 可視化と保存
  # ---------------------------------------------------------------------------
  visualize_and_save(results=results,
                     html_file='temp/new_result.html',
                     title='50年後の最終評価額のパーセンタイル分布',
                     summary_title='50年後の最終評価額サマリー（1,000回試行）')


if __name__ == "__main__":
  main()
