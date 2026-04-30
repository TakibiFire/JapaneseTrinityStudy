"""
年金受給が資産寿命に与える影響をグリッドサーチで分析するスクリプト。

実験設定:
- 期間: 60年
- 試行回数: 5,000回
- 資産構成: オルカン 100% (7%, 15%, 信託報酬 0.05775%)
- CPI: AR(12) 粘着性モデル (1970年〜)
- 税率: 20.315%

シナリオ設定 (A~G):
- A (年金なし): 年金制度が存在しない世界。支出 = S。
- B (Sanity check): 保険料を払うが受給しない。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- C (継続・60歳): 保険料を60歳まで納付、60歳繰り上げ受給。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- D (継続・65歳): 保険料を60歳まで納付、65歳受給。支出 = S-21.5、保険料 = 21.5 (60歳まで)。
- E (免除・60歳): N歳(リタイア開始)から全額免除、60歳繰り上げ受給。支出 = S-21.5、保険料 = 0。
- F (免除・65歳): N歳から全額免除、65歳受給。支出 = S-21.5、保険料 = 0。
- G (未納・65歳): N歳から未納(放置)、65歳受給。支出 = S-21.5、保険料 = 0。

計算上の前提:
- 厚生年金: 22歳からリタイア開始(N歳)まで加入。年収500万を想定し、年額 = 2.736 * (N-22) 万。
- 基礎年金 (満額): 年額 81.6万。
- 免除時の基礎年金: 免除期間(N~60歳)の受給額は 1/2 として計算。
- 繰り上げ受給 (60歳): 受給額を 76% (0.4% * 60ヶ月減額) とする。
- マクロ経済スライド: 基礎年金にのみ適用し、2057年度に終了すると想定。厚生年金はCPI連動のみ。
- 国民年金保険料: 年額 21.5万 (CPI連動) とし、60歳の誕生日前まで支払う。
"""

import argparse
import os
from dataclasses import replace
from itertools import product
from typing import Any, Dict, List

import pandas as pd

from src.core import simulate_strategy
from src.lib.scenario_builder import (ConstantSpend, Lifeplan, PensionStatus,
                                      PredefinedStock, Setup, StrategySpec,
                                      WorldConfig, create_experiment_setup)

# 設定
DATA_DIR = "data/pension"


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(description="年金受給のグリッドシミュレーションを実行する")
  parser.add_argument("--exp_name",
                      type=str,
                      default="exp1",
                      help="実験名 (exp1)。カンマ区切りで複数指定可能")
  args = parser.parse_args()

  exp_names = [name.strip() for name in args.exp_name.split(",")]

  for exp_name in exp_names:
    run_experiment(exp_name)


def run_experiment(exp_name: str):
  # 共通設定
  N_SIM = 5000
  YEARS = 60
  SEED = 42

  os.makedirs(DATA_DIR, exist_ok=True)
  csv_path = os.path.join(DATA_DIR, f"{exp_name}.csv")

  # グリッドパラメータ
  initial_money_annual_cost_list = [(5000, 200), (10000, 400), (20000, 800)]
  initial_age_list = [30, 40, 50, 60]

  if exp_name == "exp1":
    # (PensionStatus, p_start_age, label)
    scenarios = [
        (PensionStatus.NONE, 65, "NoPensionWorld"),
        (PensionStatus.FULL, 0, "PayNoReceive"),  # p_start_age=0 is dummy
        (PensionStatus.FULL, 60, "Pay_60"),
        (PensionStatus.FULL, 65, "Pay_65"),
        (PensionStatus.FULL, 70, "Pay_70"),
        (PensionStatus.FULL, 75, "Pay_75"),
        (PensionStatus.EXEMPT, 60, "Exempt_60"),
        (PensionStatus.EXEMPT, 65, "Exempt_65"),
        (PensionStatus.UNPAID, 65, "Unpaid_65"),
    ]
  else:
    print(f"Skipping unknown exp_name: {exp_name}")
    return

  all_combinations = list(
      product(initial_money_annual_cost_list, initial_age_list, scenarios))

  results: List[Dict[str, Any]] = []

  print(f"全 {len(all_combinations)} パターンのシミュレーションを実行中...")

  # 各 init_age ごとに WorldConfig が変わるので、大きなループを回すか、
  # Setup を適切に使う。ここでは init_age ごとに Setup を作るのが素直だが、
  # create_experiment_setup は複数の WorldConfig を扱えるので、1つの Setup に全部入れる。

  # ダミーのベースライン
  dummy_lp = Lifeplan(retirement_start_age=30,
                      base_spend=ConstantSpend(annual_amount=200))
  dummy_strategy = StrategySpec(
      initial_money=5000,
      initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),),
      selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN,))
  exp_setup = Setup(name="dummy_pension",
                    world=WorldConfig(n_sim=N_SIM,
                                      n_years=YEARS,
                                      start_age=30,
                                      seed=SEED),
                    lifeplan=dummy_lp,
                    strategy=dummy_strategy)

  # すべての実験を追加
  for (init_money, annual_cost), init_age, (status, p_start,
                                            label) in all_combinations:
    # 支出調整: NoPensionWorld 以外は保険料分(21.5万)を生活費から引く
    # scenario_builder 内の BaseSpend は ConstantSpend.annual_amount をそのまま使う
    # 旧コードでは current_annual_cost -= PREMIUM_ANNUAL していた
    actual_annual_cost = float(annual_cost)
    if status != PensionStatus.NONE:
      actual_annual_cost -= 21.5

    # PayNoReceive は受給開始年齢を非常に高く設定することで「受給しない」をシミュレート
    # (または受給額を0にする必要があるが、ここでは受給開始を YEARS+init_age 以降にすればよい)
    actual_p_start = p_start if label != "PayNoReceive" else (init_age + YEARS +
                                                              1)
    # NoPensionWorld の表示上の差分をなくすため 0 に設定
    if label == "NoPensionWorld":
      actual_p_start = 0

    world = WorldConfig(n_sim=N_SIM,
                        n_years=YEARS,
                        start_age=init_age,
                        seed=SEED)
    lp = Lifeplan(base_spend=ConstantSpend(annual_amount=actual_annual_cost),
                  retirement_start_age=init_age,
                  pension_status=status,
                  pension_start_age=actual_p_start)
    strategy = StrategySpec(
        initial_money=float(init_money),
        initial_asset_ratio=((PredefinedStock.SIMPLE_7_15_ORUKAN, 1.0),),
        selling_priority=(PredefinedStock.SIMPLE_7_15_ORUKAN,))

    exp_setup.add_experiment(
        name=f"{init_money}_{annual_cost}_{init_age}_{label}",
        overwrite_world=world,
        overwrite_lifeplan=lp,
        overwrite_strategy=strategy)

  # コンパイル実行
  compiled_exps = create_experiment_setup(exp_setup)

  # 実行と記録 (ベースライン[0]はダミーなので飛ばす)
  for i, exp in enumerate(compiled_exps[1:]):
    if i % 10 == 0:
      print(f"Progress: {i}/{len(all_combinations)}")

    # 元のループ変数を取り出す
    (init_money, annual_cost), init_age, (status, p_start,
                                          label) = all_combinations[i]

    res = simulate_strategy(exp.strategy,
                            exp.monthly_prices,
                            monthly_cashflows=exp.monthly_cashflows)

    # 旧コードと互換性のある nominal 計算 (記録用)
    # scenario_builder の内部ロジックを再現
    kousei_unit_annual = 2.736
    kiso_full_annual = 81.6
    reduction_rate = 1.0
    actual_p_start = p_start if label != "PayNoReceive" else 999

    if actual_p_start < 65:
      reduction_rate = 1.0 - 0.004 * (65 - actual_p_start) * 12
    else:
      reduction_rate = 1.0 + 0.007 * (actual_p_start - 65) * 12

    kousei_nominal = kousei_unit_annual * (init_age - 22) * reduction_rate
    if label == "PayNoReceive" or status == PensionStatus.NONE:
      kousei_nominal = 0.0

    if status == PensionStatus.FULL and label != "PayNoReceive":
      kiso_nominal = kiso_full_annual * reduction_rate
    elif status == PensionStatus.EXEMPT:
      kiso_nominal = (kiso_full_annual *
                      (init_age - 22) / 40.0 + kiso_full_annual *
                      (60 - init_age) / 40.0 * 0.5) * reduction_rate
    elif status == PensionStatus.UNPAID:
      kiso_nominal = (kiso_full_annual *
                      (init_age - 22) / 40.0) * reduction_rate
    else:
      kiso_nominal = 0.0

    row = {
        "initial_money": init_money,
        "initial_annual_cost": annual_cost,
        "initial_age": init_age,
        "scenario": label,
        "pension_start_age": p_start if label != "PayNoReceive" else 0,
        "initial_pension_nominal_annual": kousei_nominal + kiso_nominal
    }
    for year in range(1, YEARS + 1):
      bankrupt_count = (res.sustained_months < year * 12).sum()
      survival_rate = 1.0 - (bankrupt_count / N_SIM)
      row[str(year)] = survival_rate

    results.append(row)

  # CSV保存
  df = pd.DataFrame(results)
  df.to_csv(csv_path, index=False, encoding="utf-8-sig")
  print(f"完了。結果を {csv_path} に保存しました。")


if __name__ == "__main__":
  main()
