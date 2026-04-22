"""
Optimal Strategy V2 モデルデバッグツール。

既に生成されたモデル（data/optimal_strategy_v2_models.json）を読み込み、
特定の年齢における特定の支出率（R）の生存確率を再評価したり、
特定のパス（シナリオ）を最後まで追跡したりするためのツールです。

主な機能:
1. 指定した年齢（--age）の評価を、翌年のフィッティング済みモデルを用いて実行。
2. 複数の支出率（--test_r）に対する生存確率と、代表的な資産配分（A）での結果を表示。
3. 全年齢を再計算することなく、特定の年齢の挙動（例: 4%ルールの成功率）を確認可能。
4. 最適な資産配分（Best A）における翌年の資産状態（X_next）のパーセンタイル分析を表示。
5. 指定したパスインデックス（--trace）について、指定年齢から95歳までの推移を追跡。

使用例:
  # 40歳で支出率 0.04, 0.05, 0.10 をテスト
  python src/model_fitting_v2_debug.py --age 40 --test_r 0.04,0.05,0.10

  # 60歳で詳細なシミュレーション（n_sim=5000）を実行
  python src/model_fitting_v2_debug.py --age 60 --test_r 0.04 --n_sim 5000

  # パス 0 の 40歳から 95歳までの推移を表示
  python src/model_fitting_v2_debug.py --age 40 --test_r 0.04 --trace 0
"""

import argparse
from typing import Any, List, Union, cast

import numpy as np

from src.core import (CashflowRule, CashflowType, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (DerivedAsset, ForexAsset,
                                     SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, PensionConfig,
                                        generate_cashflows)
from src.lib.dynamic_rebalance_dp import DPOptimalStrategyPredictor
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)

# 共通定数
START_AGE = 40
END_AGE = 96  # 95歳の終わりまで
YEARS = END_AGE - START_AGE
SEED = 42

# アセット名
ORUKAN_NAME = "オルカン"
ZERO_RISK_NAME = "ゼロリスク資産"
FX_NAME = "USDJPY_0_10.53"
CPI_NAME = "Japan_CPI"
PENSION_CPI_NAME = "Pension_CPI"

# パラメータ
TRUST_FEE = 0.0005775
ZERO_RISK_YIELD = 0.04
TAX_RATE = 0.20315
CURRENT_YEAR = 2026
MACRO_ECONOMIC_SLIDE_END_YEAR = 2057


def main():
  parser = argparse.ArgumentParser(description="Optimal Strategy V2 モデルデバッグツール")
  parser.add_argument("--age", type=int, default=40, help="デバッグ対象の年齢")
  parser.add_argument("--test_r",
                      type=str,
                      default="0.04,0.05,0.10",
                      help="テストする R 値（カンマ区切り）")
  parser.add_argument("--n_sim", type=int, default=1000, help="シミュレーション回数")
  parser.add_argument("--models_path",
                      type=str,
                      default="data/optimal_strategy_v2_models.json",
                      help="モデルファイルのパス")
  parser.add_argument("--trace", type=int, default=None, help="トレースするパスのインデックス")
  args = parser.parse_args()

  n_sim = args.n_sim
  age = args.age
  test_r_vals = [float(r) for r in args.test_r.split(",")]

  # 1. アセットとキャッシュフローの生成
  fx_asset = ForexAsset(name=FX_NAME,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  base_sp500 = get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y)
  base_acwi = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)
  orukan = DerivedAsset(name=ORUKAN_NAME,
                        base=base_acwi.name,
                        trust_fee=TRUST_FEE,
                        forex=FX_NAME)
  zr_asset_obj = ZeroRiskAsset(name=ZERO_RISK_NAME, yield_rate=ZERO_RISK_YIELD)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)
  pension_cpi = SlideAdjustedCpiAsset(
      name=PENSION_CPI_NAME,
      base_cpi=CPI_NAME,
      slide_rate=0.005,
      slide_end_month=(MACRO_ECONOMIC_SLIDE_END_YEAR - CURRENT_YEAR) * 12)

  asset_configs = [
      fx_asset, base_sp500, base_acwi, orukan, base_cpi, pension_cpi
  ]
  monthly_prices = generate_monthly_asset_prices(asset_configs,
                                                 n_paths=n_sim,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  cf_configs: List[CashflowConfig] = []
  cf_configs.append(
      PensionConfig(name="Pension_Premium_Kiso",
                    amount=-20.4 / 12.0,
                    start_month=0,
                    end_month=(60 - START_AGE) * 12,
                    cpi_name=CPI_NAME))
  reduction_rate = 0.76
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kousei",
                    amount=(49.2 * reduction_rate) / 12.0,
                    start_month=(60 - START_AGE) * 12,
                    cpi_name=CPI_NAME))
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kiso",
                    amount=(81.6 * reduction_rate) / 12.0,
                    start_month=(60 - START_AGE) * 12,
                    cpi_name=PENSION_CPI_NAME))

  spending_monthly_values = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=START_AGE,
      num_years=YEARS,
      normalize=False)
  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=n_sim,
                                         n_months=YEARS * 12)

  # 2. 予測器のロード
  predictor = DPOptimalStrategyPredictor(args.models_path)

  if args.trace is not None:
    path_idx = args.trace
    if path_idx >= n_sim:
      print(f"Error: path index {path_idx} out of range (n_sim={n_sim})")
      return

    # トレース開始
    print(f"\n--- Tracing Path {path_idx} from Age {age} to 95 ---")
    print(
        f"{'Age':>3} | {'X':>10} | {'Y':>8} | {'R':>7} | {'Pred A':>7} | {'P(N)':>6} | {'P(N+1)':>6} | {'Orukan P':>8}"
    )
    print("-" * 83)

    def get_y_withdraw(target_age, p_idx):
      y_idx = target_age - START_AGE
      if y_idx < 0 or y_idx >= YEARS:
        return 0.0
      sm = y_idx * 12
      em = (y_idx + 1) * 12
      cp = monthly_prices[CPI_NAME][p_idx, sm:em]
      msb = spending_monthly_values[y_idx] / 10000.0
      mns = msb * cp
      pp = monthly_cashflows["Pension_Premium_Kiso"][p_idx, sm:em]
      pk = monthly_cashflows["Pension_Receipt_Kousei"][p_idx, sm:em]
      pi = monthly_cashflows["Pension_Receipt_Kiso"][p_idx, sm:em]
      mns -= (pp + pk + pi)
      return np.sum(np.maximum(0, mns))

    first_y = get_y_withdraw(age, path_idx)
    first_r = test_r_vals[0]
    x_n = first_y / first_r

    for cur_age in range(age, 96):
      y_n = get_y_withdraw(cur_age, path_idx)
      r_n = y_n / np.maximum(x_n, 1e-7)

      # 予測器を使用して A と P を取得
      # モデルが存在しない場合は例外が発生しプログラムが終了する
      pred_a = cast(float, predictor.predict_a_opt(cur_age, r_n))
      p_n = cast(float, predictor.predict_p_surv(cur_age, r_n))

      # 1年シミュレーション
      y_idx = cur_age - START_AGE
      sm = y_idx * 12
      em = (y_idx + 1) * 12
      # 指定したパスのみシミュレーション
      year_prices = {
          k: v[path_idx:path_idx + 1, sm:em + 1]
          for k, v in monthly_prices.items()
          if k != ZERO_RISK_NAME
      }

      # path_idx の monthly_net_spend を再計算
      cp = monthly_prices[CPI_NAME][path_idx, sm:em]
      msb = spending_monthly_values[y_idx] / 10000.0
      mns = msb * cp
      pp = monthly_cashflows["Pension_Premium_Kiso"][path_idx, sm:em]
      pk = monthly_cashflows["Pension_Receipt_Kousei"][path_idx, sm:em]
      pi = monthly_cashflows["Pension_Receipt_Kiso"][path_idx, sm:em]
      mns -= (pp + pk + pi)
      year_cf = {"Net_Spend": -mns.reshape(1, 12)}

      strategy = Strategy(
          name=f"trace_path{path_idx}_age{cur_age}",
          initial_money=np.array([x_n]),
          initial_loan=0.0,
          yearly_loan_interest=0.0,
          initial_asset_ratio={
              ORUKAN_NAME: pred_a,
              zr_asset_obj: 1.0 - pred_a
          },
          annual_cost=0.0,
          inflation_rate=None,
          selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
          tax_rate=TAX_RATE,
          rebalance_interval=0,
      )
      strategy.cashflow_rules = [
          CashflowRule(source_name="Net_Spend",
                       cashflow_type=CashflowType.REGULAR)
      ]
      res = simulate_strategy(strategy,
                              year_prices,
                              monthly_cashflows=year_cf,
                              fallback_total_months=12,
                              calculate_post_tax=True)

      assert res.post_tax_net_values is not None
      x_next = float(res.post_tax_net_values[0])

      # P(N+1)
      p_next_str = "N/A"
      if x_next > 0:
        y_next = get_y_withdraw(cur_age + 1, path_idx)
        r_next = y_next / x_next
        try:
          p_next_val = predictor.predict_p_surv(cur_age + 1, r_next)
          p_next_str = f"{p_next_val:6.4f}"
        except ValueError:
          pass
      else:
        p_next_str = f"{0.0:6.4f}"

      orukan_price = monthly_prices[ORUKAN_NAME][path_idx, sm]
      print(
          f"{cur_age:3d} | {x_n:10.2f} | {y_n:8.2f} | {r_n:7.4f} | {pred_a:7.4f} | {p_n:6.4f} | {p_next_str:>6} | {orukan_price:8.2f}"
      )

      x_n = x_next
      if x_n <= 0:
        print("Bankrupt!")
        break

    return

  # 3. 評価セットアップ
  year_idx = age - START_AGE
  start_m = year_idx * 12
  end_m = (year_idx + 1) * 12
  cpi_path = monthly_prices[CPI_NAME][:, start_m:end_m]

  monthly_net_spend = np.zeros((n_sim, 12))
  monthly_spend_base = spending_monthly_values[year_idx] / 10000.0
  monthly_net_spend += monthly_spend_base * cpi_path

  p_premium = monthly_cashflows["Pension_Premium_Kiso"][:, start_m:end_m]
  p_kousei = monthly_cashflows["Pension_Receipt_Kousei"][:, start_m:end_m]
  p_kiso = monthly_cashflows["Pension_Receipt_Kiso"][:, start_m:end_m]
  monthly_net_spend -= (p_premium + p_kousei + p_kiso)
  y_withdraw_n = np.sum(np.maximum(0, monthly_net_spend), axis=1)

  # デバッグログ: 支出のの内訳を表示
  avg_cpi = np.mean(cpi_path)
  avg_spend_base = monthly_spend_base * avg_cpi
  avg_p_premium = -np.mean(p_premium) # 支払額なので正負反転
  print(f"\n[DEBUG] Withdrawal (Y_{age}) Breakdown (Mean):")
  print(f"  - Base Spend (Consumption + Non-Consumption Excl. Pension) * CPI: {avg_spend_base * 12:.2f} 万円/年")
  print(f"  - Pension Premium (国民年金): {avg_p_premium * 12:.2f} 万円/年")
  print(f"  - Total Withdrawal: {np.mean(y_withdraw_n):.2f} 万円/年")
  print(f"  (Note: Statistics for working households at age 40 show ~504万/year, but that includes ~46万 of 厚生年金保険料 which is removed here because you are retired.)")

  # 翌年の情報
  next_age = age + 1
  next_model = predictor.get_p_surv_model(next_age)

  next_year_idx = next_age - START_AGE
  next_start_m = next_year_idx * 12
  next_end_m = (next_year_idx + 1) * 12
  next_cpi_path = monthly_prices[CPI_NAME][:, next_start_m:next_end_m]
  next_monthly_spend_base = spending_monthly_values[next_year_idx] / 10000.0
  next_p_premium = monthly_cashflows["Pension_Premium_Kiso"][:,
                                                        next_start_m:next_end_m]
  next_p_kousei = monthly_cashflows[
      "Pension_Receipt_Kousei"][:, next_start_m:next_end_m]
  next_p_kiso = monthly_cashflows[
      "Pension_Receipt_Kiso"][:, next_start_m:next_end_m]
  next_monthly_net_spend = next_monthly_spend_base * next_cpi_path - (
      next_p_premium + next_p_kousei + next_p_kiso)
  next_y_withdraw = np.sum(np.maximum(0, next_monthly_net_spend), axis=1)

  # 現在の年齢のモデル (ブルートフォースと比較用)
  try:
    current_a_model = predictor.get_a_opt_model(age)
  except ValueError:
    current_a_model = None

  print(f"\n--- Debugging Age {age} (Using Age {age+1} model for survival) ---")
  if current_a_model:
    print(f"Current Age {age} A_opt model loaded")
  print(
      f"Age {age+1} Boundaries: R_min={next_model.r_min_p:.4f}, R_max={next_model.r_max_p:.4f}, P_max={next_model.p_max:.4f}, P_min={next_model.p_min:.4f}"
  )

  # Y_age の分布を表示
  y_25 = np.percentile(y_withdraw_n, 25)
  y_50 = np.percentile(y_withdraw_n, 50)
  y_75 = np.percentile(y_withdraw_n, 75)
  print(
      f"Withdrawal Amount (Y_{age}) Distribution (万円/年): 25%ile={y_25:.2f}, 50%ile={y_50:.2f}, 75%ile={y_75:.2f}"
  )

  year_prices_all = {
      k: v[:, start_m:end_m + 1]
      for k, v in monthly_prices.items()
      if k != ZERO_RISK_NAME
  }

  year_cf = {"Net_Spend": -monthly_net_spend}

  for r in test_r_vals:
    x_p_n = y_withdraw_n / r
    x_avg = np.mean(x_p_n)
    x_50 = np.percentile(x_p_n, 50)
    print(
        f"\nTesting R={r:.4f} (Initial Capital: Avg={x_avg:.2f}, 50%ile={x_50:.2f} 万円)"
    )

    if not current_a_model:
      print(
          f"  Warning: A_opt model for age {age} not found. Skipping analysis for this R."
      )
      continue

    # 1. 予測された A の計算
    predicted_a = cast(float, predictor.predict_a_opt(age, r))

    # 2. テストする A の集合を決定
    # 0.0, 1.0, および Predicted A に最も近い 0.1 の倍数 2 つ
    a_test_set = {0.0, 1.0, predicted_a}
    a_test_set.add(np.clip(np.floor(predicted_a * 10) / 10.0, 0, 1))
    a_test_set.add(np.clip(np.ceil(predicted_a * 10) / 10.0, 0, 1))
    sorted_a_tests = sorted(list(a_test_set))

    # 3. シミュレーション実行
    results_for_analysis = {}  # a -> (res, x_next, r_next, p_next, avg_surv)

    for a in sorted_a_tests:
      strategy = Strategy(name=f"test_r{r}_a{a}",
                          initial_money=x_p_n,
                          initial_loan=0.0,
                          yearly_loan_interest=0.0,
                          initial_asset_ratio={
                              ORUKAN_NAME: a,
                              zr_asset_obj: 1.0 - a
                          },
                          annual_cost=0.0,
                          inflation_rate=None,
                          selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
                          tax_rate=TAX_RATE,
                          rebalance_interval=0)
      strategy.cashflow_rules = [
          CashflowRule(source_name="Net_Spend",
                       cashflow_type=CashflowType.REGULAR)
      ]
      res = simulate_strategy(strategy,
                              year_prices_all,
                              monthly_cashflows=year_cf,
                              fallback_total_months=12,
                              calculate_post_tax=True)
      x_next_arr = cast(np.ndarray, res.post_tax_net_values)
      bankrupt_this_year = res.sustained_months < 12

      r_next = next_y_withdraw / np.maximum(x_next_arr, 1e-7)

      # ベクトル化された生存確率予測
      p_next = predictor.predict_p_surv(next_age, r_next)

      # 破産したパスの生存確率を強制的に 0.0 に設定（p_minが0でない場合のため）
      if isinstance(p_next, np.ndarray):
        p_next[bankrupt_this_year] = next_model.p_min

      avg_surv = np.mean(p_next)
      results_for_analysis[a] = (res, x_next_arr, r_next, p_next, float(avg_surv))

    # 4. 出力
    # results_for_analysis[a] の型を Mypy に教える
    def get_analysis_result(a_val: float) -> Any:
      return results_for_analysis[a_val]

    res_pred = get_analysis_result(predicted_a)
    print(
        f"  Model Predicted A_opt: {predicted_a:.4f} -> Survival: {res_pred[4]:.4f}"
    )
    for a in sorted_a_tests:
      if a == predicted_a:
        continue
      res_a = get_analysis_result(a)
      print(f"  A={a:.1f}: Avg Survival={res_a[4]:.4f}")

    # 詳細分析 (Predicted A に対して)
    _, x_next_pred_arr, r_next_pred_arr, p_next_pred_arr, _ = res_pred
    x_next_pred = cast(np.ndarray, x_next_pred_arr)
    r_next_pred = cast(np.ndarray, r_next_pred_arr)
    p_next_pred = cast(np.ndarray, p_next_pred_arr)

    print(
        f"  [Analysis for predicted A={predicted_a:.4f}] State at Start of Age {age+1}:"
    )
    percentiles = [25, 50, 75]
    for p in percentiles:
      val = np.percentile(x_next_pred, p)
      idx = int(np.abs(x_next_pred - val).argmin())
      print(
          f"    [X-Percentile {p}%] Path {idx}: X_{age+1}={x_next_pred[idx]:.2f}, Y_{age+1}={next_y_withdraw[idx]:.2f}, R_{age+1}={r_next_pred[idx]:.4f}, P_surv={p_next_pred[idx]:.4f}"
      )

    print("")
    for p in percentiles:
      val = np.percentile(r_next_pred, p)
      idx = int(np.abs(r_next_pred - val).argmin())
      print(
          f"    [R-Percentile {p}%] Path {idx}: X_{age+1}={x_next_pred[idx]:.2f}, Y_{age+1}={next_y_withdraw[idx]:.2f}, R_{age+1}={r_next_pred[idx]:.4f}, P_surv={p_next_pred[idx]:.4f}"
      )


if __name__ == "__main__":
  main()
