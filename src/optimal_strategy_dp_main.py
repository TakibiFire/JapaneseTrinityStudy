"""
Optimal Strategy のモデルフィッティングを行うスクリプト。

このスクリプトは、後ろ向き動的計画法（Backward DP）を用いて、各年齢における
最適な資産配分（オルカン比率）と、その時の生存確率を計算し、回帰モデルとして保存します。

状態変数として「年間支出率 R」を採用しています：
  R = 年間の純支出合計 / 年始の総資産
ここで、純支出合計は（支出 - 年金受取 + 年金保険料）の月次合計のうち、正の値を合算したものです。

アルゴリズムの概要：
1. 最終年齢から開始し、開始年齢まで1年ずつ遡ります。
2. 各年齢において、R のグリッド（0.005から20以上まで）を作成します。
3. 各 R に対して、オルカン比率 A（0.0から1.0）を変化させて1年間のシミュレーションを実行します。
4. 翌年の生存確率モデルを用いて、期待生存確率を最大化する最適な A (A_opt) を見つけます。
5. R と A_opt、および R と生存確率の関係を多項式回帰モデルでフィッティングします。
6. 結果を `data/optimal_strategy_dp/${scenario}.json` に保存します。

実行方法:
  python src/optimal_strategy_dp_main.py --scenario re40_pen60_95 --n_sim 2000
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from sklearn.isotonic import IsotonicRegression

import src.lib.world_setup as world_setup
from src.core import (CashflowRule, CashflowType, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.scenario_builder import create_experiment_setup

# 共通定数
SEED = 42

# アセット名
ORUKAN_NAME = "ORUKAN_155"
ZERO_RISK_NAME = "ZERO_RISK_4PCT"
CPI_NAME = "Japan_CPI"

# パラメータ
TAX_RATE = 0.20315
EFFECTIVE_ZERO_RISK_YIELD = 0.04 * (1.0 - TAX_RATE)


def adaptive_sample(evaluate_fn: Any,
                    r_start: float,
                    r_end: float,
                    threshold_a: float = 0.1,
                    threshold_p: float = 0.02,
                    max_depth: int = 3,
                    r_min_a: Optional[float] = None,
                    r_max_a: Optional[float] = None,
                    current_depth: int = 0) -> None:
  """
  R の範囲 [r_start, r_end] において、a_max または p が線形補間から大きく乖離する場合のみ
  再帰的に二分探索してサンプリング密度を高めます。

  r_min_a, r_max_a が指定されている場合、その範囲外では a を固定してサンプリングを高速化します。
  """
  if max_depth <= 0:
    return

  # 端点の評価
  res_start = evaluate_fn(r_start, stage="適応的サンプリング", depth=current_depth)
  res_end = evaluate_fn(r_end, stage="適応的サンプリング", depth=current_depth)

  a_max_start, p_start = res_start[4], res_start[1]
  a_max_end, p_end = res_end[4], res_end[1]

  r_mid = (r_start + r_end) / 2

  # A のサンプリングを高速化するか判定
  a_fixed = None
  reason = ""
  if r_min_a is not None and r_max_a is not None:
    if r_mid < r_min_a:
      a_fixed = 1.0
      reason = f"R={r_mid:.4f} < R_min_a={r_min_a:.4f} なので A=1.0 に固定"
    elif r_mid > r_max_a:
      if abs(a_max_start - a_max_end) < 1e-4:
        a_fixed = a_max_start
        reason = f"R={r_mid:.4f} > R_max_a={r_max_a:.4f} かつ端点の A_opt が一致するため A={a_fixed:.2f} に固定"
    else:
      # 遷移領域内でも、端点の A_opt が一致していれば固定を試みる（高速化）
      if abs(a_max_start - a_max_end) < 1e-4:
        a_fixed = a_max_start
        reason = f"R={r_mid:.4f} は遷移領域内だが端点の A_opt が一致するため A={a_fixed:.2f} に固定"

  res_mid = evaluate_fn(r_mid,
                        a_fixed=a_fixed,
                        stage="適応的サンプリング",
                        depth=current_depth,
                        reason=reason,
                        segment=(r_start, r_end),
                        segment_a_opts=(a_max_start, a_max_end))
  a_max_mid, p_mid = res_mid[4], res_mid[1]

  # 線形補間値との差分
  a_max_linear = (a_max_start + a_max_end) / 2.0
  p_linear = (p_start + p_end) / 2.0

  if abs(a_max_mid - a_max_linear) > threshold_a or abs(p_mid -
                                                        p_linear) > threshold_p:
    # 乖離が大きい場合のみ、さらに深く探索
    adaptive_sample(evaluate_fn, r_start, r_mid, threshold_a, threshold_p,
                    max_depth - 1, r_min_a, r_max_a, current_depth + 1)
    adaptive_sample(evaluate_fn, r_mid, r_end, threshold_a, threshold_p,
                    max_depth - 1, r_min_a, r_max_a, current_depth + 1)


def filter_anchors(r: np.ndarray, y: np.ndarray,
                   threshold: float) -> Tuple[np.ndarray, np.ndarray]:
  """
  線形補間からの乖離が閾値以下になるように、アンカーポイントを削減します。
  """
  if len(r) <= 2:
    return r, y

  indices = [0, len(r) - 1]

  def refine(start_idx: int, end_idx: int):
    if end_idx - start_idx <= 1:
      return

    r_sub = r[start_idx:end_idx + 1]
    y_sub = y[start_idx:end_idx + 1]

    y_linear = np.interp(r_sub, [r[start_idx], r[end_idx]],
                         [y[start_idx], y[end_idx]])
    deviations = np.abs(y_sub - y_linear)
    max_dev_idx = np.argmax(deviations)

    if deviations[max_dev_idx] > threshold:
      actual_idx = int(start_idx + max_dev_idx)
      if actual_idx not in indices:
        indices.append(actual_idx)
        refine(start_idx, actual_idx)
        refine(actual_idx, end_idx)

  refine(0, len(r) - 1)
  final_indices = sorted(indices)
  return r[final_indices], y[final_indices]


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="Optimal Strategy V2 のモデルフィッティング")
  parser.add_argument("--scenario",
                      type=str,
                      default="re40_pen60_95",
                      help="シナリオ名 (src/lib/world_setup.py 内の関数名)")
  parser.add_argument("--n_sim", type=int, default=2000, help="シミュレーション回数")
  parser.add_argument("--debug_level",
                      type=int,
                      default=0,
                      help="デバッグレベル (0: 通常, 1: 詳細, 2: 超詳細, 3: キャッシュフロー詳細)")
  parser.add_argument("--debug_age",
                      type=int,
                      default=None,
                      help="特定の年齢の詳細なフィッティングデータを表示する")
  parser.add_argument("--debug_paths",
                      type=str,
                      default=None,
                      help="デバッグ情報を表示するパスのインデックス（カンマ区切り、例: 0,1,2）")
  parser.add_argument("--net_prediction",
                      type=str,
                      choices=["ar1_residual"],
                      default="ar1_residual",
                      help="来年の支出額（および支出率 R）の予測手法")
  parser.add_argument("--output_path",
                      type=str,
                      default=None,
                      help="出力ファイルのパス。指定しない場合はデフォルトのパスを使用")
  parser.add_argument(
      "--tie_breaker_method",
      type=str,
      choices=["legacy", "goal_based", "survival_first_goal_based"],
      default="survival_first_goal_based",
      help=
      "タイブレークの手法 (legacy: 最大の A を選択, goal_based: 勝利確率を考慮, survival_first_goal_based: 生存確率 100% の場合のみ勝利確率を考慮)"
  )
  parser.add_argument("--disable_shortcuts",
                      action="store_true",
                      help="A のサンプリングの高速化（ショートカット）を無効化する")
  parser.add_argument("--min_y_withdraw",
                      type=float,
                      default=5.0,
                      help="初期資産 X を計算する際の純支出 Y の下限 (万円)。Y=0 の時の特異点を回避するために使用。")
  args = parser.parse_args()

  # 互換性維持のための後処理
  net_prediction = args.net_prediction

  n_sim = args.n_sim
  debug = args.debug_level > 0
  debug_paths = [int(p) for p in args.debug_paths.split(",")
                ] if args.debug_paths else []

  # 1. アセットとキャッシュフローの生成
  setup_fn = getattr(world_setup, args.scenario)
  setup = setup_fn(n_sim=n_sim, seed=SEED)
  exp = create_experiment_setup(setup)[0]

  # 共通定数の抽出
  start_age = setup.world.start_age
  years = setup.world.n_years
  end_age = start_age + years  # 排他的な上限 (例: start_age=40, years=56 なら end_age=96)

  monthly_prices = exp.monthly_prices
  monthly_cashflows = exp.monthly_cashflows
  cf_map = exp.cf_name_map
  # 支出の統計的な成長率を計算するために、ベース支出の実質推移を取得する
  spending_annual_real = exp.annual_cost_real
  assert spending_annual_real is not None

  zr_asset_obj = ZeroRiskAsset(ZERO_RISK_NAME, 0.04)

  print(f"Scenario: {args.scenario}")
  print(f"Generating asset prices for {years} years, {n_sim} paths...")

  # CPI の統計計算 (想定外のジャンプを計算するため)
  # 年次 CPI 倍率の平均と標準偏差を計算
  cpi_data = monthly_prices[CPI_NAME]
  annual_cpi_jumps = []
  for y in range(years):
    # 年始 (前年末) から年末への倍率
    if y == 0:
      jumps = cpi_data[:, 11] / 1.0  # 初期値は 1.0
    else:
      jumps = cpi_data[:, (y + 1) * 12 - 1] / cpi_data[:, y * 12 - 1]
    annual_cpi_jumps.extend(jumps.tolist())

  cpi_annual_mu = float(np.mean(annual_cpi_jumps)) - 1.0
  cpi_annual_sigma = float(np.std(annual_cpi_jumps))
  # 99%ile (Z=2.326) の想定外ジャンプ倍率
  unexpected_cpi_jump = (1.0 + cpi_annual_mu +
                         2.326 * cpi_annual_sigma) / (1.0 + cpi_annual_mu)
  print(
      f"CPI Stats: mu={cpi_annual_mu:.4f}, sigma={cpi_annual_sigma:.4f}, unexpected_jump={unexpected_cpi_jump:.4f}"
  )

  print("Analyzing cashflows...")
  # 各年齢のキャッシュフローデータを抽出
  # age -> y_withdraw_n (np.ndarray)
  age_cashflow_data: Dict[int, np.ndarray] = {}
  # age -> gross_spend_n (np.ndarray)
  age_gross_spend_data: Dict[int, np.ndarray] = {}

  for age in range(start_age, end_age):
    year_idx = age - start_age
    start_m = year_idx * 12
    end_m = (year_idx + 1) * 12

    monthly_net_spend = np.zeros((n_sim, 12))
    # 基本支出のハッシュ化された名前を取得
    base_spend_key = cf_map["BaseSpend"]
    # 支出額 (名目) は monthly_cashflows に負の値で入っている。
    # 単位は 万円/月
    m_base_spend = -monthly_cashflows[base_spend_key][:, start_m:end_m]
    monthly_net_spend += m_base_spend
    age_gross_spend_data[age] = np.sum(m_base_spend, axis=1)

    if args.debug_level >= 3 and age == args.debug_age:
      print(
          f"    [Age {age} Debug] Logical: BaseSpend, Hashed: {base_spend_key}")
      print(
          f"    [Age {age} Debug] BaseSpend (Month 0, path 0): {monthly_cashflows[base_spend_key][0, start_m]:.2f}"
      )
      # If age >= end_age - 1, dump all monthly_net_spend for each path.
      if age >= end_age - 1:
        print(
            f"    [Age {age} Debug] Path 0 full monthly_net_spend (reversed sign):"
        )
        print(monthly_cashflows[base_spend_key][0, start_m:end_m])

    # 年金等 (ハッシュ化された名前で検索)
    pension_total = np.zeros((n_sim, 12))
    for logical_name in ["PensionPremium", "PensionKousei", "PensionKiso"]:
      hashed_name = cf_map.get(logical_name)
      if hashed_name and hashed_name in monthly_cashflows:
        # PensionConfig.generate() は名目万円/月を返す。
        # 収入は正、保険料は負の値。
        cf_array = monthly_cashflows[hashed_name][:, start_m:end_m]
        pension_total += cf_array
        if args.debug_level >= 3 and age == args.debug_age:
          print(
              f"    [Age {age} Debug] Logical: {logical_name}, Hashed: {hashed_name}"
          )
          print(f"    [Age {age} Debug] {logical_name} Path 0 full:")
          print(cf_array[0, :])

    monthly_net_spend -= pension_total
    # 各パスの年間合計正味支出 (Withdrawal amount) 万円/年
    age_cashflow_data[age] = np.sum(np.maximum(0, monthly_net_spend), axis=1)
    if args.debug_level >= 3 and age == args.debug_age:
      print(
          f"    [Age {age} Debug] monthly_net_spend (sum(max(0, ...)), path 0): {age_cashflow_data[age][0]:.2f}"
      )

  # 1.5. AR(1) Residual-Based Regression (Offline)
  ar1_models: Dict[int, Dict[str, Any]] = {}
  print("Computing AR(1) residual models...")
  # Age 60の末（Month 11）のリバランスでは Age 61 のモデルが使用される。
  # 最後のリバランス（Age 93の末）では Age 94 のモデルが使用される。
  # よって start_age + 1 から end_age - 1 までのAR(1)パラメータがあればよい。
  for age in range(start_age, end_age - 1):
    # 次の年齢の実際の支出 (age+1 の1年間の合計)
    y_next_actual = age_cashflow_data[age + 1]  # shape (n_sim,)

    # age+1 の期首（cpi_curr）と age の期首（cpi_prev）の CPI
    # これらを用いて age+1 の支出分布を予測する
    idx_curr = (age + 1 - start_age) * 12
    idx_prev = (age - start_age) * 12

    cpi_curr = monthly_prices[CPI_NAME][:, idx_curr]
    cpi_prev = monthly_prices[CPI_NAME][:, idx_prev]

    # y_next = a * cpi_prev + b * cpi_curr + c の重回帰モデルを解く
    X = np.column_stack([cpi_prev, cpi_curr, np.ones(n_sim)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_next_actual, rcond=None)
    a, b, c = coeffs

    # 残差を計算
    y_fit = a * cpi_prev + b * cpi_curr + c
    residuals = y_next_actual - y_fit

    # ガウス分布の z-score に対応する 7 つのパーセンタイルを抽出
    percentiles = [
        0.1349898, 2.275013, 15.865525, 50.0, 84.134475, 97.724987, 99.86501
    ]
    resid_points = np.percentile(residuals, percentiles).tolist()

    # R^2 (決定係数) を計算
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_next_actual - np.mean(y_next_actual))**2)
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0

    ar1_models[age + 1] = {
        "cpi_prev_coef": float(a),
        "cpi_curr_coef": float(b),
        "intercept": float(c),
        "resid_points": resid_points,
        "r_squared": float(r_sq)
    }
    print(
        f"  Age {age+1} Prediction: Net ~ {a:.4f}*CPI_prev + {b:.4f}*CPI_curr + {c:.4f} (R^2 = {r_sq:.4f})"
    )

  # 2. Backward DP
  models: Dict[str, Any] = {
      "cpi_annual_mu": cpi_annual_mu,
      "cpi_annual_sigma": cpi_annual_sigma,
      "net_prediction": net_prediction,
  }
  # age -> { "y_withdraw": array, "p_model": {coef}, "r_min": float, "r_max": float, "p_min": float, "p_max": float }
  dp_results: Dict[int, Any] = {}

  # 勝利しきい値 W_N の計算用 (PV of all future net spending)
  last_w = 0.0
  last_w_path = np.zeros(n_sim)

  # 年齢 end_age - 1 から start_age まで逆算
  ages_to_process = list(range(end_age - 1, start_age - 1, -1))
  if args.debug_level > 0:
    ages_to_process = list(range(end_age - 1, end_age - 6, -1))  # デバッグ時は直近5年分のみ

  for age in ages_to_process:
    print(f"\n--- Processing age {age} ---")

    # この年のキャッシュフロー (12ヶ月分) のインデックス
    year_idx = age - start_age
    start_m = year_idx * 12
    end_m = (year_idx + 1) * 12
    cpi_path = monthly_prices[CPI_NAME][:, start_m:end_m]

    # 各パスの年間合計正味支出 (Withdrawal amount)
    y_withdraw_n = age_cashflow_data[age]
    if args.debug_level >= 3 and age == args.debug_age:
      print(
          f"    [Age {age} Debug] y_withdraw_n (mean): {np.mean(y_withdraw_n):.2f}"
      )
      print(
          f"    [Age {age} Debug] y_withdraw_n (path 0): {y_withdraw_n[0]:.2f}")

    # 全パスの平均支出額を記録（実験スクリプトでの投影に使用）
    avg_y_withdraw_n = float(np.mean(y_withdraw_n))

    # 勝利しきい値 W_N, M_N の計算 (Legacy)
    if age == end_age - 1:
      # 最終年は 3ヶ月のバッファを載せて計算 (1.25倍)
      w_n = avg_y_withdraw_n * 1.25 / (1.0 + EFFECTIVE_ZERO_RISK_YIELD)
    else:
      w_n = (avg_y_withdraw_n + last_w) / (1.0 + EFFECTIVE_ZERO_RISK_YIELD)

    last_w = w_n

    if avg_y_withdraw_n > 1e-6:
      m_winning_multiplier = w_n / avg_y_withdraw_n
    else:
      m_winning_multiplier = 0.0

    # 堅牢な勝利しきい値マルチプライヤー M_robust の計算 (Gross Spend ベース)
    # 従来の Net Withdrawal ベースの計算では、年金受給開始後に分母が小さくなり、
    # マルチプライヤーが不安定になる問題があった。
    # そこで、より安定した Gross Spend（税・社会保険料等を含む総支出）を分母とし、
    # 各パスごとに将来の必要資金の現在価値（Future PV）を計算することで、
    # 統計的に安定したマルチプライヤーを算出する。
    if age == end_age - 1:
      # 最終年は 3ヶ月のバッファを載せて計算 (1.25倍)
      w_n_path = y_withdraw_n * 1.25 / (1.0 + EFFECTIVE_ZERO_RISK_YIELD)
    else:
      # FuturePV_n = (NetWithdrawal_n + FuturePV_{n+1}) / (1 + yield)
      w_n_path = (y_withdraw_n + last_w_path) / (1.0 +
                                                 EFFECTIVE_ZERO_RISK_YIELD)

    last_w_path = w_n_path

    # 分母として「前年の Gross Spend」を使用する。
    # シミュレーション開始時点（start_age）では前年のデータがないため、
    # 今年のデータを平均成長率で割り戻して推定する。
    if age > start_age:
      prev_gross_spend = age_gross_spend_data[age - 1]
    else:
      # 最初の年は、翌年への投影の逆を行う
      if age + 1 in age_gross_spend_data:
        growth = np.mean(age_gross_spend_data[age + 1]) / np.mean(
            age_gross_spend_data[age])
      else:
        growth = 1.0
      prev_gross_spend = age_gross_spend_data[age] / growth

    # 各パスごとのマルチプライヤー M_n,p = FuturePV_n,p / GrossSpend_{n-1},p
    m_robust_path = w_n_path / np.maximum(prev_gross_spend, 1e-6)
    # 各種パーセンタイルを計算してモデルに記録する
    percentiles = [50, 60, 70, 80, 85, 90, 95, 97, 98, 99]
    m_v2_percentiles = {
        str(p): float(np.percentile(m_robust_path, p)) for p in percentiles
    }
    m_v2_percentiles["MAX"] = float(np.max(m_robust_path))

    print(
        f"  Winning Threshold: M_N={m_winning_multiplier:.4f} (W_N={w_n:.2f})")
    print(
        f"  Robust Threshold: M_V2_50={m_v2_percentiles['50']:.4f}, M_V2_99={m_v2_percentiles['99']:.4f}, M_V2_max={m_v2_percentiles['MAX']:.4f}"
    )

    if args.debug_level >= 2:
      print(f"  [Level 2 Info] Cashflow:")
      print(f"    Avg Y_withdraw_n (Yearly): {np.mean(y_withdraw_n):.2f} 万円/年")

    # R (支出率) と A (オルカン比率) のグリッド
    a_grid = np.linspace(0.0, 1.0, 21)  # 0.05刻み

    # キャッシュ済み evaluate_r の結果: r -> (best_a, best_survival, survivals_per_a, a_min, a_max)
    eval_cache: Dict[float, Tuple[float, float, Dict[float, float], float,
                                  float]] = {}
    # 探索ログ: データの探索過程を記録する
    search_logs: List[Dict[str, Any]] = []
    # 境界の初期化
    r_min_p, r_max_p, r_min_a, r_max_a = None, None, None, None

    def evaluate_r(
        r: float,
        a_fixed: Optional[float] = None,
        stage: str = "",
        depth: Optional[int] = None,
        reason: str = "",
        segment: Optional[Tuple[float, float]] = None,
        segment_a_opts: Optional[Tuple[float, float]] = None,
        zr_asset_obj=zr_asset_obj
    ) -> Tuple[float, float, Dict[float, float], float, float]:
      # キャッシュにあればそれを返す (浮動小数点の誤差を考慮して丸める)
      r_key = round(r, 6)
      if r_key in eval_cache:
        return eval_cache[r_key]

      # 初期資産 X_p,N = Y_withdraw,p,N / r
      # Y=0 の時の特異点を回避するため、Y に下限を設ける。
      # DPの評価対象である「年間支出率 R」は、本来 Y/X で定義される。
      # Y=0 の時、Xがいくらであっても R=0 となるはずだが、現在の逆算ロジック
      # (X = Y/R) では X=0 と算出されてしまい、翌年以降に Y>0 となった際に
      # 確実に破産してしまう。これを避けるため、仮想的な最小支出(Y_min)を設定し、
      # X = max(Y, Y_min) / R とすることで、富裕層（低R）の評価時に妥当な
      # 資産額をシミュレーションに与える。
      y_withdraw_eff = np.maximum(y_withdraw_n, args.min_y_withdraw)
      x_p_n = y_withdraw_eff / r

      # 最適な生存確率
      best_survival = -1.0
      # 最適な勝利確率（tie-breaker用）
      best_p_win = -1.0
      # 最適な資産配分 A
      best_a = 0.0
      # A ごとの生存確率を保持する辞書
      survivals_per_a: Dict[float, float] = {}

      # 探索する A のリストを決定
      search_a_list = [a_fixed] if a_fixed is not None else a_grid

      # ログ項目の準備（シミュレーション実行前に一部記録）
      log_entry = {
          "r":
              float(r),
          "stage":
              stage,
          "depth":
              depth,
          "segment": [float(s) for s in segment] if segment else None,
          "segment_a_opts": [float(a) for a in segment_a_opts]
                            if segment_a_opts else None,
          "tried_as": [float(a) for a in search_a_list],
          "r_min_a":
              float(r_min_a) if r_min_a is not None else None,
          "r_max_a":
              float(r_max_a) if r_max_a is not None else None,
          "decision_reason":
              reason
      }
      search_logs.append(log_entry)

      for a in search_a_list:
        # 12ヶ月のシミュレーション
        strategy = Strategy(
            name=f"DP_age{age}_r{r:.4f}_a{a:.2f}",
            initial_money=x_p_n,  # np.ndarray を渡す
            initial_loan=0.0,
            yearly_loan_interest=0.0,
            initial_asset_ratio={
                ORUKAN_NAME: a,
                zr_asset_obj: 1.0 - a
            },
            selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
            tax_rate=TAX_RATE,
            rebalance_interval=0  # 1年なのでリバランスなし
        )
        year_prices = {
            k: v[:, start_m:end_m + 1]
            for k, v in monthly_prices.items()
            if k != ZERO_RISK_NAME
        }

        # この年齢の正味支出（名目）を再現する
        # y_withdraw_n は年間の合計だが、simulate_strategy は月次の cf を必要とする
        # ここでは、元の monthly_net_spend をそのまま使用する
        year_idx = age - start_age
        start_m_local = year_idx * 12
        end_m_local = (year_idx + 1) * 12

        # monthly_net_spend を再計算 (evaluate_r 内で y_withdraw_n を使うためではなく、月次を渡すため)
        m_net_spend = np.zeros((n_sim, 12))
        m_net_spend -= monthly_cashflows[
            cf_map["BaseSpend"]][:, start_m_local:end_m_local]
        p_total = np.zeros((n_sim, 12))
        for ln in ["PensionPremium", "PensionKousei", "PensionKiso"]:
          hn = cf_map.get(ln)
          if hn and hn in monthly_cashflows:
            p_total += monthly_cashflows[hn][:, start_m_local:end_m_local]
        m_net_spend -= p_total

        year_cf = {"Net_Spend": -m_net_spend}  # 支出は負で渡す
        strategy.cashflow_rules = [
            CashflowRule(source_name="Net_Spend",
                         cashflow_type=CashflowType.REGULAR)
        ]

        # シミュレーション実行
        res = simulate_strategy(strategy,
                                year_prices,
                                monthly_cashflows=year_cf,
                                fallback_total_months=12,
                                calculate_post_tax=True)
        x_next = cast(np.ndarray, res.post_tax_net_values)  # shape (n_sim,)

        # 今年の破産判定
        bankrupt_this_year = res.sustained_months < 12

        # 生存判定
        if age == end_age - 1:
          # 最終年 (95歳) は今年生存していれば P=1.0
          survival = (~bankrupt_this_year).astype(float)
        else:
          # 次年度の生存確率を CPI 分布に基づいて期待値として計算
          # (以前の「未来予知」実装から、不確実性を考慮した確率的 DP に移行)

          # 7点離散近似（標準正規分布）
          # 各点は z=-3, -2, -1, 0, 1, 2, 3 を代表値とし、
          # 境界は -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 とした時の確率密度
          z_scores = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
          weights = np.array([
              0.00620967, 0.06059754, 0.24173034, 0.38292490, 0.24173034,
              0.06059754, 0.00620967
          ])

          # 今年の支出 Y_N から来年の支出 Y_{N+1} の分布を推定
          model = ar1_models[age + 1]

          # 状態変数: cpi_prev (当期首CPI), cpi_curr (次期首CPIの予測シナリオ)
          # 回帰モデルでは y_next = a * cpi_prev + b * cpi_curr + c を使用
          cpi_prev_p = monthly_prices[CPI_NAME][:, start_m]

          # 次期首の CPI シナリオを生成: CPI_next = CPI_curr * (1 + mu + z*sigma)
          cpi_growth_scenarios = (1.0 + cpi_annual_mu +
                                  z_scores * cpi_annual_sigma)
          cpi_curr_p_scenarios = cpi_prev_p[:,
                                            np.newaxis] * cpi_growth_scenarios

          # 回帰式による予測値 (n_sim, 7)
          y_base = model["cpi_prev_coef"] * cpi_prev_p[:, np.newaxis] + model[
              "cpi_curr_coef"] * cpi_curr_p_scenarios + model["intercept"]

          # 残差（ショック）を加えて来年の支出分布を生成
          resid_array = np.array(model["resid_points"])
          y_next_dist = np.maximum(0, y_base + resid_array)

          # 7つの R_next シナリオを計算
          # x_next shape: (n_sim,) -> (n_sim, 7)
          r_next_scenarios = y_next_dist / np.maximum(x_next[:, np.newaxis],
                                                      1e-7)

          # 次年度の生存確率モデルを取得
          next_model = dp_results[age + 1]["p_model"]
          next_r_min = cast(float, dp_results[age + 1]["r_min_p"])
          next_r_max = cast(float, dp_results[age + 1]["r_max_p"])
          next_p_max = cast(float, dp_results[age + 1].get("p_max", 1.0))
          next_p_min = cast(float, dp_results[age + 1].get("p_min", 0.0))

          # 2D配列で生存確率を計算 (n_sim, 7)
          p_next_scenarios = np.zeros((n_sim, 7))

          # マスクの作成
          bankrupt_mask = bankrupt_this_year[:, np.newaxis]
          # ブロードキャストされる
          p_next_scenarios[~bankrupt_mask &
                           (r_next_scenarios <= next_r_min)] = next_p_max
          p_next_scenarios[~bankrupt_mask &
                           (r_next_scenarios >= next_r_max)] = next_p_min

          in_range = ~bankrupt_mask & (r_next_scenarios > next_r_min) & (
              r_next_scenarios < next_r_max)
          if np.any(in_range):
            # pchip_interpolate は 1D 配列を期待するため、flatten して適用
            p_next_scenarios[in_range] = pchip_interpolate(
                next_model["r_points"], next_model["p_points"],
                r_next_scenarios[in_range])

          # 期待値を計算 (各シナリオの重み付き平均)
          survival = np.sum(p_next_scenarios * weights,
                            axis=1)  # shape: (n_sim,)

        # デバッグ情報の表示
        if age == args.debug_age and debug_paths:
          print(f"      [Path Debug] R={r:.4f}, A={a:.2f}")
          for p_idx in debug_paths:
            if p_idx < n_sim:
              # Y_next, R_next は 期待値（z=0, index 3）を表示
              y_next_expected = 0.0
              r_next_expected = 0.0
              if age < end_age - 1:
                y_next_expected = float(y_next_dist[p_idx, 3])
                r_next_expected = float(r_next_scenarios[p_idx, 3])
              print(
                  f"        Path {p_idx}: X_next={x_next[p_idx]:.2f}, Y_next(exp)={y_next_expected:.2f}, R_next(exp)={r_next_expected:.4f}, P_surv={survival[p_idx]:.4f}"
              )

        # 全パスの 平均生存確率
        avg_survival = float(np.mean(survival))
        survivals_per_a[a] = avg_survival

        # タイブレーク用の追加指標: 勝利確率
        p_win = 0.0
        if args.tie_breaker_method in [
            "goal_based", "survival_first_goal_based"
        ]:
          if age == end_age - 1:
            # 最終年は生存確率そのものを勝利確率とする
            p_win = avg_survival
          else:
            # 翌年の Winning Threshold を超える確率を計算
            m_v2 = models[str(age + 1)]["m_winning_multiplier_v2"]
            # 70%ile を目標とする。70%ile の winning threshold が best performance
            # を示したため。
            m_goal = m_v2["70"]
            gross_spend_curr = age_gross_spend_data[age]
            w_next_path = m_goal * gross_spend_curr
            is_win = x_next >= w_next_path
            p_win = float(np.mean(is_win))

        # 最適 A の更新判定
        is_better = False
        if avg_survival > best_survival + 1e-9:
          is_better = True
        elif abs(avg_survival - best_survival) < 1e-9:
          # 生存確率がタイの場合
          if args.tie_breaker_method == "legacy":
            # 以前のロジック: A が大きい方を選択
            if a > best_a:
              is_better = True
          elif args.tie_breaker_method == "goal_based":
            # 新しいロジック: 勝利確率が高い方を選択。それもタイなら A が大きい方を選択
            if p_win > best_p_win + 1e-9:
              is_better = True
            elif abs(p_win - best_p_win) < 1e-9:
              if a > best_a:
                is_better = True
          elif args.tie_breaker_method == "survival_first_goal_based":
            # 生存確率がほぼ 100%（妥協ルールの範囲内）の場合のみ、勝利確率を考慮する
            # n_sim=1000 の場合、1パスの重みは 0.001 であるため、0.999 以上を 100% 圏内とみなす
            if avg_survival >= 0.999 - 1e-9:
              if p_win > best_p_win + 1e-9:
                is_better = True
              elif abs(p_win - best_p_win) < 1e-9:
                if a > best_a:
                  is_better = True
            else:
              # 生存確率が 100% 未満の場合は、従来の A が大きい方を選択
              if a > best_a:
                is_better = True

        if is_better:
          best_survival = avg_survival
          best_p_win = p_win
          best_a = a

      # 許容範囲 [a_min, a_max] の算出 (P >= P_max * 0.999)
      p_max_row = max(survivals_per_a.values())
      threshold_val = p_max_row * 0.999
      valid_a = [
          a_val for a_val, p_val in survivals_per_a.items()
          if p_val >= threshold_val
      ]
      a_min = min(valid_a)
      a_max = max(valid_a)

      result = (float(best_a), float(best_survival), survivals_per_a,
                float(a_min), float(a_max))
      eval_cache[r_key] = result

      # ログに結果を追記
      log_entry["a_opt_result"] = float(best_a)
      log_entry["p_survival_result"] = float(best_survival)
      log_entry["p_win_result"] = float(best_p_win)

      return result

    # 1. R 広域探索 (Exponential Search)
    exp_r_vals = [0.005 * (2**k) for k in range(13)]
    exp_results = []
    for r in exp_r_vals:
      res = evaluate_r(r, stage="広域探索")
      exp_results.append((r, res[0], res[1]))

    p_surv_vals = [res[2] for res in exp_results]
    p_surv_max = max(p_surv_vals)
    p_surv_min = min(p_surv_vals)

    # 2. 境界探索 (Binary Search)
    # R_min_P
    drop_idx = -1
    for i in range(len(p_surv_vals) - 1):
      if p_surv_vals[i] >= p_surv_max - 1e-4 and p_surv_vals[
          i + 1] < p_surv_max - 1e-4:
        drop_idx = i
        break
    if drop_idx != -1:
      r_low, r_high = exp_r_vals[drop_idx], exp_r_vals[drop_idx + 1]
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        if evaluate_r(r_mid, stage="境界探索 R_min_P")[1] >= p_surv_max - 1e-4:
          r_low = r_mid
        else:
          r_high = r_mid
      r_min_p = r_low
    else:
      r_min_p = exp_r_vals[0] if p_surv_vals[
          0] < p_surv_max - 1e-4 else exp_r_vals[-1]

    # R_min_A
    def is_free(r: float, stage: str = "境界探索") -> bool:
      res = evaluate_r(r, stage=stage)
      return res[3] <= 0.01 and res[4] >= 0.99

    is_free_vals = [is_free(r, stage="境界探索 R_min_A") for r in exp_r_vals]
    a_drop_idx = -1
    for i in range(len(is_free_vals) - 1):
      if is_free_vals[i] and not is_free_vals[i + 1]:
        a_drop_idx = i
        break
    if a_drop_idx != -1:
      r_low, r_high = exp_r_vals[a_drop_idx], exp_r_vals[a_drop_idx + 1]
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        if is_free(r_mid, stage="境界探索 R_min_A"):
          r_low = r_mid
        else:
          r_high = r_mid
      r_min_a = r_low
    else:
      r_min_a = exp_r_vals[0] if not is_free_vals[0] else exp_r_vals[-1]

    # R_max_P
    hit_min_idx = -1
    for i in range(len(p_surv_vals) - 1):
      if p_surv_vals[i] > p_surv_min + 1e-4 and p_surv_vals[
          i + 1] <= p_surv_min + 1e-4:
        hit_min_idx = i
        break
    if hit_min_idx != -1:
      r_low, r_high = exp_r_vals[hit_min_idx], exp_r_vals[hit_min_idx + 1]
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        if evaluate_r(r_mid, stage="境界探索 R_max_P")[1] <= p_surv_min + 1e-4:
          r_high = r_mid
        else:
          r_low = r_mid
      r_max_p = r_high
    else:
      r_max_p = exp_r_vals[-1] if p_surv_vals[
          -1] > p_surv_min + 1e-4 else exp_r_vals[0]

    # R_max_A
    a_hit_idx = -1
    is_free_vals_max_a = [is_free(r, stage="境界探索 R_max_A") for r in exp_r_vals]
    for i in range(a_drop_idx + 1, len(is_free_vals_max_a) - 1):
      if not is_free_vals_max_a[i] and is_free_vals_max_a[i + 1]:
        a_hit_idx = i
        break
    if a_hit_idx != -1:
      r_low, r_high = exp_r_vals[a_hit_idx], exp_r_vals[a_hit_idx + 1]
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        if is_free(r_mid, stage="境界探索 R_max_A"):
          r_high = r_mid
        else:
          r_low = r_mid
      r_max_a = r_high
    else:
      r_max_a = exp_r_vals[-1]

    if r_min_p > r_max_p:
      r_max_p = r_min_p
    if r_min_a > r_max_a:
      r_max_a = r_min_a

    r_min_sampling = min(r_min_p, r_min_a)
    r_max_sampling = max(r_max_p, r_max_a)

    # 3. 遷移領域のサンプリング
    num_steps = 15
    if r_max_sampling > r_min_sampling:
      step_r_vals = np.geomspace(r_min_sampling, r_max_sampling, num_steps)
      for r in step_r_vals:
        # A_opt が安定している領域ではサンプリングを高速化
        a_fixed = None
        reason = ""
        # goal_based 系の手法または disable_shortcuts が有効な場合はショートカットを行わない
        if args.tie_breaker_method == "legacy" and not args.disable_shortcuts:
          if r < r_min_a:
            a_fixed = 1.0
            reason = f"R={r:.4f} < R_min_a={r_min_a:.4f} なので A=1.0 に固定"
          elif r > r_max_a:
            # 境界での a_max を参考にする
            res_boundary = evaluate_r(r_max_a, stage="遷移領域サンプリング（境界値確認）")
            a_fixed = res_boundary[4]
            reason = f"R={r:.4f} > R_max_a={r_max_a:.4f} なので A={a_fixed:.2f} に固定"
        evaluate_r(r, a_fixed=a_fixed, stage="遷移領域サンプリング", reason=reason)
      for i in range(len(step_r_vals) - 1):
        # goal_based 系の手法または disable_shortcuts の場合は a_fixed による高速化を行わない
        use_shortcuts = args.tie_breaker_method == "legacy" and not args.disable_shortcuts
        r_min_a_val = r_min_a if use_shortcuts else None
        r_max_a_val = r_max_a if use_shortcuts else None
        adaptive_sample(evaluate_r,
                        step_r_vals[i],
                        step_r_vals[i + 1],
                        r_min_a=r_min_a_val,
                        r_max_a=r_max_a_val)

    # 評価結果の集約
    age_results = []
    for r, (a, p, survivals, a_min, a_max) in eval_cache.items():
      row_data = {
          "r": r,
          "a_opt": a,
          "p_survival": p,
          "a_opt_min": a_min,
          "a_opt_max": a_max
      }
      for a_val, p_val in survivals.items():
        row_data[f"{a_val:.2f}"] = p_val
      age_results.append(row_data)
    df_age = pd.DataFrame(age_results).sort_values("r")

    # フィッティング用データ抽出
    df_fit_p = df_age[(df_age["r"] >= r_min_p - 1e-9) &
                      (df_age["r"] <= r_max_p + 1e-9)].copy()
    df_fit_a = df_age[(df_age["r"] >= r_min_a - 1e-9) &
                      (df_age["r"] <= r_max_a + 1e-9)].copy()

    # P_surv モデル: Isotonic + PCHIP + Anchor Filtering
    iso_reg = IsotonicRegression(y_min=p_surv_min,
                                 y_max=p_surv_max,
                                 increasing=False,
                                 out_of_bounds='clip')
    p_iso = iso_reg.fit_transform(df_fit_p["r"], df_fit_p["p_survival"])
    unique_r_p, unique_idx_p = np.unique(df_fit_p["r"], return_index=True)
    p_iso_unique = p_iso[unique_idx_p]
    # Anchor point 削減 (1% threshold)
    r_points_p, p_points = filter_anchors(unique_r_p,
                                          p_iso_unique,
                                          threshold=0.01)

    # A_opt モデル: PCHIP on a_opt (legacy の場合は a_max と等価) + Anchor Filtering
    unique_r_a, unique_idx_a = np.unique(df_fit_a["r"], return_index=True)
    if args.tie_breaker_method == "legacy":
      a_fit_targets = df_fit_a["a_opt_max"].values[unique_idx_a]
    elif args.tie_breaker_method == "survival_first_goal_based":
      # 生存確率がほぼ 100% なら a_opt (goal_based), そうでなければ a_opt_max (legacy 互換の 0.1% 妥協ルール)
      is_100 = df_fit_a["p_survival"] >= 0.999 - 1e-9
      a_targets_raw = np.where(is_100, df_fit_a["a_opt"], df_fit_a["a_opt_max"])
      a_fit_targets = a_targets_raw[unique_idx_a]
    else:
      a_fit_targets = df_fit_a["a_opt"].values[unique_idx_a]

    # Anchor point 削減 (0% threshold to disable reduction)
    r_points_a, a_points = filter_anchors(unique_r_a,
                                          a_fit_targets,
                                          threshold=0.0)

    if args.debug_age is not None and age == args.debug_age:
      print(
          f"\n[DEBUG Age {age}] Anchor counts: P={len(r_points_p)}, A={len(r_points_a)}"
      )
      print("index, R, P_obs, P_fit, A_opt, A_max, A_fit")
      for i, (idx, age_row) in enumerate(df_age.iterrows()):
        rv = float(age_row["r"])
        p_fit = p_surv_max if rv < r_min_p else (
            p_surv_min if rv > r_max_p else pchip_interpolate(
                r_points_p, p_points, rv))
        a_fit = 1.0 if (rv < r_min_a or rv > r_max_a) else pchip_interpolate(
            r_points_a, a_points, rv)
        print(
            f"{i}, {rv:.6f}, {age_row['p_survival']:.6f}, {p_fit:.6f}, {age_row['a_opt']:.2f}, {age_row['a_opt_max']:.2f}, {a_fit:.2f}"
        )

    # 結果表示
    print(
        f"  R range: {df_age['r'].min():.4f} to {df_age['r'].max():.4f} (Total {len(df_age)} points)"
    )
    print(f"  P_surv range: P_min={p_surv_min:.4f}, P_max={p_surv_max:.4f}")
    print(
        f"  Detected Boundaries: R_min_P={r_min_p:.4f}, R_min_A={r_min_a:.4f}, R_max_P={r_max_p:.4f}, R_max_A={r_max_a:.4f}"
    )
    print(f"  A_opt model: PCHIP Spline ({len(r_points_a)} anchors)")
    print(f"  P_surv model: PCHIP Spline ({len(r_points_p)} anchors)")

    # 詳細データを temp/ に保存
    os.makedirs("temp", exist_ok=True)
    dump_data = {
        "age": int(age),
        "config": {
            "r_min_p": float(r_min_p),
            "r_min_a": float(r_min_a),
            "r_max_p": float(r_max_p),
            "r_max_a": float(r_max_a),
            "p_min": float(p_surv_min),
            "p_max": float(p_surv_max)
        },
        "models": {
            "p_survival": {
                "r_points": [float(r) for r in r_points_p],
                "p_points": [float(p) for p in p_points]
            },
            "a_optimal": {
                "r_points": [float(r) for r in r_points_a],
                "a_points": [float(a) for a in a_points]
            }
        },
        "all_points": df_age.to_dict(orient="records"),
        "training_points_p": df_fit_p.to_dict(orient="records"),
        "training_points_a": df_fit_a.to_dict(orient="records"),
        "search_logs": search_logs
    }
    with open(f"temp/age_{age}.json", "w", encoding="utf-8") as f:
      json.dump(dump_data, f, indent=2, ensure_ascii=False)

    # 結果保存
    dp_results[age] = {
        "y_withdraw": y_withdraw_n,
        "p_model": {
            "r_points": r_points_p,
            "p_points": p_points
        },
        "r_min_p": r_min_p,
        "r_min_a": r_min_a,
        "r_max_p": r_max_p,
        "r_max_a": r_max_a,
        "p_min": p_surv_min,
        "p_max": p_surv_max
    }
    models[str(age)] = {
        "avg_y_withdraw": avg_y_withdraw_n,
        "m_winning_multiplier": m_winning_multiplier,
        "m_winning_multiplier_v2": m_v2_percentiles,
        "a_opt_model": {
            "r_points": [float(r) for r in r_points_a],
            "a_points": [float(a) for a in a_points],
            "r_min_a": float(r_min_a),
            "r_max_a": float(r_max_a)
        },
        "p_survival_model": {
            "r_points": [float(r) for r in r_points_p],
            "p_points": [float(p) for p in p_points],
            "r_min_p": float(r_min_p),
            "r_max_p": float(r_max_p)
        },
        "p_min": float(p_surv_min),
        "p_max": float(p_surv_max)
    }
    if age in ar1_models:
      models[str(age)].update(ar1_models[age])

  if args.debug_level == 0:
    if args.output_path:
      output_path = args.output_path
    else:
      binary_name = os.path.splitext(os.path.basename(__file__))[0]
      if binary_name.endswith("_main"):
        binary_name = binary_name[:-5]
      output_dir = os.path.join("data", binary_name)
      output_path = os.path.join(output_dir, f"{args.scenario}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(models, f, indent=2)
    print(f"\nSuccessfully exported models to {output_path}")
  else:
    print("\nDebug mode: models not exported.")


if __name__ == "__main__":
  main()
