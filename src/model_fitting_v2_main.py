"""
Optimal Strategy V2 のモデルフィッティングを行うスクリプト。

このスクリプトは、後ろ向き動的計画法（Backward DP）を用いて、各年齢（35歳から95歳）における
最適な資産配分（オルカン比率）と、その時の生存確率を計算し、回帰モデルとして保存します。

状態変数として「年間支出率 R」を採用しています：
  R = 年間の純支出合計 / 年始の総資産
ここで、純支出合計は（支出 - 年金受取 + 年金保険料）の月次合計のうち、正の値を合算したものです。

アルゴリズムの概要：
1. 最終年齢（95歳）から開始し、35歳まで1年ずつ遡ります。
2. 各年齢において、R のグリッド（0.005から20以上まで）を作成します。
3. 各 R に対して、オルカン比率 A（0.0から1.0）を変化させて1年間のシミュレーションを実行します。
4. 翌年の生存確率モデルを用いて、期待生存確率を最大化する最適な A (A_opt) を見つけます。
5. R と A_opt、および R と生存確率の関係を多項式回帰モデルでフィッティングします。
6. 結果を `data/optimal_strategy_v2_models.json` に保存します。

実行方法:
  python src/model_fitting_v2_main.py --n_sim 2000
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.core import (CashflowRule, CashflowType, Strategy, ZeroRiskAsset,
                      simulate_strategy)
from src.lib.asset_generator import (AssetConfigType, DerivedAsset, ForexAsset,
                                     SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, PensionConfig,
                                        generate_cashflows)
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)

# 共通定数
START_AGE = 35
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


def logit(p: np.ndarray, p_max: float, p_min: float) -> np.ndarray:
  """
  P_surv_max, P_surv_min を考慮したロジット関数。
  p を [p_min, p_max] から [0, 1] の範囲に正規化してからロジット変換を行う。

  Args:
    p: 生存確率の配列
    p_max: 生存確率の最大値
    p_min: 生存確率の最小値

  Returns:
    ロジット変換後の値
  """
  # ゼロ割や対数(0)を防ぐため、ごくわずかなマージンを持たせる
  margin = 1e-7
  if p_max - p_min < 2 * margin:
    # minとmaxがほぼ同じ場合は、全て0.5とみなす（モデルとして意味をなさないがエラー回避のため）
    normalized_p = np.full_like(p, 0.5)
  else:
    normalized_p = (p - p_min) / (p_max - p_min)

  p_clipped = np.clip(normalized_p, margin, 1 - margin)
  return np.log(p_clipped / (1 - p_clipped))


def sigmoid_inv(x: Union[float, np.ndarray], p_max: float,
                p_min: float) -> Union[float, np.ndarray]:
  """
  正規化されたシグモイド値から元の確率空間に戻す逆関数。

  Args:
    x: シグモイド関数への入力値
    p_max: 生存確率の最大値
    p_min: 生存確率の最小値

  Returns:
    元の確率空間における値
  """
  # x が極端に大きい/小さい場合のオーバーフローを防ぐ
  s = 1 / (1 + np.exp(-np.clip(x, -100, 100)))
  return p_min + s * (p_max - p_min)


def get_base_features(r: np.ndarray) -> np.ndarray:
  """
  R から基底特徴量を作成する: R, 1/R, log(R)
  exp(R) を含めると PolynomialFeatures で R が大きい時に overflow するため除外。

  Args:
    r: 支出率の配列

  Returns:
    特徴量行列
  """
  r_safe = np.maximum(r, 1e-5)
  return np.column_stack([r_safe, 1.0 / r_safe, np.log(r_safe)])


def fit_model(
    x: np.ndarray,
    y: np.ndarray) -> Tuple[List[float], float, float, float, List[str]]:
  """
  モデルのフィッティングを行い、係数、切片、R2、調整済みR2、特徴量名を返す。

  Args:
    x: 入力データ (R)
    y: ターゲットデータ

  Returns:
    (係数リスト, 切片, R2, 調整済みR2, 特徴量名)
  """
  if len(y) == 0:
    return [], 0.0, 0.0, 0.0, []

  if np.allclose(y, y[0], atol=1e-9):
    # 分散がない場合は定数モデルとして扱う
    # PolynomialFeatures(degree=3) で R, 1/R, log(R) (3つ) -> 20特徴量
    return [0.0] * 20, float(y[0]), 1.0, 1.0, []

  base_feats = get_base_features(x)
  # R, 1/R, log(R) の degree=3 多項式組み合わせ
  poly = PolynomialFeatures(degree=3, include_bias=True)
  features = poly.fit_transform(base_feats)

  base_names = ["R", "invR", "logR"]
  feature_names = poly.get_feature_names_out(base_names).tolist()

  model = LinearRegression(fit_intercept=False)
  model.fit(features, y)

  r2 = float(model.score(features, y))
  n = len(y)
  p = features.shape[1] - 1
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

  return model.coef_.tolist(), 0.0, r2, adj_r2, feature_names


def main():
  # 引数の処理
  parser = argparse.ArgumentParser(
      description="Optimal Strategy V2 のモデルフィッティング")
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
  args = parser.parse_args()

  n_sim = args.n_sim
  debug = args.debug_level > 0
  debug_paths = [int(p) for p in args.debug_paths.split(",")] if args.debug_paths else []

  # 1. アセットとキャッシュフローの生成
  # 為替 (USDJPY 0%, 10.53%)
  fx_asset = ForexAsset(name=FX_NAME,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  # オルカン (共通モデルから取得)
  base_sp500 = get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y)
  base_acwi = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)
  # 投資対象としてのオルカン (為替と信託報酬を適用)
  orukan = DerivedAsset(name=ORUKAN_NAME,
                        base=base_acwi.name,
                        trust_fee=TRUST_FEE,
                        forex=FX_NAME)
  # ゼロリスク資産 (利回り 4%)
  zr_asset_obj = ZeroRiskAsset(name=ZERO_RISK_NAME, yield_rate=ZERO_RISK_YIELD)
  # CPI (共通モデル)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)
  # 年金用CPI (マクロ経済スライド 0.5% 抑制)
  pension_cpi = SlideAdjustedCpiAsset(
      name=PENSION_CPI_NAME,
      base_cpi=CPI_NAME,
      slide_rate=0.005,
      slide_end_month=(MACRO_ECONOMIC_SLIDE_END_YEAR - CURRENT_YEAR) * 12)

  asset_configs: List[AssetConfigType] = [
      fx_asset, base_sp500, base_acwi, orukan, base_cpi, pension_cpi
  ]

  print(f"Generating asset prices for {YEARS} years, {n_sim} paths...")
  monthly_prices = generate_monthly_asset_prices(asset_configs,
                                                 n_paths=n_sim,
                                                 n_months=YEARS * 12,
                                                 seed=SEED)

  # キャッシュフロー設定 (一人世帯 H1, 年金60歳受給 P60相当)
  cf_configs: List[CashflowConfig] = []
  cf_rules: List[CashflowRule] = []

  # 年金保険料 (35歳から60歳まで)
  cf_configs.append(
      PensionConfig(name="Pension_Premium",
                    amount=-20.4 / 12.0,
                    start_month=0,
                    end_month=(60 - START_AGE) * 12,
                    cpi_name=CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Premium",
                   cashflow_type=CashflowType.REGULAR))

  # 年金受給 (60歳から)
  # 基礎年金 81.6 * 0.76 (繰上げ), 厚生年金 76.6 * 0.76 (繰上げ)
  reduction_rate = 0.76
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kousei",
                    amount=(76.6 * reduction_rate) / 12.0,
                    start_month=(60 - START_AGE) * 12,
                    cpi_name=CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kousei",
                   cashflow_type=CashflowType.REGULAR))

  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kiso",
                    amount=(81.6 * reduction_rate) / 12.0,
                    start_month=(60 - START_AGE) * 12,
                    cpi_name=PENSION_CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kiso",
                   cashflow_type=CashflowType.REGULAR))

  # 年齢による月額支出（名目ベースライン、円）の取得 (35歳から60年間)
  # normalize=False を指定して、実際の統計値ベースの月額（円）を取得する。
  spending_monthly_values = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=START_AGE,
      num_years=YEARS,
      normalize=False)

  print("Generating cashflows...")
  monthly_cashflows = generate_cashflows(cf_configs,
                                         monthly_prices,
                                         n_sim=n_sim,
                                         n_months=YEARS * 12)

  # 2. Backward DP
  models: Dict[str, Any] = {}
  # age -> { "y_withdraw": array, "p_model": {coef}, "r_min": float, "r_max": float, "p_min": float, "p_max": float }
  dp_results: Dict[int, Any] = {}

  # 年齢 95 から 35 まで逆算
  ages_to_process = range(END_AGE - 1, START_AGE - 1, -1)
  if args.debug_level > 0:
    ages_to_process = range(END_AGE - 1, END_AGE - 4, -1)  # デバッグ時は直近3年分のみ

  for age in ages_to_process:
    print(f"\n--- Processing age {age} ---")

    # この年のキャッシュフロー (12ヶ月分) のインデックス
    year_idx = age - START_AGE
    start_m = year_idx * 12
    end_m = (year_idx + 1) * 12
    cpi_path = monthly_prices[CPI_NAME][:, start_m:end_m]

    monthly_net_spend = np.zeros((n_sim, 12))
    # 基本支出 (万円/月) = 統計値月額(円) / 10000 * CPI倍率
    monthly_spend_base = spending_monthly_values[year_idx] / 10000.0
    monthly_net_spend += monthly_spend_base * cpi_path

    # 年金等
    p_premium = monthly_cashflows["Pension_Premium"][:, start_m:end_m]
    p_kousei = monthly_cashflows["Pension_Receipt_Kousei"][:, start_m:end_m]
    p_kiso = monthly_cashflows["Pension_Receipt_Kiso"][:, start_m:end_m]
    pension_total = p_premium + p_kousei + p_kiso

    monthly_net_spend -= pension_total

    # 各パスの年間合計正味支出 (Withdrawal amount)
    y_withdraw_n = np.sum(np.maximum(0, monthly_net_spend),
                          axis=1)  # shape (n_sim,)

    if args.debug_level >= 2:
      print(f"  [Level 2 Info] Cashflow:")
      print(
          f"    Avg Base Spend (Month 0): {monthly_spend_base * np.mean(cpi_path[:,0]):.2f} 万円/月"
      )
      print(
          f"    Avg Pension (Month 0): {np.mean(pension_total[:,0]):.2f} 万円/月")
      print(f"    Avg Y_withdraw_n (Yearly): {np.mean(y_withdraw_n):.2f} 万円/年")

    # R (支出率) と A (オルカン比率) のグリッド
    a_grid = np.linspace(0.0, 1.0, 11)  # 0.1刻み

    # キャッシュ済み evaluate_r の結果
    eval_cache: Dict[float, Tuple[float, float]] = {}

    def evaluate_r(r: float) -> Tuple[float, float]:
      # キャッシュにあればそれを返す (浮動小数点の誤差を考慮して丸める)
      r_key = round(r, 6)
      if r_key in eval_cache:
        return eval_cache[r_key]

      # 初期資産 X_p,N = Y_withdraw,p,N / r
      x_p_n = y_withdraw_n / r
      best_survival = -1.0
      best_a = 0.0
      for a in a_grid:
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
            annual_cost=0.0,  # すべて monthly_cashflows で制御
            inflation_rate=None,
            selling_priority=[ORUKAN_NAME, ZERO_RISK_NAME],
            tax_rate=TAX_RATE,
            rebalance_interval=0  # 1年なのでリバランスなし
        )
        year_prices = {
            k: v[:, start_m:end_m + 1]
            for k, v in monthly_prices.items()
            if k != ZERO_RISK_NAME
        }

        year_cf = {"Net_Spend": -monthly_net_spend}  # 支出は負で渡す
        strategy.cashflow_rules = [
            CashflowRule(source_name="Net_Spend",
                         cashflow_type=CashflowType.REGULAR)
        ]

        # シミュレーション実行
        res = simulate_strategy(strategy,
                                year_prices,
                                monthly_cashflows=year_cf,
                                fallback_total_months=12)
        x_next = res.net_values  # shape (n_sim,)

        # 今年の破産判定
        bankrupt_this_year = res.sustained_months < 12

        # 生存判定
        if age == END_AGE - 1:
          # 最終年 (95歳) は今年生存していれば P=1.0
          survival = (~bankrupt_this_year).astype(float)
        else:
          # 次年度の R = Y_withdraw_{N+1} / X_{N+1}
          next_y_withdraw = dp_results[age + 1]["y_withdraw"]
          r_next = next_y_withdraw / np.maximum(x_next, 1e-7)

          # 次年度の生存確率モデルを取得
          next_model = dp_results[age + 1]["p_model"]
          next_r_min = dp_results[age + 1]["r_min"]
          next_r_max = dp_results[age + 1]["r_max"]
          next_p_max = dp_results[age + 1].get("p_max", 1.0)
          next_p_min = dp_results[age + 1].get("p_min", 0.0)

          p_next = np.zeros(n_sim)

          # 今年生きていて、かつ来年の R が R_min 以下なら P_max
          p_next[(~bankrupt_this_year) & (r_next <= next_r_min)] = next_p_max
          # 今年生きていて、かつ来年の R が R_max 以上なら P_min
          p_next[(~bankrupt_this_year) & (r_next >= next_r_max)] = next_p_min

          # モデル適用範囲内
          in_range = (~bankrupt_this_year) & (r_next > next_r_min) & (
              r_next < next_r_max)
          if np.any(in_range):
            base_feats = get_base_features(r_next[in_range])
            poly = PolynomialFeatures(degree=3, include_bias=True)
            features = poly.fit_transform(base_feats)
            logit_p = features @ np.array(next_model["coef"])
            p_next[in_range] = sigmoid_inv(logit_p, next_p_max, next_p_min)

          survival = p_next

        # デバッグ情報の表示
        if age == args.debug_age and debug_paths:
          print(f"      [Path Debug] R={r:.4f}, A={a:.2f}")
          for p_idx in debug_paths:
            if p_idx < n_sim:
              print(f"        Path {p_idx}: X_next={x_next[p_idx]:.2f}, Y_next={next_y_withdraw[p_idx] if age < END_AGE-1 else 0:.2f}, R_next={r_next[p_idx] if age < END_AGE-1 else 0:.4f}, P_surv={survival[p_idx]:.4f}")

        # 全パスの平均生存確率
        avg_survival = float(np.mean(survival))
        if avg_survival > best_survival:
          best_survival = avg_survival
          best_a = a

      result = (float(best_a), float(best_survival))
      eval_cache[r_key] = result
      if args.debug_level >= 3:
        print(f"    Tested R={r:.4f}, P_surv={result[1]:.4f}")
      return result

    # 1. R 広域探索 (Exponential Search)
    # 0.005 から 2倍ずつ 13ステップ (最大 20.48)
    exp_r_vals = [0.005 * (2**k) for k in range(13)]
    exp_results = []

    for r in exp_r_vals:
      a_opt, p_surv = evaluate_r(r)
      exp_results.append((r, a_opt, p_surv))

    p_surv_vals = [res[2] for res in exp_results]
    p_surv_max = max(p_surv_vals)
    p_surv_min = min(p_surv_vals)

    # 2. 境界探索 (Binary Search)
    # P_surv_max から下がり始める区間を見つける
    drop_idx = -1
    for i in range(len(p_surv_vals) - 1):
      if p_surv_vals[i] >= p_surv_max - 1e-4 and p_surv_vals[
          i + 1] < p_surv_max - 1e-4:
        drop_idx = i
        break

    if drop_idx != -1:
      r_low = exp_r_vals[drop_idx]
      r_high = exp_r_vals[drop_idx + 1]
      # R_min の二分探索
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        _, p_surv = evaluate_r(r_mid)
        if p_surv >= p_surv_max - 1e-4:
          r_low = r_mid
        else:
          r_high = r_mid
        # 確率の差が十分小さければ終了
        if abs(p_surv - p_surv_max) < 0.0001:
          break
      r_min = r_low
    else:
      # P_surv_max から下がらないか、最初から下がっている
      if p_surv_vals[0] < p_surv_max - 1e-4:
        r_min = exp_r_vals[0]
      else:
        r_min = exp_r_vals[-1]

    # P_surv_min に到達する区間を見つける
    hit_min_idx = -1
    for i in range(len(p_surv_vals) - 1):
      if p_surv_vals[i] > p_surv_min + 1e-4 and p_surv_vals[
          i + 1] <= p_surv_min + 1e-4:
        hit_min_idx = i
        break

    if hit_min_idx != -1:
      r_low = exp_r_vals[hit_min_idx]
      r_high = exp_r_vals[hit_min_idx + 1]
      # R_max の二分探索
      for _ in range(10):
        r_mid = (r_low + r_high) / 2
        _, p_surv = evaluate_r(r_mid)
        if p_surv <= p_surv_min + 1e-4:
          r_high = r_mid
        else:
          r_low = r_mid
        # 確率の差が十分小さければ終了
        if abs(p_surv - p_surv_min) < 0.0001:
          break
      r_max = r_high
    else:
      # P_surv_min に到達しないか、最初から最小値
      if p_surv_vals[-1] > p_surv_min + 1e-4:
        r_max = exp_r_vals[-1]
      else:
        r_max = exp_r_vals[0]

    # 論理的順序の保証
    if r_min > r_max:
      r_max = r_min

    # 3. 遷移領域のステップサーチ
    num_steps = 30
    if r_max > r_min:
      # 対数スケールでサンプリングすることで、高確率（低R）領域の解像度を高める
      step_r_vals = np.geomspace(r_min, r_max, num_steps)
      for r in step_r_vals:
        evaluate_r(r)

    # 評価結果をまとめてフィッティング
    age_results = [{
        "r": r,
        "a_opt": a,
        "p_survival": p
    } for r, (a, p) in eval_cache.items()]
    df_age = pd.DataFrame(age_results).sort_values("r")

    # P_surv が (P_min, P_max) の間にあるデータを抽出
    mask_fit = (df_age["p_survival"] < p_surv_max -
                1e-4) & (df_age["p_survival"] > p_surv_min + 1e-4)

    if mask_fit.sum() < 20:
      mask_fit_loose = (df_age["p_survival"]
                        <= p_surv_max) & (df_age["p_survival"] >= p_surv_min)
      if mask_fit_loose.sum() < 20:
        x_fit, a_fit, p_fit = df_age["r"].values, df_age[
            "a_opt"].values, df_age["p_survival"].values
      else:
        x_fit, a_fit, p_fit = df_age[mask_fit_loose]["r"].values, df_age[
            mask_fit_loose]["a_opt"].values, df_age[mask_fit_loose][
                "p_survival"].values
    else:
      x_fit, a_fit, p_fit = df_age[mask_fit]["r"].values, df_age[mask_fit][
          "a_opt"].values, df_age[mask_fit]["p_survival"].values

    x_fit_arr = np.array(x_fit, dtype=np.float64)
    a_fit_arr = np.array(a_fit, dtype=np.float64)
    p_fit_arr = np.array(p_fit, dtype=np.float64)

    # モデル fitting
    a_coef, _, a_r2, a_adj_r2, feat_names = fit_model(x_fit_arr, a_fit_arr)
    logit_p_fit_arr = logit(p_fit_arr, p_surv_max, p_surv_min)
    p_coef, _, p_r2, p_adj_r2, _ = fit_model(x_fit_arr, logit_p_fit_arr)

    if args.debug_age is not None and age == args.debug_age:
      print(f"\n[DEBUG Age {age}] Transition Region Data (num_steps={num_steps}):")
      print(f"R_min: {r_min:.6f}, R_max: {r_max:.6f}")
      print(f"P_max: {p_surv_max:.6f}, P_min: {p_surv_min:.6f}")

      # ステップサーチ結果の表示
      step_r_vals = np.geomspace(r_min, r_max, num_steps)
      print("index, R, P_obs, P_fit")
      for i, r in enumerate(step_r_vals):
        a_opt, p_obs = evaluate_r(r)

        # フィッティングされた P を計算
        base_feats = get_base_features(np.array([r]))
        poly = PolynomialFeatures(degree=3, include_bias=True)
        features = poly.fit_transform(base_feats)
        logit_p_val = features @ np.array(p_coef)
        p_fit = sigmoid_inv(logit_p_val, p_surv_max, p_surv_min)[0]

        print(f"{i}, {r:.6f}, {p_obs:.6f}, {p_fit:.6f}")

      print(f"Included in x_fit_arr: {len(x_fit_arr)} points total.")

    # 結果表示
    print(
        f"  R range: {df_age['r'].min():.4f} to {df_age['r'].max():.4f} (Total {len(df_age)} points)"
    )
    print(f"  P_surv range: P_min={p_surv_min:.4f}, P_max={p_surv_max:.4f}")
    print(f"  Detected Boundaries: R_min={r_min:.4f}, R_max={r_max:.4f}")
    print(f"  A_opt model R2={a_r2:.4f}, AdjR2={a_adj_r2:.4f}")
    print(f"  P_surv model R2={p_r2:.4f}, AdjR2={p_adj_r2:.4f}")

    if debug:
      if args.debug_level >= 2:
        print(f"  [Level 2 Info] Model Fitting Details:")
        print(f"    Num Features: {len(feat_names)}")
        print(
            f"    A_opt model coefs: {dict(zip(feat_names, [round(c, 5) for c in a_coef])) if feat_names else 'Constant'}"
        )

    # 結果保存
    dp_results[age] = {
        "y_withdraw": y_withdraw_n,
        "p_model": {
            "coef": p_coef
        },
        "r_min": float(r_min),
        "r_max": float(r_max),
        "p_min": float(p_surv_min),
        "p_max": float(p_surv_max)
    }
    models[str(age)] = {
        "a_opt_model": {
            "coef": a_coef,
            "r2": a_r2,
            "adj_r2": a_adj_r2
        },
        "p_survival_model": {
            "coef": p_coef,
            "r2": p_r2,
            "adj_r2": p_adj_r2
        },
        "r_min": float(r_min),
        "r_max": float(r_max),
        "p_min": float(p_surv_min),
        "p_max": float(p_surv_max),
        "feature_names": feat_names
    }

  # 3. エクスポート
  if args.debug_level == 0:
    os.makedirs("data", exist_ok=True)
    output_path = "data/optimal_strategy_v2_models.json"
    with open(output_path, "w") as f:
      json.dump(models, f, indent=2)
    print(f"\nSuccessfully exported models to {output_path}")
  else:
    print("\nDebug mode: models not exported.")


if __name__ == "__main__":
  main()
