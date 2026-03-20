"""
analyze_simulation_models_main.py

このスクリプトは、リタイアメントシミュレーション用の各種アセットモデル（S&P500、ACWI）
について、具体的な統計パラメータ（平均、標準偏差、最適な確率分布のパラメータなど）を
計算・比較評価するためのメインスクリプトです。

# 目的
`research/simulation_models_plan.md` に記載されている各種シミュレーションモデル
（全期間、直近30年間、SP500を用いたACWIの線形近似など）の数値を算出し、
尤度（Log-Likelihood）や情報量規準（BIC）を用いてモデルの妥当性を比較・検証します。

# 内部計算のロジック
1. data/asset_daily_prices.csv から日次価格データを読み込み、月次リターン（単利・対数）に変換。
2. 指定された期間（1871-2025、1995-2024、2008-2025など）でデータをスライス。
3. 単純正規分布、対数正規分布、および最も適合する分布（Best Fit: Laplace, Genlogistic等）
   のパラメータを scipy.stats を用いて最尤推定（MLE）で算出。
4. モンデカルロシミュレーションを用いて、各分布における「年率の期待リターンとリスク（ボラティリティ）」
   を算出し、直感的に理解しやすいパーセンテージでの期待値に変換。
5. ACWIについては、「ACWIの直接モデリング」と「SP500をベースにした線形近似モデリング」を
   対数尤度（Log-Likelihood）を共通の土俵（Induced Marginal Distribution）にして直接比較・評価。

# 実行方法
$ python src/analyze_simulation_models_main.py
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.lib import asset_model

warnings.filterwarnings('ignore')


def main():
  print("Loading data...")
  df = pd.read_csv('data/asset_daily_prices.csv')
  df['Date'] = pd.to_datetime(df['Date'])

  # Process monthly returns
  print("Processing monthly returns...")
  monthly_returns = asset_model.process_returns(df, 'M')

  # Setup date ranges
  end_date = '2025-12-31'
  
  # SP500 Monthly Data
  sp500_simple = monthly_returns['SP500_simple'].dropna()
  sp500_log = monthly_returns['SP500_log'].dropna()
  
  sp500_simple_all = sp500_simple[sp500_simple.index <= end_date]
  sp500_log_all = sp500_log[sp500_log.index <= end_date]
  print(f"SP500 All-Time: {sp500_simple_all.index.min().strftime('%Y-%m')} to {sp500_simple_all.index.max().strftime('%Y-%m')}")

  start_30y = '1996-01-01'
  sp500_simple_30y = sp500_simple[(sp500_simple.index >= start_30y) & (sp500_simple.index <= end_date)]
  sp500_log_30y = sp500_log[(sp500_log.index >= start_30y) & (sp500_log.index <= end_date)]
  print(f"SP500 Recent 30y: {sp500_simple_30y.index.min().strftime('%Y-%m')} to {sp500_simple_30y.index.max().strftime('%Y-%m')}")

  print("\n--- SP500 Models ---")

  # SP500-1: Simple Normal (All)
  res_sp1 = asset_model.fit_normal_simple(sp500_simple_all)
  mu_y1, std_y1 = asset_model.simulate_annual_stats_simple(stats.norm, (res_sp1['mu'], res_sp1['std']))
  print(f"SP500-1 (Simple Normal, All): mu_M={res_sp1['mu']:.6f}, sigma_M={res_sp1['std']:.6f}, mu_Y={mu_y1:.6f}, sigma_Y={std_y1:.6f}, BIC={res_sp1['bic']:.2f}")

  # SP500-2: Log-Normal (All)
  res_sp2 = asset_model.fit_normal_log(sp500_log_all)
  mu_y2, std_y2 = asset_model.simulate_annual_stats_log(stats.norm, (res_sp2['mu'], res_sp2['std']))
  print(f"SP500-2 (Log-Normal, All): mu_M={res_sp2['mu']:.6f}, sigma_M={res_sp2['std']:.6f}, mu_Y={mu_y2:.6f}, sigma_Y={std_y2:.6f}, BIC={res_sp2['bic']:.2f}")

  # SP500-3: Best Fit (All)
  res_sp3 = asset_model.find_best_distribution(sp500_log_all, top_n=1)[0]
  dist3 = getattr(stats, res_sp3['name'])
  mu_y3, std_y3 = asset_model.simulate_annual_stats_log(dist3, res_sp3['params'])
  print(f"SP500-3 (Best Fit, All): dist={res_sp3['name']}, params={res_sp3['params']}, mu_Y={mu_y3:.6f}, sigma_Y={std_y3:.6f}, BIC={res_sp3['bic']:.2f}")

  # SP500-4: MR-GBM (All, Monthly)
  # For MR-GBM we need prices, not returns. So we resample prices.
  df_monthly = df.set_index('Date').resample('ME').last()
  sp500_prices = df_monthly['SP500'].dropna()
  sp500_prices_all = sp500_prices[sp500_prices.index <= end_date]
  dt_monthly = 1.0 / 12.0
  
  # Calculate MR-GBM without strict negative b check to see what it is
  x = np.log(sp500_prices_all.values)
  x_t = x[:-1]
  dx = x[1:] - x_t
  A = np.vstack([x_t, np.ones(len(x_t))]).T
  b, a = np.linalg.lstsq(A, dx, rcond=None)[0]
  if b < 0:
    theta = -np.log(b + 1) / dt_monthly
    mu = a / (-b)
    res = dx - (a + b * x_t)
    var_res = np.var(res)
    sigma = np.sqrt(var_res * 2 * theta / (1 - np.exp(-2 * theta * dt_monthly)))
    print(f"SP500-4 (MR-GBM, Monthly): theta={theta:.6f}, mu={mu:.6f}, sigma={sigma:.6f}")
  else:
    print(f"SP500-4 (MR-GBM, Monthly): Failed to fit, positive slope b={b:.6f}")

  # SP500-5: Log-Normal (30y)
  res_sp5 = asset_model.fit_normal_log(sp500_log_30y)
  mu_y5, std_y5 = asset_model.simulate_annual_stats_log(stats.norm, (res_sp5['mu'], res_sp5['std']))
  print(f"SP500-5 (Log-Normal, 30y): mu_M={res_sp5['mu']:.6f}, sigma_M={res_sp5['std']:.6f}, mu_Y={mu_y5:.6f}, sigma_Y={std_y5:.6f}, BIC={res_sp5['bic']:.2f}")

  # SP500-6: Best Fit (30y)
  res_sp6 = asset_model.find_best_distribution(sp500_log_30y, top_n=1)[0]
  dist6 = getattr(stats, res_sp6['name'])
  mu_y6, std_y6 = asset_model.simulate_annual_stats_log(dist6, res_sp6['params'])
  print(f"SP500-6 (Best Fit, 30y): dist={res_sp6['name']}, params={res_sp6['params']}, mu_Y={mu_y6:.6f}, sigma_Y={std_y6:.6f}, BIC={res_sp6['bic']:.2f}")

  print("\n--- ACWI Models ---")

  acwi_simple = monthly_returns['ACWI_simple'].dropna()
  acwi_log = monthly_returns['ACWI_log'].dropna()

  acwi_simple_all = acwi_simple[acwi_simple.index <= end_date]
  acwi_log_all = acwi_log[acwi_log.index <= end_date]
  print(f"ACWI All-Time: {acwi_simple_all.index.min().strftime('%Y-%m')} to {acwi_simple_all.index.max().strftime('%Y-%m')}")

  # ACWI-1: Simple Normal (All)
  res_ac1 = asset_model.fit_normal_simple(acwi_simple_all)
  mu_ya1, std_ya1 = asset_model.simulate_annual_stats_simple(stats.norm, (res_ac1['mu'], res_ac1['std']))
  print(f"ACWI-1 (Simple Normal, All): mu_M={res_ac1['mu']:.6f}, sigma_M={res_ac1['std']:.6f}, mu_Y={mu_ya1:.6f}, sigma_Y={std_ya1:.6f}, BIC={res_ac1['bic']:.2f}")

  # ACWI-2: Log-Normal (All)
  res_ac2 = asset_model.fit_normal_log(acwi_log_all)
  mu_ya2, std_ya2 = asset_model.simulate_annual_stats_log(stats.norm, (res_ac2['mu'], res_ac2['std']))
  print(f"ACWI-2 (Log-Normal, All): mu_M={res_ac2['mu']:.6f}, sigma_M={res_ac2['std']:.6f}, mu_Y={mu_ya2:.6f}, sigma_Y={std_ya2:.6f}, BIC={res_ac2['bic']:.2f}")

  # ACWI-3: Best Fit (All)
  res_ac3 = asset_model.find_best_distribution(acwi_log_all, top_n=1)[0]
  dista3 = getattr(stats, res_ac3['name'])
  mu_ya3, std_ya3 = asset_model.simulate_annual_stats_log(dista3, res_ac3['params'])
  print(f"ACWI-3 (Best Fit, All): dist={res_ac3['name']}, params={res_ac3['params']}, mu_Y={mu_ya3:.6f}, sigma_Y={std_ya3:.6f}, BIC={res_ac3['bic']:.2f}")

  print("\n--- ACWI Approximations ---")

  overlap = monthly_returns[['ACWI_log', 'SP500_log']].dropna()
  overlap = overlap[overlap.index <= end_date]
  slope, intercept, r_value, p_value, std_err = stats.linregress(overlap['SP500_log'], overlap['ACWI_log'])
  print(f"Linear Fit: ACWI = {slope:.4f} * SP500 {intercept:+.6f} (R^2 = {r_value**2:.4f})")

  residuals = overlap['ACWI_log'] - (slope * overlap['SP500_log'] + intercept)

  # ACWI-APP-1: Simple Normal fit to residuals
  res_app1 = asset_model.fit_normal_simple(residuals)
  # Annual stats for APP models are complex because it depends on SP500 too, 
  # so we won't print mu_Y here directly, but we will print parameters.
  print(f"ACWI-APP-1 (Normal Noise): mu_M={res_app1['mu']:.6f}, sigma_M={res_app1['std']:.6f}, BIC={res_app1['bic']:.2f}")

  # ACWI-APP-2: Best Fit to residuals
  res_app2 = asset_model.find_best_distribution(residuals, top_n=1)[0]
  print(f"ACWI-APP-2 (Best Fit Noise): dist={res_app2['name']}, params={res_app2['params']}, BIC={res_app2['bic']:.2f}")

  print("\n--- Unified Evaluation: Marginal Log-Likelihood on ACWI All-Time data ---")
  print(f"1. ACWI-2 (Direct Log-Normal): LL = {res_ac2['loglik']:.2f}")
  print(f"2. ACWI-3 (Direct Best Fit, {res_ac3['name']}): LL = {res_ac3['loglik']:.2f}")

  # Calculate Induced Marginal from ACWI-APP-1 + SP500-2
  induced_mu = slope * res_sp2['mu'] + intercept + res_app1['mu']
  induced_std = np.sqrt((slope**2) * (res_sp2['std']**2) + (res_app1['std']**2))
  ll_induced = stats.norm.logpdf(acwi_log_all, loc=induced_mu, scale=induced_std).sum()
  print(f"3. ACWI-APP-1 + SP500-2 (Induced Marginal Normal): LL = {ll_induced:.2f}")

  # Calculate Induced Marginal from ACWI-APP-1 + SP500-5
  induced_mu_30y = slope * res_sp5['mu'] + intercept + res_app1['mu']
  induced_std_30y = np.sqrt((slope**2) * (res_sp5['std']**2) + (res_app1['std']**2))
  ll_induced_30y = stats.norm.logpdf(acwi_log_all, loc=induced_mu_30y, scale=induced_std_30y).sum()
  print(f"4. ACWI-APP-1 + SP500-5 (Induced Marginal Normal): LL = {ll_induced_30y:.2f}")

  print(f"(Note: A higher Log-Likelihood (closer to 0) means the model is more likely to generate the actual ACWI data.)")



if __name__ == '__main__':
  main()
