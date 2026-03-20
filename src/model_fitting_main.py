"""
model_fitting_main.py

このスクリプトは、各種アセット（S&P 500、ACWI、BTC）のリターンデータに対して、
様々な確率分布モデル（正規分布、対数正規分布、ベストフィット分布など）や、
平均回帰型幾何ブラウン運動（MR-GBM）モデルを当てはめ、フィッティングの精度を評価するツールです。

# 目的
金融市場の価格変動が、標準的な幾何ブラウン運動（対数正規分布）で十分説明可能か、
あるいは複雑なファットテール分布（Laplace, Genlogisticなど）や平均回帰モデル（MR-GBM）、
周期性（BTCの4年サイクルなど）を導入する必要があるかを定量的に分析します。

# 内部計算のロジック
1. `data/asset_daily_prices.csv` の日次データを読み込み、日次および月次の単利・対数リターンを計算。
2. それぞれのアセット・期間に対して以下を適用:
   - Model A: 単利リターンを正規分布にフィット
   - Model B: 対数リターンを正規分布にフィット
   - Model C: SciPyの多数の連続分布（t分布、ラプラス、ガンマ等）を当てはめ、MSE（ヒストグラムとの誤差）が最小のものを探索
3. S&P 500とACWIの相関分析（日次および月次での線形回帰）と、その残差の分布推定。
4. 平均回帰性（MR-GBM）のパラメータ（theta, mu, sigma）を自己回帰（AR(1)）モデルを用いて推定。
5. BTC特有の4年サイクル（半減期）の季節性を除去した上での残差分析とMR-GBMのフィッティング。
6. 各モデルの適合度をAIC/BIC/MSEを用いて数値化し、端末に出力します。

# 実行方法
$ python src/model_fitting_main.py
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.lib import asset_model

warnings.filterwarnings('ignore')


def analyze():
  df = pd.read_csv('data/asset_daily_prices.csv')

  daily_returns = asset_model.process_returns(df, 'D')
  monthly_returns = asset_model.process_returns(df, 'M')

  print("=== Model Fitting Results ===")
  for asset in ['SP500', 'ACWI', 'BTC']:
    print(f"\n--- {asset} ---")
    for freq, rets in [('Daily', daily_returns), ('Monthly', monthly_returns)]:
      print(f"[{freq}]")
      simple = rets[f'{asset}_simple']
      log = rets[f'{asset}_log']

      if simple.dropna().empty:
        print("No data")
        continue

      res_a = asset_model.fit_normal_simple(simple)
      res_b = asset_model.fit_normal_log(log)

      print(
          f"Model A (Simple Normal): mu={res_a['mu']:.6f}, std={res_a['std']:.6f}, BIC={res_a['bic']:.2f}, MSE={res_a['mse']:.6f}"
      )
      print(
          f"Model B (Log Normal):    mu={res_b['mu']:.6f}, std={res_b['std']:.6f}, BIC={res_b['bic']:.2f}, MSE={res_b['mse']:.6f}"
      )

      # For C, run best distribution fit
      # Since it might be slow, print it with caution
      res_c_list = asset_model.find_best_distribution(log, top_n=1)
      if res_c_list:
        res_c = res_c_list[0]
        print(
            f"Model C (Best Dist MSE): dist={res_c['name']}, params={res_c['params']}, BIC={res_c['bic']:.2f}, MSE={res_c['mse']:.6f}"
        )
      else:
        print("Model C: Failed to fit")

  print("\n=== S&P 500 Local Fit (ACWI Overlap Period) ===")
  overlap_dates = daily_returns[['ACWI_log', 'SP500_log']].dropna().index
  local_sp500_log = daily_returns.loc[overlap_dates, 'SP500_log']
  local_sp500_simple = daily_returns.loc[overlap_dates, 'SP500_simple']

  res_a_local = asset_model.fit_normal_simple(local_sp500_simple)
  res_b_local = asset_model.fit_normal_log(local_sp500_log)
  print(f"Period: {overlap_dates[0].date()} to {overlap_dates[-1].date()}")
  print(f"Model A (Simple): mu={res_a_local['mu']:.6f}, std={res_a_local['std']:.6f}, BIC={res_a_local['bic']:.2f}, MSE={res_a_local['mse']:.6f}")
  print(f"Model B (Log):    mu={res_b_local['mu']:.6f}, std={res_b_local['std']:.6f}, BIC={res_b_local['bic']:.2f}, MSE={res_b_local['mse']:.6f}")
  res_c_local_list = asset_model.find_best_distribution(local_sp500_log, top_n=1)
  if res_c_local_list:
    res_c_local = res_c_local_list[0]
    print(f"Model C (Best):  dist={res_c_local['name']}, params={res_c_local['params']}, BIC={res_c_local['bic']:.2f}, MSE={res_c_local['mse']:.6f}")

  print("\n=== S&P500 1871-2024 Verify Claim ===")
  df['Date'] = pd.to_datetime(df['Date'])
  sp500_data = df.set_index('Date')['SP500']
  if isinstance(sp500_data, pd.Series):
    sp500_data = sp500_data.dropna()
    sp500_data = sp500_data[(sp500_data.index >= '1871-01-01') &
                            (sp500_data.index <= '2024-12-31')]
    # Annual return
    years = (sp500_data.index[-1] - sp500_data.index[0]).days / 365.25
    cagr = (sp500_data.iloc[-1] / sp500_data.iloc[0])**(1 / years) - 1

    # Annual risk (std of log returns) * sqrt(12) for monthly data
    monthly_sp500 = sp500_data.resample('ME').last()
    _sp500_ret = monthly_sp500 / monthly_sp500.shift(1)
    monthly_log_returns = pd.Series(np.log(np.array(_sp500_ret, dtype=float)),
                                    index=_sp500_ret.index).dropna()
    annual_risk = monthly_log_returns.std() * np.sqrt(12)
    print(f"Expected Return (CAGR): {cagr*100:.2f}% (Claim: 9.9%)")
    print(f"Risk (Annualized Std): {annual_risk*100:.2f}% (Claim: 14.0%)")

  print("\n=== Correlation ACWI vs S&P500 (Daily) ===")
  overlap_daily = daily_returns[['ACWI_log', 'SP500_log']].dropna()
  slope_d, intercept_d, r_d, p_d, std_err_d = stats.linregress(
      overlap_daily['SP500_log'], overlap_daily['ACWI_log'])
  print(f"ACWI = {slope_d:.4f} * SP500 + {intercept_d:.6f}")
  print(f"R-squared: {r_d**2:.4f}")

  residuals_d = overlap_daily['ACWI_log'] - (slope_d * overlap_daily['SP500_log'] + intercept_d)
  res_noise_d_list = asset_model.find_best_distribution(residuals_d, top_n=1)
  if res_noise_d_list:
    res_noise_d = res_noise_d_list[0]
    print(
        f"Noise Best Fit: dist={res_noise_d['name']}, params={res_noise_d['params']}, BIC={res_noise_d['bic']:.2f}, MSE={res_noise_d['mse']:.6f}"
    )
  else:
    print("Noise Best Fit: Failed")

  print("\n=== Correlation ACWI vs S&P500 (Monthly) ===")
  overlap_monthly = monthly_returns[['ACWI_log', 'SP500_log']].dropna()
  slope_m, intercept_m, r_m, p_m, std_err_m = stats.linregress(
      overlap_monthly['SP500_log'], overlap_monthly['ACWI_log'])
  print(f"ACWI = {slope_m:.4f} * SP500 + {intercept_m:.6f}")
  print(f"R-squared: {r_m**2:.4f}")

  residuals_m = overlap_monthly['ACWI_log'] - (slope_m * overlap_monthly['SP500_log'] + intercept_m)
  res_noise_m_list = asset_model.find_best_distribution(residuals_m, top_n=1)
  if res_noise_m_list:
    res_noise_m = res_noise_m_list[0]
    print(
        f"Noise Best Fit: dist={res_noise_m['name']}, params={res_noise_m['params']}, BIC={res_noise_m['bic']:.2f}, MSE={res_noise_m['mse']:.6f}"
    )
  else:
    print("Noise Best Fit: Failed")

  print("\n=== BTC 4-Year Seasonality & MR-GBM ===")
  btc_data = df.set_index('Date')['BTC'].dropna()
  if isinstance(btc_data, pd.Series):
    # 1yr and 4yr seasonality removal
    # Calculate returns first
    _btc_ret = btc_data / btc_data.shift(1)
    # Use explicit cast to handle mypy concerns
    _btc_log_vals = np.log(_btc_ret.values.astype(float))
    btc_log_ret = pd.Series(_btc_log_vals, index=_btc_ret.index).dropna()

    # Calculate mean return for each of the 48 "cycle months"
    # Cycle starts from the first month of BTC data
    start_date = btc_log_ret.index[0]

    # De-seasonalize
    def get_cycle_month(d):
      return ((d.year - start_date.year) * 12 + (d.month - start_date.month)) % 48

    cycle_idx = btc_log_ret.index.map(get_cycle_month)
    # Convert to Series for mapping
    cycle_48m = pd.Series(cycle_idx, index=btc_log_ret.index)
    seasonal_means = btc_log_ret.groupby(cycle_48m).mean()

    # Apply de-seasonality to log returns
    _seasonal_comp = cast(pd.Series, cycle_48m.map(seasonal_means))
    btc_log_ret_clean = btc_log_ret - _seasonal_comp

    # Reconstruct price series for MR-GBM (relative to start price)
    btc_price_clean = np.exp(btc_log_ret_clean.cumsum()) * btc_data.iloc[0]

    dt = 1 / 365.25  # Daily dt
    mrgbm_res = asset_model.calculate_mrgbm(btc_price_clean, dt)
    if mrgbm_res:
      print(f"De-seasonalized BTC MR-GBM:")
      print(
          f"  Theta: {mrgbm_res['theta']:.4f}, Mu: {mrgbm_res['mu']:.4f}, Sigma: {mrgbm_res['sigma']:.4f}"
      )

      # Fit noise to t, laplace
      mrgbm_residuals = mrgbm_res['residuals']
      print(f"Residual Noise Fitting:")
      for dist_name, dist in [('t', stats.t), ('laplace', stats.laplace), ('norm', stats.norm)]:
        params = dist.fit(mrgbm_residuals)
        loglik = dist.logpdf(mrgbm_residuals, *params).sum()
        k = len(params)
        n = len(mrgbm_residuals)
        bic = k * np.log(n) - 2 * loglik
        print(f"  {dist_name}: params={params}, BIC={bic:.2f}")

    # Show 4-year cycle component (averages)
    print("\n=== BTC 4-Year Cycle (Monthly Averages) ===")
    for m in range(48):
      if m in seasonal_means:
        print(f"  Cycle Month {m:02d}: {seasonal_means[m]:.4f}")

  print("\n=== MR-GBM for S&P 500 and ACWI (Daily) ===")
  dt = 1 / 365.25
  for asset in ['SP500', 'ACWI']:
    asset_prices = df.set_index('Date')[asset].dropna()
    res = asset_model.calculate_mrgbm(asset_prices, dt)
    if res and not np.isnan(res['theta']):
      print(f"{asset} MR-GBM:")
      print(f"  Theta: {res['theta']:.4f}, Mu: {res['mu']:.4f}, Sigma: {res['sigma']:.4f}")
    else:
      print(f"{asset} MR-GBM: No mean reversion detected (theta=NaN)")

  print("\n=== BTC Model Effectiveness Analysis ===")
  btc_data = df.set_index('Date')['BTC'].dropna()
  dt = 1 / 365.25

  # Baseline: Simple GBM (Log-Normal)
  # X_t = log(P_t). dx_t = mu_gbm * dt + sigma_gbm * dW_t
  _btc_div_s = btc_data.pct_change().dropna() + 1.0
  log_rets = np.log(_btc_div_s)
  res_baseline = asset_model.fit_normal_log(log_rets)
  print(f"a) Baseline (Log-Normal): BIC={res_baseline['bic']:.2f}")

  # Cycle Only: Remove cycle, then fit Normal to residuals
  # We reuse seasonal_means from above (calculated on original BTC log returns)
  # Note: In the actual implementation above, btc_log_ret_clean was used.
  res_cycle = asset_model.fit_normal_log(btc_log_ret_clean)
  print(f"b) Cycle Only: BIC={res_cycle['bic']:.2f}")

  # MR-GBM Only: Fit MR-GBM to original BTC price
  mrgbm_orig = asset_model.calculate_mrgbm(btc_data, dt)
  if mrgbm_orig and not np.isnan(mrgbm_orig['theta']):
    resids = mrgbm_orig['residuals']
    res_mrgbm = asset_model.fit_normal_log(resids)
    print(f"c) MR-GBM Only: BIC={res_mrgbm['bic']:.2f}")
  else:
    print(f"c) MR-GBM Only: No mean reversion detected")

  # Cycle + MR-GBM: Already calculated as mrgbm_res
  if mrgbm_res and not np.isnan(mrgbm_res['theta']):
    resids_both = mrgbm_res['residuals']
    res_both = asset_model.fit_normal_log(resids_both)
    print(f"d) Cycle + MR-GBM: BIC={res_both['bic']:.2f}")


if __name__ == '__main__':
  analyze()
