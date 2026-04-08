"""
analyze_monthly_cpi_main.py

月次CPIデータを分析し、月次対数リターンの自己相関構造（ARモデル）を調査する。
AICを用いて最適なラグ数 p を自動的に選択する。
"""

from typing import Any, List, Tuple, cast

import altair as alt
import numpy as np
import pandas as pd


def load_monthly_cpi(file_path: str, start_year: int = 1970) -> pd.DataFrame:
  """月次CPIデータを読み込み、日付と月次対数リターンのDataFrameを返す。"""
  # BOM付きUTF-8を考慮
  df = pd.read_csv(file_path, encoding='utf-8-sig')

  # 4番目のカラム（インデックス3）がCPI
  cpi_col = df.columns[3]
  df['CPI'] = pd.to_numeric(df[cpi_col], errors='coerce')
  df = df.dropna(subset=['CPI'])

  # 日付の変換
  df['Date_str'] = df['時点'].str.replace('年', '-').str.replace('月', '')
  df['Date'] = pd.to_datetime(df['Date_str'], format='%Y-%m')
  
  if start_year > 1970:
    df = df[df['Date'] >= f'{start_year}-01-01']

  # 対数リターンを計算
  df['log_ret'] = np.log(df['CPI'] / df['CPI'].shift(1))
  return df.dropna(subset=['log_ret'])


def fit_ar_p(rets: np.ndarray, p: int) -> Tuple[float, float, float]:
  """
    AR(p)モデルを最小二乗法でフィットし、AICを計算する。
    r_t = c + phi_1 * r_{t-1} + ... + phi_p * r_{t-p} + e_t
    """
  n = len(rets)
  if p == 0:
    mu = float(np.mean(rets))
    rss = float(np.sum((rets - mu)**2))
    k = 1  # mean only
    aic = n * np.log(rss / n) + 2 * k
    return float(aic), float(rss), 0.0  # phi is 0

  # デザイン行列の作成
  Y = rets[p:]
  X = np.ones((len(Y), p + 1), dtype=np.float64)
  for i in range(1, p + 1):
    X[:, i] = rets[p - i:-i]

  beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None) # type: ignore

  if residuals.size > 0:
    rss = float(residuals[0])
  else:
    rss = float(np.sum((Y - X @ beta)**2))

  k = p + 1
  n_effective = len(Y)
  aic = n_effective * np.log(rss / n_effective) + 2 * k

  phi1 = float(beta[1]) if p >= 1 else 0.0
  return float(aic), float(rss), phi1


def analyze_period(file_path: str, start_year: int):
  df = load_monthly_cpi(file_path, start_year)
  rets = df['log_ret'].to_numpy()
  print(f"\n===== 分析期間: {start_year}年以降 (データ数: {len(rets)}ヶ月) =====")
  
  results = []
  for p in range(13):
    aic, rss, phi1 = fit_ar_p(rets, p)
    results.append({"p": p, "AIC": aic, "RSS": rss, "phi1": phi1})
  
  df_res = pd.DataFrame(results)
  best_idx = df_res['AIC'].idxmin()
  best_p = int(df_res.loc[best_idx, 'p']) # type: ignore
  print(f"AICによる最適ラグ数: p = {best_p}")

  def print_details(p):
    aic, rss, _ = fit_ar_p(rets, p)
    Y = rets[p:]
    X = np.ones((len(Y), p + 1), dtype=np.float64)
    for i in range(1, p + 1):
      X[:, i] = rets[p - i:-i]
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None) # type: ignore
    print(f"--- AR({p}) 詳細 ---")
    print(f"  定数項 (c): {beta[0]:.8f}")
    phis = beta[1:]
    print(f"  Phis: {phis.tolist()}")
    print(f"  残差標準偏差: {np.sqrt(rss/len(Y)):.8f}")

  print_details(1)
  if best_p > 1:
    print_details(best_p)

  # 年次リターンに集計して自己相関を確認
  df['Year'] = df['Date'].dt.year
  rets_annual = df.groupby('Year')['log_ret'].sum()
  print(f"集計された年次対数リターンの自己相関 (lag=1 year): {rets_annual.autocorr(lag=1):.4f}")

  # 直近12ヶ月の対数リターンを表示
  print("直近12ヶ月の対数リターン (古い順):")
  print(df['log_ret'].tail(12).tolist())


def main():
  file_path = "data/cpi_monthly_1970.csv"
  analyze_period(file_path, 1970)
  analyze_period(file_path, 1981)


if __name__ == "__main__":
  main()
