"""CPIデータの年次リターンの平均（mu）と標準偏差（sigma）を計算し、グラフを描画する。

CPIデータは
https://dashboard.e-stat.go.jp/timeSeriesResult?indicatorCode=0703010501010090000
から得られる時系列データを利用。
"""

import csv
import math
import os
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd


def calculate_annual_returns(cpi_data: List[float]) -> Tuple[List[float], List[float]]:
  """
  CPIデータリストから前年比の年次リターン（算術リターンおよび対数リターン）を計算する。
  
  Args:
    cpi_data: 各年のCPI（消費者物価指数）のリスト。
    
  Returns:
    (算術リターンのリスト, 対数リターンのリスト) のタプル。
  """
  simple_returns = []
  log_returns = []
  for i in range(1, len(cpi_data)):
    prev_cpi = cpi_data[i - 1]
    curr_cpi = cpi_data[i]
    # 算術リターン
    simple_return = (curr_cpi - prev_cpi) / prev_cpi
    simple_returns.append(simple_return)
    # 対数リターン
    log_return = math.log(curr_cpi / prev_cpi)
    log_returns.append(log_return)
  return simple_returns, log_returns


def compute_mu_sigma(returns: List[float]) -> Tuple[float, float]:
  """
  リターンのリストから平均（mu）と標本標準偏差（sigma）を計算する。
  
  Args:
    returns: リターンのリスト。
    
  Returns:
    平均(mu)と標本標準偏差(sigma)のタプル。
  """
  if not returns:
    return 0.0, 0.0

  n = len(returns)
  mu = sum(returns) / n

  if n < 2:
    return mu, 0.0

  variance = sum((r - mu)**2 for r in returns) / (n - 1)
  sigma = math.sqrt(variance)

  return mu, sigma


def plot_cpi_with_fits(cpi_data: List[float], start_year: int, mu: float,
                       sigma: float, output_path: str) -> None:
  """
  CPIデータとmu, sigmaに基づいたフィッティングラインを描画しSVGとして保存する。
  
  Args:
    cpi_data: 各年のCPIデータのリスト。
    start_year: データの開始年。
    mu: 年次リターンの平均（算術）。
    sigma: 年次リターンの標本標準偏差（算術）。
    output_path: 保存先のファイルパス。
  """
  years = list(range(start_year, start_year + len(cpi_data)))
  cpi_1970 = cpi_data[0]

  # 理論値を計算
  mu_fits = [cpi_1970 * ((1 + mu)**(year - start_year)) for year in years]

  # Altair用のデータフレームを作成
  df_actual = pd.DataFrame({
      "Year": years,
      "CPI": cpi_data,
      "Type": ["Actual CPI"] * len(years)
  })

  df_mu = pd.DataFrame({
      "Year": years,
      "CPI": mu_fits,
      "Type": [f"Fit (mu={mu:.4f})"] * len(years)
  })

  df = pd.concat([df_actual, df_mu])

  # グラフの描画
  chart = alt.Chart(df).mark_line().encode(
      x=alt.X("Year:O", title="年"),
      y=alt.Y("CPI:Q", title="消費者物価指数 (2020=100)"),
      color=alt.Color("Type:N",
                      title="データ種類",
                      scale=alt.Scale(
                          domain=["Actual CPI", f"Fit (mu={mu:.4f})"],
                          range=["blue", "red"])),
      strokeDash=alt.condition(
          alt.datum.Type == "Actual CPI",
          alt.value([1, 0]),  # 実線
          alt.value([5, 5])  # 点線
      )).properties(width=600, height=300, title="日本 CPI (1970-2025) と理論値の比較")

  # ディレクトリ作成
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")


def main() -> None:
  """
  メイン関数。CSVファイルを読み込み、計算結果を出力し、グラフを描画する。
  """
  file_path = "data/cpi_yearly_1970.csv"
  cpi_data: List[float] = []

  # CSVファイルからCPIデータを読み込む
  with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
      if row and row[0].strip():
        try:
          cpi_data.append(float(row[0]))
        except ValueError:
          continue

  if not cpi_data:
    print("有効なCPIデータが見つかりませんでした。")
    return

  # リターン、mu、sigmaを計算する
  simple_returns, log_returns = calculate_annual_returns(cpi_data)
  
  # 算術リターンの統計 (YearlyLogNormalArithmetic用)
  mu_arith, sigma_arith = compute_mu_sigma(simple_returns)
  
  # 対数リターンの統計 (MonthlyLogNormal用)
  mu_log_annual, sigma_log_annual = compute_mu_sigma(log_returns)
  mu_log_monthly = mu_log_annual / 12.0
  sigma_log_monthly = sigma_log_annual / math.sqrt(12.0)

  print(f"データポイント数: {len(cpi_data)}年分")
  print(f"計算されたリターン数: {len(simple_returns)}年分")
  
  print("\n--- For YearlyLogNormalArithmetic (Annual Arithmetic) ---")
  print(f"Mu (arithmetic): {mu_arith:.6f}")
  print(f"Sigma (arithmetic): {sigma_arith:.6f}")
  
  print("\n--- For MonthlyLogNormal (Monthly Log-Return) ---")
  print(f"Mu (log monthly): {mu_log_monthly:.6f}")
  print(f"Sigma (log monthly): {sigma_log_monthly:.6f}")
  
  print("\n--- Annual Log-Return Stats (Reference) ---")
  print(f"Mu (log annual): {mu_log_annual:.6f}")
  print(f"Sigma (log annual): {sigma_log_annual:.6f}")

  # グラフの描画 (1970年開始と仮定)
  output_path = "docs/imgs/cpi/historical_trend.svg"
  plot_cpi_with_fits(cpi_data,
                     start_year=1970,
                     mu=mu_arith,
                     sigma=sigma_arith,
                     output_path=output_path)


if __name__ == "__main__":
  main()
