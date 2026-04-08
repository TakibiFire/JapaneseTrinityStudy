"""CPIデータの年次リターンの平均（mu）と標準偏差（sigma）を計算し、グラフを描画する。

CPIデータは
https://dashboard.e-stat.go.jp/timeSeriesResult?indicatorCode=0703010501010090000
から得られる時系列データを利用。
"""

import csv
import math
from typing import List, Tuple

import altair as alt
import pandas as pd


def calculate_annual_returns(cpi_data: List[float]) -> List[float]:
  """
  CPIデータリストから前年比の年次リターン（算術リターン）を計算する。
  
  Args:
    cpi_data: 各年のCPI（消費者物価指数）のリスト。
    
  Returns:
    年次リターンのリスト。要素数はlen(cpi_data) - 1となる。
  """
  returns = []
  for i in range(1, len(cpi_data)):
    prev_cpi = cpi_data[i - 1]
    curr_cpi = cpi_data[i]
    annual_return = (curr_cpi - prev_cpi) / prev_cpi
    returns.append(annual_return)
  return returns


def compute_mu_sigma(returns: List[float]) -> Tuple[float, float]:
  """
  リターンのリストから平均（mu）と標本標準偏差（sigma）を計算する。
  
  Args:
    returns: 年次リターンのリスト。
    
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
    mu: 年次リターンの平均。
    sigma: 年次リターンの標本標準偏差。
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
  returns = calculate_annual_returns(cpi_data)
  mu, sigma = compute_mu_sigma(returns)

  print(f"データポイント数: {len(cpi_data)}年分")
  print(f"計算されたリターン数: {len(returns)}年分")
  print(f"Mu (平均年次リターン): {mu:.6f}")
  print(f"Sigma (年次リターンの標本標準偏差): {sigma:.6f}")

  # グラフの描画 (1970年開始と仮定)
  plot_cpi_with_fits(cpi_data,
                     start_year=1970,
                     mu=mu,
                     sigma=sigma,
                     output_path="docs/imgs/cpi.svg")


if __name__ == "__main__":
  main()
