"""
analyze_ppp_main.py

購買力平価（PPP）に関連して、為替変動と日本のインフレ率（CPI変化率）の相関を分析する。
以下の回帰モデルを仮定し、パラメータ a, b および残差（noise）の統計量を算出する。

  Δlog(CPI)_t = a * Δlog(FX)_t + b + noise

また、インフレ率の自己相関についても確認する。
"""

import csv
import math
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats


def load_cpi_data(file_path: str, start_year: int = 1970) -> pd.Series:
  """CPIデータを読み込み、年次対数リターンのSeriesを返す。"""
  cpi_values = []
  with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
      if row and row[0].strip():
        try:
          cpi_values.append(float(row[0]))
        except ValueError:
          continue

  years = range(start_year, start_year + len(cpi_values))
  df_cpi = pd.DataFrame({"Year": years, "CPI": cpi_values})
  df_cpi["CPI_log_ret"] = np.log(df_cpi["CPI"] / df_cpi["CPI"].shift(1))
  return df_cpi.set_index("Year")["CPI_log_ret"].dropna()


def load_fx_data(file_path: str) -> pd.Series:
  """FXデータを読み込み、年次（12月末）の対数リターンのSeriesを返す。"""
  dates: List[datetime] = []
  prices: List[float] = []

  # analyze_fx_main.py と同様の読み込みロジック
  try:
    with open(file_path, mode='r', encoding='shift_jis') as f:
      reader = csv.reader(f)
      for _ in range(8):
        next(reader, None)
      for row in reader:
        if not row or not row[0].strip():
          continue
        try:
          dt = datetime.strptime(row[0].strip(), "%Y/%m")
          price = float(row[1].strip())
          dates.append(dt)
          prices.append(price)
        except (ValueError, IndexError):
          continue
  except Exception:
    with open(file_path, mode='r', encoding='utf-8') as f:
      reader = csv.reader(f)
      for _ in range(8):
        next(reader, None)
      for row in reader:
        if not row or not row[0].strip():
          continue
        try:
          dt = datetime.strptime(row[0].strip(), "%Y/%m")
          price = float(row[1].strip())
          dates.append(dt)
          prices.append(price)
        except (ValueError, IndexError):
          continue

  df_fx_monthly = pd.DataFrame({"Date": dates, "FX": prices})
  df_fx_monthly["Year"] = df_fx_monthly["Date"].dt.year
  df_fx_monthly["Month"] = df_fx_monthly["Date"].dt.month

  # 各年の12月の値を取得
  df_fx_annual = df_fx_monthly[df_fx_monthly["Month"] == 12].copy()
  df_fx_annual["FX_log_ret"] = np.log(df_fx_annual["FX"] /
                                      df_fx_annual["FX"].shift(1))
  return df_fx_annual.set_index("Year")["FX_log_ret"].dropna()


def main() -> None:
  cpi_log_ret = load_cpi_data("data/cpi_yearly_1970.csv")
  fx_log_ret = load_fx_data("data/fm08_m_1.csv")

  # データの結合（年次）
  df = pd.concat([cpi_log_ret, fx_log_ret], axis=1).dropna()
  df.columns = pd.Index(["CPI_log_ret", "FX_log_ret"])

  print(f"分析期間: {df.index.min()}年 - {df.index.max()}年")
  print(f"データポイント数: {len(df)}")

  # 回帰分析
  slope, intercept, r_value, p_value, std_err = stats.linregress(
      df["FX_log_ret"], df["CPI_log_ret"])

  print("\n--- 回帰分析結果 (Δlog(CPI) = a * Δlog(FX) + b) ---")
  print(f"a (傾き): {slope:.4f}")
  print(f"b (切片): {intercept:.4f}")
  print(f"R-squared: {r_value**2:.4f}")
  print(f"p-value: {p_value:.4f}")

  # 残差の分析
  df["Residual"] = df["CPI_log_ret"] - (slope * df["FX_log_ret"] + intercept)
  res_mu = df["Residual"].mean()
  res_std = df["Residual"].std()

  print("\n--- 残差 (noise) の統計量 ---")
  print(f"平均: {res_mu:.6f}")
  print(f"標準偏差 (sigma_noise): {res_std:.6f}")

  # インフレ率の自己相関
  autocorr = df["CPI_log_ret"].autocorr(lag=1)
  print("\n--- インフレ率の自己相関 (lag=1) ---")
  print(f"自己相関係数: {autocorr:.4f}")

  # グラフ作成（散布図と回帰直線）
  scatter = alt.Chart(df).mark_point().encode(
      x=alt.X("FX_log_ret:Q", title="為替変動率 (Δlog(FX))"),
      y=alt.Y("CPI_log_ret:Q", title="インフレ率 (Δlog(CPI))"),
      tooltip=["CPI_log_ret", "FX_log_ret"])

  # 回帰直線用のデータ
  x_range = np.linspace(df["FX_log_ret"].min(), df["FX_log_ret"].max(), 100)
  y_range = slope * x_range + intercept
  df_line = pd.DataFrame({"FX_log_ret": x_range, "CPI_log_ret": y_range})

  line = alt.Chart(df_line).mark_line(color="red").encode(
      x="FX_log_ret:Q", y="CPI_log_ret:Q")

  chart = (scatter + line).properties(width=500, height=400, title="為替変動とインフレ率の相関分析")

  output_path = "docs/imgs/ppp_regression.svg"
  chart.save(output_path)
  print(f"\nグラフを保存しました: {output_path}")

  # 時系列の推移も確認
  df_reset = df.reset_index()
  df_melt = df_reset.melt(id_vars="Year",
                          value_vars=["CPI_log_ret", "FX_log_ret"],
                          var_name="Type",
                          value_name="Value")
  
  time_chart = alt.Chart(df_melt).mark_line().encode(
      x=alt.X("Year:O", title="年"),
      y=alt.Y("Value:Q", title="変動率"),
      color="Type:N").properties(width=800, height=300, title="為替変動とインフレ率の時系列推移")
  
  output_time_path = "docs/imgs/ppp_timeseries.svg"
  time_chart.save(output_time_path)
  print(f"時系列グラフを保存しました: {output_time_path}")


if __name__ == "__main__":
  main()
