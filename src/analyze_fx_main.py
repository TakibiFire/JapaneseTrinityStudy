"""
1. 為替（ドル円）の月次データから年次の平均リターン（mu）と標準偏差（sigma）を計算し比較する。

1986年〜、2000年〜、2013年〜の3つの期間において、
月次リターンを算出し、それを年率換算（mu × 12, sigma × √12）して出力する。
また、実績値と計算されたmuによるフィッティングラインをグラフ化して保存する。

2. USD/JPYとグローバル株式インデックス（S&P 500、ACWI）の月次対数リターンの相関係数
（ピアソン）を計算し、出力する。
- データの重複期間を使用（S&P 500: 1973年以降、ACWI: 2008年以降）。
- `data/asset_daily_prices.csv` を読み込み、月末基準（ME）でリサンプリングして対数リターンを算出。
- `data/fm08_m_1.csv` から抽出した月末時点のUSD/JPY為替レートの対数リターンと結合して相関を評価する。
"""

import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.lib import asset_model


def calculate_monthly_returns(prices: List[float]) -> List[float]:
  """価格データから月次単利リターンを計算する。"""
  returns = []
  for i in range(1, len(prices)):
    prev_p = prices[i - 1]
    curr_p = prices[i]
    monthly_return = (curr_p - prev_p) / prev_p
    returns.append(monthly_return)
  return returns


def compute_annualized_mu_sigma(returns: List[float]) -> Tuple[float, float]:
  """月次単利リターンのリストから、年率換算したmuとsigmaを計算する。"""
  if not returns:
    return 0.0, 0.0

  n = len(returns)
  mu_monthly = sum(returns) / n

  if n < 2:
    return mu_monthly * 12, 0.0

  variance_monthly = sum((r - mu_monthly)**2 for r in returns) / (n - 1)
  sigma_monthly = math.sqrt(variance_monthly)

  mu_annual = mu_monthly * 12
  sigma_annual = sigma_monthly * math.sqrt(12)

  return mu_annual, sigma_annual


def plot_fx_with_fits(dates: List[datetime], prices: List[float],
                       fits: Dict[str, Tuple[int,
                                             float]], output_path: str) -> None:
  """実績の為替レートと各期間のmuに基づくフィッティングラインを描画する。

  Args:
    dates: 日付リスト（1986年以降）
    prices: 価格リスト（1986年以降の実績）
    fits: {"ラベル名": (開始年のインデックス, mu_annual)} の辞書
    output_path: 保存先のファイルパス
  """
  # 実績データのDataFrame
  df_actual = pd.DataFrame({
      "Date": dates,
      "FX Rate": prices,
      "Type": ["Actual FX Rate"] * len(dates)
  })

  dfs = [df_actual]

  # 各期間のフィッティングラインを作成
  for label, (start_idx, mu_annual) in fits.items():
    mu_monthly = mu_annual / 12.0
    start_date = dates[start_idx]
    start_price = prices[start_idx]

    # フィッティング用の日付と価格のリスト
    fit_dates = dates[start_idx:]
    fit_prices = [
        start_price * ((1 + mu_monthly)**i) for i in range(len(fit_dates))
    ]

    df_fit = pd.DataFrame({
        "Date": fit_dates,
        "FX Rate": fit_prices,
        "Type": [f"Fit {label}"] * len(fit_dates)
    })
    dfs.append(df_fit)

  df = pd.concat(dfs)

  # グラフの描画
  chart = alt.Chart(df).mark_line().encode(
      x=alt.X("Date:T", title="年月"),
      y=alt.Y("FX Rate:Q", title="ドル円レート (円)"),
      color=alt.Color("Type:N",
                      title="データ種類",
                      scale=alt.Scale(domain=[
                          "Actual FX Rate", "Fit 1986-", "Fit 2000-",
                          "Fit 2013-"
                      ],
                                      range=["blue", "green", "orange",
                                             "red"])),
      strokeDash=alt.condition(
          alt.datum.Type == "Actual FX Rate",
          alt.value([1, 0]),  # 実線
          alt.value([5, 5])  # 点線
      )).properties(width=800, height=400, title="ドル円レート実績 (1986-) と各期間の理論値の比較")

  # ディレクトリ作成
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")


def analyze_correlations(fx_dates: List[datetime], fx_prices: List[float]) -> None:
  """USD/JPYと株式インデックス（S&P 500, ACWI）の月次対数リターンの相関を計算・表示する。"""
  # 1. USD/JPY の DataFrame 作成と月次対数リターンの計算
  # 日付を月の末日に揃える (Period('M') を使用)
  df_fx = pd.DataFrame({
      'Date': pd.to_datetime(fx_dates),
      'USDJPY': fx_prices
  })
  df_fx['Month'] = df_fx['Date'].dt.to_period('M')
  # 月末時点で一意にする（元のデータは月次なので基本はそのまま）
  df_fx = df_fx.drop_duplicates(subset=['Month'], keep='last').set_index('Month')
  
  # USDJPY の対数リターンを計算
  df_fx['USDJPY_log'] = np.log(df_fx['USDJPY'] / df_fx['USDJPY'].shift(1))

  # 2. 株式インデックスの読み込みと月次対数リターンの計算
  df_assets = pd.read_csv('data/asset_daily_prices.csv')
  df_assets['Date'] = pd.to_datetime(df_assets['Date'])
  # asset_model.process_returns は 'M'（内部的には 'ME'） でリサンプリングし、対数リターンを計算
  monthly_returns = asset_model.process_returns(df_assets, 'M')
  
  # 月次リターンのインデックス（DatetimeIndexの月末日）を Period('M') に変換してマージしやすくする
  monthly_returns.index = monthly_returns.index.to_period('M')

  # 3. データの結合
  # USDJPY の対数リターンと、S&P 500, ACWI の対数リターンを結合
  merged_df = pd.concat([
      df_fx['USDJPY_log'], 
      monthly_returns['SP500_log'], 
      monthly_returns['ACWI_log']
  ], axis=1).dropna(how='all')

  print("\n=== 相関係数分析 (USD/JPY vs 株価インデックス 月次対数リターン) ===")
  
  # S&P 500 との相関
  sp500_data = merged_df[['USDJPY_log', 'SP500_log']].dropna()
  if not sp500_data.empty:
    corr_sp500, p_val_sp500 = stats.pearsonr(sp500_data['USDJPY_log'], sp500_data['SP500_log'])
    start_dt = sp500_data.index[0]
    end_dt = sp500_data.index[-1]
    print(f"USD/JPY vs S&P 500:")
    print(f"  期間: {start_dt} 〜 {end_dt} ({len(sp500_data)}ヶ月)")
    print(f"  相関係数 (Pearson r): {corr_sp500:.4f} (p-value: {p_val_sp500:.4e})")
  
  # ACWI との相関
  acwi_data = merged_df[['USDJPY_log', 'ACWI_log']].dropna()
  if not acwi_data.empty:
    corr_acwi, p_val_acwi = stats.pearsonr(acwi_data['USDJPY_log'], acwi_data['ACWI_log'])
    start_dt = acwi_data.index[0]
    end_dt = acwi_data.index[-1]
    print(f"\nUSD/JPY vs ACWI:")
    print(f"  期間: {start_dt} 〜 {end_dt} ({len(acwi_data)}ヶ月)")
    print(f"  相関係数 (Pearson r): {corr_acwi:.4f} (p-value: {p_val_acwi:.4e})")


def main() -> None:
  """メイン関数。"""
  file_path = "data/fm08_m_1.csv"

  all_dates: List[datetime] = []
  all_prices: List[float] = []
  
  # 相関計算用（全期間）
  all_dates_full: List[datetime] = []
  all_prices_full: List[float] = []

  try:
    with open(file_path, mode='r', encoding='shift_jis') as f:
      reader = csv.reader(f)
      for _ in range(8):
        next(reader, None)
      for row in reader:
        if not row or not row[0].strip():
          continue
        date_str = row[0].strip()
        price_str = row[1].strip()
        if not price_str:
          continue
        try:
          dt = datetime.strptime(date_str, "%Y/%m")
          price = float(price_str)
          
          # 相関計算用には1973年以降の全データを保持
          all_dates_full.append(dt)
          all_prices_full.append(price)

          if dt.year >= 1986:
            all_dates.append(dt)
            all_prices.append(price)

        except ValueError:
          continue
  except Exception:
    # Shift-JISで失敗した場合はUTF-8で再試行
    all_dates.clear()
    all_prices.clear()
    all_dates_full.clear()
    all_prices_full.clear()
    with open(file_path, mode='r', encoding='utf-8') as f:
      reader = csv.reader(f)
      for _ in range(8):
        next(reader, None)
      for row in reader:
        if not row or not row[0].strip():
          continue
        date_str = row[0].strip()
        price_str = row[1].strip()
        if not price_str:
          continue
        try:
          dt = datetime.strptime(date_str, "%Y/%m")
          price = float(price_str)
          
          all_dates_full.append(dt)
          all_prices_full.append(price)
          
          if dt.year >= 1986:
            all_dates.append(dt)
            all_prices.append(price)
        except ValueError:
          continue

  if not all_dates:
    print("データが読み込めませんでした。")
    return

  # インデックスを探す (グラフ用: 1986年以降)
  idx_1986 = 0
  idx_2000 = next((i for i, d in enumerate(all_dates) if d.year >= 2000), -1)
  idx_2013 = next((i for i, d in enumerate(all_dates) if d.year >= 2013), -1)

  # 期間ごとのデータと結果を格納
  periods = {"1986-": idx_1986, "2000-": idx_2000, "2013-": idx_2013}

  fits_info = {}

  for label, start_idx in periods.items():
    if start_idx == -1:
      print(f"{label}: データがありません")
      continue

    prices_period = all_prices[start_idx:]
    returns = calculate_monthly_returns(prices_period)
    mu, sigma = compute_annualized_mu_sigma(returns)

    fits_info[label] = (start_idx, mu)

    print(f"--- {label} 以降 ---")
    print(f"データポイント数: {len(prices_period)}ヶ月分 ({len(prices_period)/12:.1f}年)")
    print(f"Mu (年率換算リターン): {mu:.4f} ({mu*100:.2f}%)")
    print(f"Sigma (年率換算リスク): {sigma:.4f} ({sigma*100:.2f}%)")
    print()

  # グラフ描画
  output_path = "docs/imgs/forex/historical_trend.svg"
  plot_fx_with_fits(all_dates, all_prices, fits_info, output_path)

  # 追加機能: 相関係数の分析
  if all_dates_full and all_prices_full:
    analyze_correlations(all_dates_full, all_prices_full)


if __name__ == "__main__":
  main()
