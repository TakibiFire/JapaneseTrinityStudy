"""為替（ドル円）の月次データから年次の平均リターン（mu）と標準偏差（sigma）を計算し比較する。

1986年〜、2000年〜、2013年〜の3つの期間において、
月次リターンを算出し、それを年率換算（mu × 12, sigma × √12）して出力する。
また、実績値と計算されたmuによるフィッティングラインをグラフ化して保存する。
"""

import csv
import math
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd


def calculate_monthly_returns(prices: List[float]) -> List[float]:
  """価格データから月次リターンを計算する。"""
  returns = []
  for i in range(1, len(prices)):
    prev_p = prices[i - 1]
    curr_p = prices[i]
    monthly_return = (curr_p - prev_p) / prev_p
    returns.append(monthly_return)
  return returns


def compute_annualized_mu_sigma(returns: List[float]) -> Tuple[float, float]:
  """月次リターンのリストから、年率換算したmuとsigmaを計算する。"""
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

  chart.save(output_path)
  print(f"グラフを保存しました: {output_path}")


def main() -> None:
  file_path = "data/fm08_m_1.csv"

  all_dates: List[datetime] = []
  all_prices: List[float] = []

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

          if dt.year >= 1986:
            all_dates.append(dt)
            all_prices.append(price)

        except ValueError:
          continue
  except Exception:
    # Shift-JISで失敗した場合はUTF-8で再試行
    all_dates.clear()
    all_prices.clear()
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
          if dt.year >= 1986:
            all_dates.append(dt)
            all_prices.append(price)
        except ValueError:
          continue

  if not all_dates:
    print("データが読み込めませんでした。")
    return

  # インデックスを探す
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
  plot_fx_with_fits(all_dates, all_prices, fits_info, "docs/imgs/fx.svg")


if __name__ == "__main__":
  main()
