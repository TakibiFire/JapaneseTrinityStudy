"""
S&P500およびACWIの価格データを用いたファットテールの分析。

- 日次・月次リターンの分布の可視化
- 正規分布との比較
- 最適な非対称分布のフィッティング
- 統計的指標（AIC, BIC）の算出
"""

import os
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from src.lib.asset_model import (find_best_distribution_with_fixed_mean,
                                 process_returns)
from src.lib.data_collection import fetch_asset_data

# 分析対象とする「30年間」の期間定義
START_30Y = pd.Timestamp("1995-01-01")
END_30Y = pd.Timestamp("2025-12-31")


def save_chart(chart: alt.Chart, filename: str) -> None:
  """AltairチャートをSVGとして保存する"""
  path = os.path.join("docs/imgs/fat_tails", filename)
  chart.save(path)
  print(f"✅ {path} を作成しました。")


def plot_dist_with_norm(data: pd.Series,
                        title: str,
                        filename: str,
                        zoom_tail: bool = False) -> None:
  """データと正規分布のPDFを重ねてプロットする (Altair版)"""
  data = data.dropna()
  mu, std = stats.norm.fit(data)

  # ヒストグラム用データの作成
  counts, bin_edges = np.histogram(data, bins=100, density=True)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  df_hist = pd.DataFrame({'リターン': bin_centers, '頻度': counts, 'タイプ': '生データ'})

  # 正規分布の曲線データ作成
  x = np.linspace(data.min(), data.max(), 1000)
  p = stats.norm.pdf(x, mu, std)
  df_norm = pd.DataFrame({
      'リターン': x,
      '頻度': p,
      'タイプ': f'正規分布へのフィット (μ={mu:.4f}, σ={std:.4f})'
  })

  if zoom_tail:
    # 左側の裾をズーム
    threshold = mu - 2 * std
    df_hist = df_hist[df_hist['リターン'] < threshold]
    df_norm = df_norm[df_norm['リターン'] < threshold]
    title += " (左側の裾をズーム)"

  # チャート作成
  hist = alt.Chart(df_hist).mark_bar(opacity=0.6,
                                     color='gray').encode(x=alt.X('リターン:Q',
                                                                  title='リターン'),
                                                          y=alt.Y('頻度:Q',
                                                                  title='頻度'))

  line = alt.Chart(df_norm).mark_line(color='red', size=2).encode(
      x='リターン:Q',
      y='頻度:Q',
      color=alt.Color('タイプ:N',
                      scale=alt.Scale(range=['red']),
                      legend=alt.Legend(orient='top')))

  chart = (hist + line).properties(width=600, height=400,
                                   title=title).configure_axis(grid=True)

  save_chart(chart, filename)


def generate_outlier_table(data: pd.Series, filename: str) -> None:
  """実際と正規分布の発生頻度を比較するテーブルを作成する"""
  data = data.dropna()
  mu, std = stats.norm.fit(data)
  n_total = len(data)

  # 条件の定義
  conditions = [
      ("平均 (±0.5σ以内)", mu - 0.5 * std, mu + 0.5 * std),
      ("暴落 (-3σ以下)", -np.inf, mu - 3 * std),
  ]

  rows = []
  for name, low, high in conditions:
    actual_count = int(((data >= low) & (data <= high)).sum())
    actual_pct = actual_count / n_total * 100

    # 理論値（正規分布）
    theo_pct = (stats.norm.cdf(high, mu, std) -
                stats.norm.cdf(low, mu, std)) * 100

    rows.append({
        "条件": name,
        "発生回数": actual_count,
        "実際の発生確率 %": f"{actual_pct:.2f}%",
        "理論的な発生確率 %": f"{theo_pct:.2f}%",
        "割合 (実際/理論)": f"{actual_pct / theo_pct:.2f}x" if theo_pct > 0 else "N/A"
    })

  df = pd.DataFrame(rows)
  path = os.path.join("docs/data/fat_tails", filename)
  df.to_markdown(path, index=False)
  print(f"✅ {path} を作成しました。")


def plot_monthly_fits(data: pd.Series,
                      title: str,
                      filename: str,
                      best_dist_name: str = 'johnsonsu') -> None:
  """月次リターンに対して、正規分布と最適な非対称分布を重ねてプロットする (Altair版)"""
  data = data.dropna()

  # 正規分布のフィット
  mu_norm, std_norm = stats.norm.fit(data)

  # 最適な非対称分布のフィット (平均固定)
  target_dist = [getattr(stats, best_dist_name)]
  best_fit = find_best_distribution_with_fixed_mean(data,
                                                    distributions=target_dist)

  # ヒストグラム用データの作成
  counts, bin_edges = np.histogram(data, bins=50, density=True)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  df_hist = pd.DataFrame({'リターン': bin_centers, '頻度': counts})

  # 曲線データ作成
  x = np.linspace(data.min() - 0.05, data.max() + 0.05, 1000)

  # Normal
  df_norm = pd.DataFrame({
      'リターン': x,
      '頻度': stats.norm.pdf(x, mu_norm, std_norm),
      'モデル': '正規分布'
  })

  # Best Fit
  df_best = pd.DataFrame()
  if best_fit:
    fit = best_fit[0]
    dist_obj = getattr(stats, fit['name'])
    df_best = pd.DataFrame({
        'リターン': x,
        '頻度': dist_obj.pdf(x, *fit['params']),
        'モデル': f'最適フィット ({fit["name"]})'
    })

  df_lines = pd.concat([df_norm, df_best])

  hist = alt.Chart(df_hist).mark_bar(opacity=0.4, color='gray').encode(
      x=alt.X('リターン:Q', title='対数リターン'), y=alt.Y('頻度:Q', title='頻度'))

  lines = alt.Chart(df_lines).mark_line(size=2).encode(
      x='リターン:Q',
      y='頻度:Q',
      color=alt.Color('モデル:N',
                      scale=alt.Scale(
                          domain=['正規分布', f'最適フィット ({best_dist_name})'],
                          range=['red', 'blue']),
                      legend=alt.Legend(orient='top')),
      strokeDash=alt.condition(alt.datum.モデル == '正規分布', alt.value([5, 5]),
                               alt.value([1, 0])))

  chart = (hist + lines).properties(width=600, height=400, title=title)

  save_chart(chart, filename)


def main() -> None:
  # データの取得
  print("データを読み込み中...")
  df = fetch_asset_data()

  # 日次リターンの計算
  daily_returns = process_returns(df, freq='D')

  # 月次リターンの計算
  monthly_returns = process_returns(df, freq='ME')

  # 30年間のデータ欠損チェック (S&P500 月次)
  # 1995-01 から 2025-12 までの 372 ヶ月分が欠損なく揃っているか確認する
  sp500_monthly_30y = monthly_returns.loc[(monthly_returns.index >= START_30Y) &
                                          (monthly_returns.index <= END_30Y),
                                          'SP500_log']
  n_months = len(sp500_monthly_30y)
  n_missing_months = sp500_monthly_30y.isna().sum()

  print(
      f"📊 {START_30Y.strftime('%Y-%m-%d')}〜{END_30Y.strftime('%Y-%m-%d')} の月次データ検証 (S&P500):"
  )
  print(f"  - 合計月数: {n_months} (期待値: 372)")
  print(f"  - 欠損月数: {n_missing_months}")

  if n_months != 372 or n_missing_months > 0:
    print(f"⚠️ 警告: 月次データに不備があります。")
  else:
    print(f"✅ 1995-01 から 2025-12 までの 372 ヶ月分のデータが完全に揃っています。")

  # 1. S&P500 日次分布 (1995-2025)
  sp500_daily_30y = daily_returns.loc[(daily_returns.index >= START_30Y) &
                                      (daily_returns.index <= END_30Y),
                                      'SP500_simple'].dropna()
  plot_dist_with_norm(sp500_daily_30y,
                      f"S&P500 日次 単純収益率 ({START_30Y.year}-{END_30Y.year})",
                      "sp500_daily_dist.svg")
  plot_dist_with_norm(sp500_daily_30y,
                      f"S&P500 日次 単純収益率 ({START_30Y.year}-{END_30Y.year})",
                      "sp500_daily_tail.svg",
                      zoom_tail=True)
  generate_outlier_table(sp500_daily_30y, "daily_outlier_table.md")

  # 2. 日次 vs 月次リターン
  # SP500 月次 全期間
  plot_monthly_fits(monthly_returns['SP500_log'], "S&P500 月次 対数リターン (全期間)",
                    "sp500_monthly_long.svg", 'genlogistic')
  # SP500 月次 30年
  plot_monthly_fits(
      monthly_returns.loc[(monthly_returns.index >= START_30Y) &
                          (monthly_returns.index <= END_30Y),
                          'SP500_log'].dropna(),
      f"S&P500 月次 対数リターン ({START_30Y.year}-{END_30Y.year})",
      "sp500_monthly_30y.svg", 'genlogistic')
  # ACWI 月次
  plot_monthly_fits(monthly_returns['ACWI_log'], "ACWI 月次 対数リターン",
                    "acwi_monthly.svg", 'johnsonsu')


if __name__ == "__main__":
  main()
