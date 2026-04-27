"""
analyze_sp500_acwi_main.py

S&P 500 と ACWI (オール・カントリー・ワールド・インデックス) の歴史的な年次算術リターンと
年次算術ボラティリティを計算し、Altairを用いて可視化するスクリプト。

計算ロジック:
1. 各年における月次リターンデータを取得。
2. 年率算術リターン = (月次リターンの平均) * 12
3. 年率算術ボラティリティ = (月次リターンの標準偏差) * sqrt(12)
"""

import os
from typing import Any, cast

import altair as alt
import numpy as np
import pandas as pd

from src.lib import asset_model


def main() -> None:
  # 1. データの読み込み
  df = pd.read_csv('data/asset_daily_prices.csv')
  monthly_returns = asset_model.process_returns(df, 'ME')

  # 月次リターン(単利)の抽出
  rets_df = monthly_returns[['SP500_simple', 'ACWI_simple']].dropna(how='all')
  # Mypy対応: DatetimeIndexのyear属性を使用
  rets_df['Year'] = pd.to_datetime(rets_df.index).year

  results = []

  # 各年・各アセットごとに計算
  melted = rets_df.melt(id_vars='Year', var_name='Asset',
                        value_name='Return').dropna()
  # groupbyの結果を明示的に扱う
  for key, group in melted.groupby(['Year', 'Asset']):
    # key は (year, asset) のタプル
    year_val = key[0]
    asset_val = key[1]

    # 計算精度を保つため、ある程度のデータ点が必要。
    # 2008年はACWIが9ヶ月分のデータがあるため、9個以上を条件とする。
    if len(group) < 9:
      continue

    returns = group['Return']
    ann_return = float(returns.mean()) * 12
    ann_volatility = float(returns.std()) * np.sqrt(12)

    # asset_valはGroupByのキーなのでHashable。文字列に変換して比較する。
    asset_str = str(asset_val)
    asset_name = 'S&P 500' if 'SP500' in asset_str else 'ACWI (Orukan)'

    results.append({
        'Year': int(cast(Any, year_val)),
        'Asset': asset_name,
        'Annual Arithmetic Return': ann_return,
        'Annual Arithmetic Volatility': ann_volatility
    })

  results_df = pd.DataFrame(results)

  # 期間のフィルタリング
  # S&P 500: 直近30年 (1996 - 2025)
  # ACWI: 直近18年 (2008 - 2025)
  sp500_30y = results_df[(results_df['Asset'] == 'S&P 500') &
                         (results_df['Year'] >= 1996) &
                         (results_df['Year'] <= 2025)]
  acwi_18y = results_df[(results_df['Asset'] == 'ACWI (Orukan)') &
                        (results_df['Year'] >= 2008) &
                        (results_df['Year'] <= 2025)]

  plot_df = pd.concat([sp500_30y, acwi_18y])

  # 2. Altairによる可視化

  # 色の設定
  colors = alt.Scale(domain=['S&P 500', 'ACWI (Orukan)'],
                     range=['#1f77b4', '#ff7f0e'])

  # 共通のベースチャート
  base = alt.Chart(plot_df).encode(
      x=alt.X('Year:O', title='年'),
      color=alt.Color('Asset:N',
                      scale=colors,
                      title='アセット',
                      legend=alt.Legend(orient='top',
                                        direction='horizontal',
                                        title=None))).properties(width=600,
                                                                 height=300)

  # 1st Graph: 年率算術リターン
  return_chart = base.mark_line(point=True).encode(
      y=alt.Y('Annual Arithmetic Return:Q',
              title='年率算術リターン',
              axis=alt.Axis(format='%')),
      tooltip=[
          'Year', 'Asset',
          alt.Tooltip('Annual Arithmetic Return:Q', format='.2%')
      ]).properties(title='年率算術リターンの推移')

  # 2nd Graph: 年率算術ボラティリティ
  vol_chart = base.mark_line(point=True).encode(
      y=alt.Y('Annual Arithmetic Volatility:Q',
              title='年率算術ボラティリティ',
              axis=alt.Axis(format='%')),
      tooltip=[
          'Year', 'Asset',
          alt.Tooltip('Annual Arithmetic Volatility:Q', format='.2%')
      ]).properties(title='年率算術ボラティリティの推移')

  # 3. 保存
  img_dir = "docs/imgs/sp500_acwi"
  os.makedirs(img_dir, exist_ok=True)

  charts = [(return_chart, 'sp500_acwi_annual_return.svg'),
            (vol_chart, 'sp500_acwi_annual_volatility.svg')]

  for chart, filename in charts:
    output_path = os.path.join(img_dir, filename)
    try:
      chart.save(output_path)
      print(f"✅ 可視化結果を {output_path} に保存しました。")
    except Exception as e:
      print(f"❌ {filename} の保存に失敗しました: {e}")
      html_path = output_path.replace('.svg', '.html')
      chart.save(html_path)
      print(f"ℹ️ 代わりに {html_path} に保存しました。")

  # 4. 統計量の表示

  # 年ごとの値を平均したもの
  print("\n--- 平均統計量 (年ごとの指標の単純平均) ---")
  if not plot_df.empty:
    print(
        plot_df.groupby('Asset')[[
            'Annual Arithmetic Return', 'Annual Arithmetic Volatility'
        ]].mean())

  # 全期間のデータから直接算出
  print("\n--- 全期間データから直接算出 ---")

  # 期間の定義
  sp500_30y_rets = rets_df['SP500_simple'][(rets_df.index >= '1996-01-01') & (
      rets_df.index <= '2025-12-31')].dropna()
  sp500_overlap_rets = rets_df['SP500_simple'][
      (rets_df.index >= '2008-03-01') &
      (rets_df.index <= '2025-12-31')].dropna()
  acwi_overlap_rets = rets_df['ACWI_simple'][(rets_df.index >= '2008-03-01') & (
      rets_df.index <= '2025-12-31')].dropna()

  stats_cases = [("S&P 500 (30y: 1996-2025)", sp500_30y_rets),
                 ("S&P 500 (Overlap: 2008-03~)", sp500_overlap_rets),
                 ("ACWI (Overlap: 2008-03~)", acwi_overlap_rets)]

  overall_results = []
  for name, rets in stats_cases:
    ann_return = float(rets.mean()) * 12
    ann_volatility = float(rets.std()) * np.sqrt(12)
    overall_results.append({
        'Case': name,
        'Annual Arithmetic Return': ann_return,
        'Annual Arithmetic Volatility': ann_volatility
    })

  overall_df = pd.DataFrame(overall_results).set_index('Case')
  print(overall_df)


if __name__ == "__main__":
  main()
