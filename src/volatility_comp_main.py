"""
ボラティリティ比較のシミュレーションを実行し、結果のサマリーを出力するスクリプト。

オルカンの期待リターンは固定とし、ボラティリティ（シグマ）のみを変化させた場合の
複数のセットアップを比較します。

出力ファイル:
- `temp/volatility_comp_result.html`: HTML形式の詳細結果
- `docs/imgs/volatility/comp_result.svg`: 資産分布グラフ
- `docs/data/volatility/result.md`: 結果のサマリーテーブル
- `docs/imgs/volatility/paths_0.svg`: ボラ0%の100パスグラフ
- `docs/imgs/volatility/paths_17.svg`: ボラ17%の100パスグラフ
- `docs/imgs/volatility/hist_30y.svg`: 30年後の資産分布ヒストグラム
- `docs/imgs/volatility/hist_years.svg`: ボラ15%の各年ごとの資産分布ヒストグラム
- `docs/data/volatility/prob_10x.md`: 10倍達成確率のテーブル
- `docs/data/volatility/prob_100x.md`: 100倍達成確率のテーブル
- `docs/imgs/volatility/withdrawal_result.svg`: 300万取り崩し時の結果グラフ
- `docs/data/volatility/withdrawal_result.md`: 300万取り崩し時の結果のサマリーテーブル
"""

import os

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.visualize import create_styled_summary, visualize_and_save


def main():
  sigmas = [0, 11, 13, 15, 17]

  # 新エンジンはシミュレーションの月数とパス数をシミュレーション時に指定するため、
  # 変数として定義しておきます。
  N_SIM = 5000
  YEARS = 50
  N_MONTHS = YEARS * 12
  SEED = 42

  # 1. 資産の定義
  assets = [
      Asset(name=f"オルカン v{v}%",
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=v / 100.0),
            trust_fee=0,
            leverage=1) for v in sigmas
  ]

  # 2. 戦略(Plan)の定義
  strategies = [
      Strategy(name=f"ボラ={v}%",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={f"オルカン v{v}%": 1.0},
               annual_cost=0.0,
               inflation_rate=None,
               tax_rate=0.0,
               selling_priority=[f"オルカン v{v}%"]) for v in sigmas
  ]

  # 3. シミュレーションの実行
  print("新エンジン: 月次価格の推移を生成中...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=N_SIM,
                                                       n_months=N_MONTHS,
                                                       seed=SEED)

  results = {}
  print("新エンジン: 各戦略のシミュレーションを実行中...")
  for strategy in strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    results[strategy.name] = res

  # 4. サマリーの出力
  print("\n--- シミュレーション結果 ---")

  img_dir = "docs/imgs/volatility"
  md_dir = "docs/data/volatility"
  os.makedirs(img_dir, exist_ok=True)
  os.makedirs(md_dir, exist_ok=True)

  # 結果の可視化と保存
  html_path = "temp/volatility_comp_result.html"
  visualize_and_save(
      results,
      html_file=html_path,
      distribution_image_file=os.path.join(img_dir, "comp_result.svg"),
      survival_image_file=None,  # このスクリプトでは生存確率グラフは不要の場合
      title="ボラティリティ比較のシミュレーション結果",
      distribution_title="50年後の資産の分布 (ボラティリティ比較)",
      summary_title="最終評価額サマリー (1,000回試行)",
      bankruptcy_years=[],
      open_browser=False)

  formatted_df, _ = create_styled_summary(
      results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[])
  # Markdown形式で表示
  print(
      formatted_df.to_markdown(colalign=("left",) +
                               ("right",) * len(formatted_df.columns)))

  # Markdownとして保存
  md_path = os.path.join(md_dir, "result.md")
  with open(md_path, "w") as f:
    f.write(
        formatted_df.to_markdown(colalign=("left",) +
                                 ("right",) * len(formatted_df.columns)))
  print(f"✅ Markdownデータを {md_path} に保存しました。")

  # --- 新規追加部分 ---

  # 1. ボラティリティによる資産推移グラフ (100パス)
  print("\nパスのグラフを生成中...")
  for v in [0, 17]:
    asset_name = f"オルカン v{v}%"

    # 最初の100パスを取り出す [100パス, n_months+1]
    prices = monthly_asset_prices[asset_name][:100, :]

    # 10,000万円からスタートするので、価格に10,000を掛ける
    asset_values = prices * 10000.0

    plot_data = []
    for path_idx in range(100):
      for month in range(N_MONTHS + 1):
        plot_data.append({
            'Month': month,
            'Year': month / 12,
            'Path': path_idx,
            'Value (億円)': asset_values[path_idx, month] / 10000.0
        })
    df_paths = pd.DataFrame(plot_data)

    # log scale にする場合は: scale=alt.Scale(type='log')
    y_max = 35 if v == 0 else 350
    chart = alt.Chart(df_paths).mark_line(
        opacity=0.4, strokeWidth=2, clip=True).encode(
            x=alt.X('Year:Q', title='経過年数 (年)'),
            y=alt.Y('Value (億円):Q',
                    title='資産額 (億円)',
                    scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color('Path:N', legend=None),
            detail='Path:N',
        ).properties(title=f'{v}% ボラティリティでの資産推移 (100パス)', width=600, height=300)

    img_path = os.path.join(img_dir, f"paths_{v}.svg")
    chart.save(img_path)
    print(f"✅ パスのグラフを {img_path} に保存しました。")

  # 2. 30年後の資産分布ヒストグラム
  print("\n30年後の資産分布ヒストグラムを生成中...")
  plot_data = []
  month_30y = 30 * 12
  x_eval_30y = np.linspace(0, 30, 200)

  for v in sigmas:
    if v == 0:
      continue
    strategy_name = f"ボラ={v}%"
    asset_name = f"オルカン v{v}%"

    # 取り崩しなしのため、単純に初期資産 × 価格倍率
    values_30y = monthly_asset_prices[asset_name][:, month_30y] * 10000.0
    values_30y_okuen = values_30y / 10000.0

    kde = gaussian_kde(values_30y_okuen)
    y_eval = kde(x_eval_30y)
    for x, y_val in zip(x_eval_30y, y_eval):
      plot_data.append({
          'Strategy': strategy_name,
          'Value (億円)': x,
          'Density': y_val
      })

  df_hist_30y = pd.DataFrame(plot_data)
  chart_hist_30y = alt.Chart(df_hist_30y).mark_line(clip=True).encode(
      x=alt.X('Value (億円):Q', title='資産額 (億円)',
              scale=alt.Scale(domain=[0, 30])),
      y=alt.Y('Density:Q', title='頻度'),
      color='Strategy:N').properties(title='30年後の資産分布', width=600, height=300)
  hist_30y_path = os.path.join(img_dir, "hist_30y.svg")
  chart_hist_30y.save(hist_30y_path)
  print(f"✅ ヒストグラムを {hist_30y_path} に保存しました。")

  # 3. 15%ボラティリティの各年の資産分布ヒストグラム
  print("\n15%ボラティリティの各年資産分布ヒストグラムを生成中...")
  plot_data = []
  target_asset = "オルカン v15%"
  years_to_plot = [10, 20, 30, 40, 50]
  x_eval_years = np.linspace(0, 60, 200)

  for y in years_to_plot:
    m = y * 12
    values_y = monthly_asset_prices[target_asset][:, m] * 10000.0
    values_y_okuen = values_y / 10000.0

    kde = gaussian_kde(values_y_okuen)
    y_eval = kde(x_eval_years)
    for x, y_val in zip(x_eval_years, y_eval):
      plot_data.append({'Year': f"{y}年後", 'Value (億円)': x, 'Density': y_val})

  df_hist_years = pd.DataFrame(plot_data)
  chart_hist_years = alt.Chart(df_hist_years).mark_line(clip=True).encode(
      x=alt.X('Value (億円):Q', title='資産額 (億円)',
              scale=alt.Scale(domain=[0, 60])),
      y=alt.Y('Density:Q', title='頻度'),
      color='Year:N').properties(title='15% ボラティリティでの資産分布の推移',
                                 width=600,
                                 height=300)
  hist_years_path = os.path.join(img_dir, "hist_years.svg")
  chart_hist_years.save(hist_years_path)
  print(f"✅ ヒストグラムを {hist_years_path} に保存しました。")

  # 4. 10倍、100倍になる確率テーブル
  print("\n確率テーブルを生成中...")
  prob_10x_data = []
  prob_100x_data = []

  # 初期資産 10,000万円
  # 10倍 = 100,000万円
  # 100倍 = 1,000,000万円
  target_10x = 100000.0
  target_100x = 1000000.0

  for strategy_name, res in results.items():
    final_values = res.net_values

    prob_10x = np.mean(final_values >= target_10x) * 100.0
    prob_100x = np.mean(final_values >= target_100x) * 100.0

    prob_10x_data.append({
        '戦略': strategy_name,
        '10倍(10億円)達成確率': f"{prob_10x:.1f}%"
    })

    prob_100x_data.append({
        '戦略': strategy_name,
        '100倍(100億円)達成確率': f"{prob_100x:.1f}%"
    })

  df_prob_10x = pd.DataFrame(prob_10x_data)
  df_prob_100x = pd.DataFrame(prob_100x_data)

  prob_10x_path = os.path.join(md_dir, "prob_10x.md")
  with open(prob_10x_path, "w") as f:
    f.write(df_prob_10x.to_markdown(index=False))
  print(f"✅ 10倍達成確率を {prob_10x_path} に保存しました。")

  prob_100x_path = os.path.join(md_dir, "prob_100x.md")
  with open(prob_100x_path, "w") as f:
    f.write(df_prob_100x.to_markdown(index=False))
  print(f"✅ 100倍達成確率を {prob_100x_path} に保存しました。")

  # 5. 年間300万取り崩しシミュレーション
  print("\n年間300万円取り崩しシミュレーションを実行中...")
  withdrawal_strategies = [
      Strategy(name=f"ボラ={v}% (取崩)",
               initial_money=10000,
               initial_loan=0,
               yearly_loan_interest=2.125 / 100,
               initial_asset_ratio={f"オルカン v{v}%": 1.0},
               annual_cost=300.0,
               inflation_rate=None,
               tax_rate=0.0,
               selling_priority=[f"オルカン v{v}%"]) for v in sigmas
  ]

  withdrawal_results = {}
  for strategy in withdrawal_strategies:
    res = simulate_strategy(strategy, monthly_asset_prices)
    withdrawal_results[strategy.name] = res

  withdrawal_img_path = os.path.join(img_dir, "withdrawal_result.svg")
  withdrawal_html_path = "temp/volatility_withdrawal_result.html"
  visualize_and_save(withdrawal_results,
                     html_file=withdrawal_html_path,
                     distribution_image_file=withdrawal_img_path,
                     survival_image_file=None,
                     title="ボラティリティ比較 (年間300万取り崩し)",
                     distribution_title="50年後の資産の分布 (300万/年 取崩)",
                     summary_title="最終評価額サマリー (年間300万取り崩し)",
                     bankruptcy_years=[],
                     open_browser=False)
  print(f"✅ 取り崩しシミュレーションのグラフを {withdrawal_img_path} に保存しました。")

  # 取り崩しシナリオのサマリーテーブル
  withdrawal_df, _ = create_styled_summary(
      withdrawal_results,
      quantiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90],
      bankruptcy_years=[])

  withdrawal_md_path = os.path.join(md_dir, "withdrawal_result.md")
  with open(withdrawal_md_path, "w") as f:
    f.write(
        withdrawal_df.to_markdown(colalign=("left",) +
                                  ("right",) * len(withdrawal_df.columns)))
  print(f"✅ 取り崩しシミュレーションのMarkdownデータを {withdrawal_md_path} に保存しました。")

  print(f"\nRun this to see results: open {html_path}")
  print(f"Run this to see withdrawal results: open {withdrawal_html_path}")


if __name__ == "__main__":
  main()
