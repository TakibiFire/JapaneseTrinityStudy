"""
収益率配列のリスク（Sequence of Returns Risk）を分析し、可視化するスクリプト。

このスクリプトは、期待リターンが同じでも収益の順序によって最終的な結果（特に取り崩し時）が
どのように変わるかを分析します。

主な分析内容:
1. 取り崩しなしで中央値付近（約18.3億円）に終わるパスを抽出
2. それらのパスに対して年間400万円の取り崩しを適用した場合の破産状況
3. 破産時期別の資産インデックス推移の深掘り

出力ファイル:
- `docs/imgs/sequence/median_paths_linear.svg`: 中央値付近のパス（線形）
- `docs/imgs/sequence/median_paths_log.svg`: 中央値付近のパス（対数）
- `docs/imgs/sequence/bankrupt_paths_withdrawal.svg`: 中央値パスのうち破産した人の資産推移
- `docs/imgs/sequence/bankrupt_paths_asset_log.svg`: 中央値パスのうち破産した人の元のアセット推移
- `docs/imgs/sequence/bankrupt_paths_asset_percentile.svg`: 中央値パスのうち破産した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/success_paths_asset_percentile.svg`: 中央値パスのうち成功した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/survival_7_15_400.svg`: 生存確率の推移
- `docs/imgs/sequence/raw_index_bankrupt_0_10.svg`: 0-10年で破産した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/raw_index_bankrupt_11_20.svg`: 11-20年で破産した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/raw_index_bankrupt_21_30.svg`: 21-30年で破産した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/raw_index_bankrupt_31_40.svg`: 31-40年で破産した人のアセット推移 (パーセンタイル)
- `docs/imgs/sequence/raw_index_bankrupt_41_50.svg`: 41-50年で破産した人のアセット推移 (パーセンタイル)
"""

import os

import altair as alt
import numpy as np
import pandas as pd

from src.core import Strategy, simulate_strategy
from src.lib.asset_generator import (Asset, YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowRule,
                                        CashflowType, generate_cashflows)
from src.lib.visualize import create_survival_probability_chart


def save_percentile_chart(prices_subset, title, filename, n_months):
  """
  指定された価格推移サブセットから10%, 50%, 90%パーセンタイルを計算し、グラフを保存する。
  """
  if len(prices_subset) == 0:
    return

  percentiles = [10, 50, 90]
  p_values = np.percentile(prices_subset, percentiles, axis=0)

  plot_data = []
  for i, p in enumerate(percentiles):
    for m in range(n_months + 1):
      plot_data.append({
          'Month': m,
          'Year': m / 12.0,
          'Percentile': f'{p}%',
          'Index': p_values[i, m]
      })

  df = pd.DataFrame(plot_data)
  chart = alt.Chart(df).mark_line().encode(
      x=alt.X('Year:Q', title='経過年数 (年)'),
      y=alt.Y('Index:Q',
              title='アセットインデックス',
              scale=alt.Scale(type='log', domainMin=0.2)),
      color=alt.Color('Percentile:N',
                      sort=['10%', '50%', '90%'],
                      title='パーセンタイル')).properties(title=title,
                                                  width=600,
                                                  height=400)
  chart.save(filename)


def main():
  # 基本設定
  N_SIM = 5000
  YEARS = 50
  N_MONTHS = YEARS * 12
  SEED = 42
  INITIAL_MONEY = 10000.0  # 1億円 (単位: 万円)
  ANNUAL_WITHDRAWAL = 400.0  # 400万円

  IMG_DIR = "docs/imgs/sequence/"
  DATA_DIR = "docs/data/sequence/"
  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(DATA_DIR, exist_ok=True)

  # 1. 資産の定義
  # 7% リターン, 15% ボラティリティ
  asset_name = "オルカン_7_15"
  assets = [
      Asset(name=asset_name,
            dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
            trust_fee=0,
            leverage=1)
  ]

  # 2. 月次価格推移の生成
  print(f"価格推移を生成中 (N_SIM={N_SIM}, YEARS={YEARS})...")
  monthly_asset_prices = generate_monthly_asset_prices(assets,
                                                       n_paths=N_SIM,
                                                       n_months=N_MONTHS,
                                                       seed=SEED)
  raw_prices = monthly_asset_prices[asset_name]  # (N_SIM, N_MONTHS + 1)

  # ---------------------------------------------------------------------------
  # Section 1 - 中央値付近で終わるパスの抽出
  # ---------------------------------------------------------------------------
  print("Section 1: 中央値付近のパスを抽出中...")

  # 取り崩しなしのシミュレーション
  strategy_no_withdrawal = Strategy(name="No Withdrawal",
                                    initial_money=INITIAL_MONEY,
                                    initial_loan=0,
                                    yearly_loan_interest=0.0,
                                    initial_asset_ratio={asset_name: 1.0},
                                    cashflow_rules=[],
                                    tax_rate=0.0,
                                    selling_priority=[asset_name])
  res_no_withdrawal = simulate_strategy(strategy_no_withdrawal,
                                        monthly_asset_prices)

  # 最終資産額が 15.0B 〜 21.6B (150,000 〜 216,000 万円) のパスを抽出
  # 中央値は約 18.3B
  final_values = res_no_withdrawal.net_values
  mask_median = (final_values >= 150000) & (final_values <= 216000)
  median_indices = np.where(mask_median)[0]

  # 再現性のためにソートして最大50個選択
  selected_median_indices = np.sort(median_indices)[:50]
  print(f"抽出された中央値パス数: {len(median_indices)}中 {len(selected_median_indices)}")

  # 抽出したパスの資産推移を可視化 (linear & log)
  median_paths_prices = raw_prices[selected_median_indices, :]
  median_paths_values = median_paths_prices * INITIAL_MONEY

  plot_data = []
  for i, path_idx in enumerate(selected_median_indices):
    for m in range(N_MONTHS + 1):
      plot_data.append({
          'Month': m,
          'Year': m / 12.0,
          'PathID': int(path_idx),
          'Value_OKUEN': median_paths_values[i, m] / 10000.0
      })
  df_median_paths = pd.DataFrame(plot_data)

  base_chart = alt.Chart(df_median_paths).mark_line(
      opacity=0.5,
      strokeWidth=1).encode(x=alt.X('Year:Q', title='経過年数 (年)'),
                            color=alt.Color('PathID:N', legend=None),
                            detail='PathID:N').properties(width=600, height=400)

  # linear y-axis
  chart_linear = base_chart.encode(
      y=alt.Y('Value_OKUEN:Q', title='資産額 (億円)')).properties(
          title='中央値付近で終わるパスの推移 (線形スケール)')
  chart_linear.save(os.path.join(IMG_DIR, "median_paths_linear.svg"))

  # log y-axis
  chart_log = base_chart.encode(
      y=alt.Y('Value_OKUEN:Q', title='資産額 (億円)', scale=alt.Scale(
          type='log'))).properties(title='中央値付近で終わるパスの推移 (対数スケール)')
  chart_log.save(os.path.join(IMG_DIR, "median_paths_log.svg"))

  # これらのパスに対して取り崩しありのシミュレーションを実行
  print("中央値パスに対して取り崩しシミュレーションを実行中...")

  # 1. キャッシュフロールールの定義
  spend_config = BaseSpendConfig(
      name="生活費",
      amount=ANNUAL_WITHDRAWAL,
      cpi_name=None
  )
  cashflow_rules = [
      CashflowRule(source_name=spend_config.name,
                   cashflow_type=CashflowType.REGULAR)
  ]
  monthly_cashflows = generate_cashflows(
      [spend_config], monthly_asset_prices, N_SIM, YEARS * 12)

  # 全体に対して一度実行し、後でインデックスで抽出する
  strategy_withdrawal = Strategy(name="4M Withdrawal",
                                 initial_money=INITIAL_MONEY,
                                 initial_loan=0,
                                 yearly_loan_interest=0.0,
                                 initial_asset_ratio={asset_name: 1.0},
                                 cashflow_rules=cashflow_rules,
                                 tax_rate=0.0,
                                 selling_priority=[asset_name])
  # core.py の simulate_strategy は全パスに対して実行される
  # 特定のパスだけ実行する機能はないため、全体を回す（N_SIM=5000 なので高速）
  res_withdrawal_all = simulate_strategy(strategy_withdrawal,
                                         monthly_asset_prices,
                                         monthly_cashflows=monthly_cashflows)

  # 中央値パスのうち、破産した人を特定
  # 破産 = sustained_months < N_MONTHS
  bankrupt_mask_in_median = res_withdrawal_all.sustained_months[
      median_indices] < N_MONTHS
  bankrupt_indices_in_median = median_indices[bankrupt_mask_in_median]
  success_indices_in_median = median_indices[~bankrupt_mask_in_median]
  print(f"中央値パス内の破産者数: {len(bankrupt_indices_in_median)}")
  print(f"中央値パス内の成功者数: {len(success_indices_in_median)}")

  # 破産した人の資産推移 (取り崩しあり) をプロット
  # 注意: simulate_strategy は各月の資産額を返さないため、手動で再計算するか、
  # あるいは簡易的に、破産月までの推移を raw_prices * INITIAL_MONEY - (累計支出) で近似するか。
  # 本来は core.py を修正して各月の推移を返せるようにすべきだが、ここでは
  # 破産した人の資産推移を可視化するために、再度そのパスだけシミュレーション的な計算を行う

  def get_asset_paths_with_withdrawal(indices, prices, initial_money,
                                      annual_withdrawal):
    n = len(indices)
    m_plus_1 = prices.shape[1]
    # (n, m_plus_1) の配列を作成
    paths = np.zeros((n, m_plus_1))
    for i, idx in enumerate(indices):
      current_val = initial_money
      paths[i, 0] = current_val
      for m in range(m_plus_1 - 1):
        # 月初のリターン適用
        ret = prices[idx, m + 1] / prices[idx, m]
        current_val *= ret
        # 月末の取り崩し
        current_val -= annual_withdrawal / 12.0
        if current_val < 0:
          current_val = 0
        paths[i, m + 1] = current_val
    return paths

  bankrupt_paths_val = get_asset_paths_with_withdrawal(
      bankrupt_indices_in_median, raw_prices, INITIAL_MONEY, ANNUAL_WITHDRAWAL)

  plot_data_bankrupt = []
  for i, path_idx in enumerate(bankrupt_indices_in_median):
    for m in range(N_MONTHS + 1):
      val = bankrupt_paths_val[i, m]
      if val <= 0 and m > 0 and bankrupt_paths_val[i, m - 1] <= 0:
        continue  # 破産後は描画しない
      plot_data_bankrupt.append({
          'Month': m,
          'Year': m / 12.0,
          'PathID': int(path_idx),
          'Value_OKUEN': val / 10000.0
      })

  if plot_data_bankrupt:
    df_bankrupt = pd.DataFrame(plot_data_bankrupt)
    chart_bankrupt_withdrawal = alt.Chart(df_bankrupt).mark_line(
        opacity=0.4, strokeWidth=1).encode(x=alt.X('Year:Q', title='経過年数 (年)'),
                            y=alt.Y('Value_OKUEN:Q', title='資産額 (億円)'),
                            color=alt.Color('PathID:N', legend=None),
                            detail='PathID:N').properties(
                                title='中央値パスで破産した人の資産推移 (400万取崩, 対数スケール)',
                                width=600,
                                height=400)
    chart_bankrupt_withdrawal.save(
        os.path.join(IMG_DIR, "bankrupt_paths_withdrawal.svg"))

  # 破産した人の「生のアセットインデックス」をプロット
  plot_data_raw_bankrupt = []
  for i, path_idx in enumerate(bankrupt_indices_in_median):
    for m in range(N_MONTHS + 1):
      plot_data_raw_bankrupt.append({
          'Month': m,
          'Year': m / 12.0,
          'PathID': int(path_idx),
          'Index': raw_prices[path_idx, m]
      })

  if plot_data_raw_bankrupt:
    df_raw_bankrupt = pd.DataFrame(plot_data_raw_bankrupt)
    chart_raw_bankrupt = alt.Chart(df_raw_bankrupt).mark_line(
        opacity=0.4, strokeWidth=1).encode(x=alt.X('Year:Q', title='経過年数 (年)'),
                            y=alt.Y('Index:Q',
                                    title='アセットインデックス',
                                    scale=alt.Scale(type='log')),
                            color=alt.Color('PathID:N', legend=None),
                            detail='PathID:N').properties(
                                title='中央値パスで破産した人の元のアセット推移',
                                width=600,
                                height=400)
    chart_raw_bankrupt.save(
        os.path.join(IMG_DIR, "bankrupt_paths_asset_log.svg"))

  # 破産した人と成功した人のパーセンタイル推移を追加プロット
  save_percentile_chart(
      prices_subset=raw_prices[bankrupt_indices_in_median],
      title='中央値パスで破産した人のアセットパーセンタイル推移',
      filename=os.path.join(IMG_DIR, "bankrupt_paths_asset_percentile.svg"),
      n_months=N_MONTHS
  )

  save_percentile_chart(
      prices_subset=raw_prices[success_indices_in_median],
      title='中央値パスで成功した人のアセットパーセンタイル推移',
      filename=os.path.join(IMG_DIR, "success_paths_asset_percentile.svg"),
      n_months=N_MONTHS
  )

  # ---------------------------------------------------------------------------
  # Section 2 - 破産タイミングの深掘り
  # ---------------------------------------------------------------------------
  print("Section 2: 破産タイミングの分析中...")

  # 生存確率グラフの再描画
  results_for_survival = {"7% 15% 4M": res_withdrawal_all}
  _, survival_chart = create_survival_probability_chart(results_for_survival,
                                                        max_years=YEARS)
  survival_chart.save(os.path.join(IMG_DIR, "survival_7_15_400.svg"))

  # 全5000シミュレーションにおける破産者を特定
  all_sustained_months = res_withdrawal_all.sustained_months
  all_bankrupt_mask = all_sustained_months < N_MONTHS
  all_bankrupt_indices = np.where(all_bankrupt_mask)[0]

  # 年代ごとにグループ化
  # 0-10y, 11-20y, 21-30y, 31-40y, 41-50y
  decades = [(0, 10, "0_10"), (11, 20, "11_20"), (21, 30, "21_30"),
             (31, 40, "31_40"), (41, 50, "41_50")]

  for start_y, end_y, suffix in decades:
    # 破産した月が [start_y*12, end_y*12) の範囲にある人
    mask = (all_sustained_months[all_bankrupt_indices] >= (start_y-1 if start_y > 0 else 0) * 12) & \
           (all_sustained_months[all_bankrupt_indices] < end_y * 12)

    # 修正: 0-10y は 0 <= months < 120, 11-20y は 120 <= months < 240 ...
    start_m = (start_y - 1 if start_y > 0 else 0) * 12
    if start_y == 0:
      start_m = 0
    else:
      start_m = (start_y - 1) * 12 + 12  # つまり start_y * 12 だけど、境界を明確にする

    # シンプルに:
    mask = (all_sustained_months[all_bankrupt_indices]
            >= (start_y - (1 if start_y > 0 else 0)) * 12)  # 以前のロジックが怪しいので書き直し

    # 正確な年代判定
    m_start = (start_y - 10 if start_y > 0 else 0) * 12  # 0, 120, 240...
    if start_y == 0:
      m_start = 0
      m_end = 10 * 12
    else:
      m_start = (start_y - 1) * 12
      m_end = end_y * 12

    mask = (all_sustained_months[all_bankrupt_indices] >= m_start) & \
           (all_sustained_months[all_bankrupt_indices] < m_end)

    group_indices = all_bankrupt_indices[mask]

    print(f"年代 {start_y}-{end_y}y の破産者数 (サンプル): {len(group_indices)}")

    if len(group_indices) > 0:
      save_percentile_chart(
          prices_subset=raw_prices[group_indices],
          title=f'{start_y}-{end_y}年に破産した人のアセット推移',
          filename=os.path.join(IMG_DIR, f"raw_index_bankrupt_{suffix}.svg"),
          n_months=N_MONTHS
      )
    else:
      # 空のファイルを作らないか、あるいは空の旨を記したグラフを作る
      print(f"Warning: No bankruptcies in {start_y}-{end_y}y range.")

  print("\n分析完了。")
  print(f"画像出力先: {IMG_DIR}")


if __name__ == "__main__":
  main()
