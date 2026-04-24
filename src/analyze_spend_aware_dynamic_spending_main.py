"""
Spend-Aware Dynamic Spending シミュレーション結果の分析・可視化スクリプト。

引数 --exp_name によって異なる実験結果を可視化します。
Altair を使用して SVG グラフを生成します。
"""

import argparse
import os

import altair as alt
import pandas as pd

from src.lib.visualize_all_yr import create_spend_percentile_chart


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name",
                      type=str,
                      default="v1_v2_comp",
                      choices=["simple", "v1_v2_comp"])
  args = parser.parse_args()

  data_dir = "data/spend_aware_dynamic_spending"
  summary_path = os.path.join(data_dir, f"{args.exp_name}_summary.csv")
  survival_path = os.path.join(data_dir, f"{args.exp_name}_survival.csv")
  spends_path = os.path.join(data_dir, f"{args.exp_name}_spends.csv")
  output_dir = "docs/imgs/spend_aware_dynamic_spending"
  os.makedirs(output_dir, exist_ok=True)

  if not os.path.exists(summary_path) or not os.path.exists(spends_path):
    print(
        f"Error: Data files not found. Run simulation first with --exp_name={args.exp_name}"
    )
    return

  df_summary = pd.read_csv(summary_path)
  df_survival = pd.read_csv(survival_path)
  df_spends = pd.read_csv(spends_path)

  # 1. 生存確率の推移比較 (Survival Probability Time-series)
  # 3.0% 〜 4.5% の範囲に限定
  df_surv_plot = df_survival[df_survival["rule"].isin([3.0, 3.5, 4.0,
                                                       4.5])].copy()

  # 日本語ラベルへの変換 (Altairでの改行用にセパレータ '@' を使用)
  strat_map = {
      "DRv2_DSv1": "支出率を目標にする@ダイナミックスペンディング",
      "DRv2_DSv2": "生存確率を目標にする@ダイナミックスペンディング",
      "FixedSpend": "定額取り崩し"
  }
  df_surv_plot["strategy_jp"] = df_surv_plot["strategy"].map(strat_map)
  df_surv_plot["rule_label"] = df_surv_plot["rule"].apply(
      lambda x: f"初期支出 {x}%")
  df_surv_plot[
      "Survival Probability (%)"] = df_surv_plot["survival_rate"] * 100.0

  min_val = df_surv_plot['Survival Probability (%)'].min()
  y_min = max(50.0, (min_val // 10) * 10)
  y_max = 100.0

  display_survival_title = '経過年数と生存確率の推移'
  if y_min > 0:
    display_survival_title += f"（生存確率 {y_min:.0f}%以下は描画を省略）"

  # strokeDash の設定
  # 指定されたラベルに対して点線を割り当てる
  chart_surv = alt.Chart(df_surv_plot).mark_line().encode(
      x=alt.X('year:Q', title='経過年数 (年)'),
      y=alt.Y('Survival Probability (%):Q',
              title='生存確率 (%)',
              scale=alt.Scale(domain=[y_min, y_max])),
      color=alt.Color('rule_label:N', title='初期支出ルール'),
      strokeDash=alt.StrokeDash(
          'strategy_jp:N',
          title='戦略',
          legend=alt.Legend(labelExpr="split(datum.label, '@')"),
          scale=alt.Scale(
              domain=["支出率を目標にする@ダイナミックスペンディング", "生存確率を目標にする@ダイナミックスペンディング"],
              range=[[4, 4], [0, 0]])),
      tooltip=[
          'year', 'rule', 'strategy_jp',
          alt.Tooltip('Survival Probability (%):Q', format='.1f')
      ]).properties(title=display_survival_title, width=600,
                    height=400).interactive()

  surv_chart_path = os.path.join(output_dir,
                                 f"{args.exp_name}_survival_comparison.svg")
  chart_surv.save(surv_chart_path)
  print(f"✅ {surv_chart_path} に保存しました。")

  # 2. 各ルールごとの実質支出額の推移
  rules = df_summary["rule"].unique()
  for rule in rules:
    df_rule_spends = df_spends[df_spends["rule"] == rule]

    pivot_data = []
    for strategy in df_rule_spends["strategy"].unique():
      df_strat = df_rule_spends[df_rule_spends["strategy"] == strategy]
      for vtype in ["p25", "p50", "p75"]:
        row = {
            "strategy": strat_map.get(strategy, strategy),
        }
        if vtype == "p25":
          row["value_type"] = "spend25p"
        elif vtype == "p50":
          row["value_type"] = "spend50p"
        elif vtype == "p75":
          row["value_type"] = "spend75p"

        for _, r_data in df_strat.iterrows():
          row[str(int(r_data["year"]))] = r_data[vtype]
        pivot_data.append(row)

    df_visualize = pd.DataFrame(pivot_data)

    file_name = f"{args.exp_name}_real_spend_comparison_{rule}p.svg"
    chart_path = os.path.join(output_dir, file_name)

    create_spend_percentile_chart(
        df_visualize,
        title=f'実質支出額の推移比較 (初期支出率: {rule}%, 共通生存パスのみ)',
        output_path=chart_path,
        start_age=40,
        num_years=55)


if __name__ == "__main__":
  main()
