"""
data/dynamic_spending_grid_comp.csv の結果を可視化し、数式フィッティングを行うスクリプト。
指定された年数（30年、50年など）とダイナミックリバランスの有無に応じた生存確率を
ヒートマップとして表示し、さらにダイナミックリバランスなしの場合の数式近似を出力する。
"""

import os

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fitting_model(X, a, b, c, d, e, f, g):
  """
  フィットさせる数式モデル。
  L: lower_limit, U: upper_limit
  """
  L, U = X
  term1 = a * np.power(np.abs(L - b), c)
  term2 = d * np.power(np.abs(U - e), f)
  return term1 + term2 + g


def run_fitting(df: pd.DataFrame, target_years: list[str]):
  """
  ダイナミックリバランスなしのデータに対して数式フィッティングを実行し、結果を表示する。
  """
  df_fit = df[df["is_dynamic_rebalance"] == 0]
  L = df_fit["lower_limit"].values
  U = df_fit["upper_limit"].values

  print("\n" + "=" * 50)
  print("数式フィッティング結果 (ダイナミックリバランスなし)")
  print("=" * 50)

  for year in target_years:
    y = df_fit[year].values
    # 初期推測値
    p0 = [-5.0, -0.05, 1.0, -2.0, -0.05, 1.0, 1.0]
    # 探索範囲
    bounds = ([-100, -1, 0.1, -100, -1, 0.1, -10], [0, 1, 5, 0, 1, 5, 10])

    try:
      popt, _ = curve_fit(fitting_model, (L, U), y, p0=p0, bounds=bounds, maxfev=50000)
      a, b, c, d, e, f, g = popt
      y_pred = fitting_model((L, U), *popt)
      r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

      print(f"\n[{year}年生存確率]")
      print(f"  rate = {a:.4f} * (L - {b:.4f})^{c:.4f} + {d:.4f} * (U - {e:.4f})^{f:.4f} + {g:.4f}")
      print(f"  R-squared: {r2:.6f}")
    except Exception as err:
      print(f"  Error for {year}yr fitting: {err}")


def print_representative_stats(df: pd.DataFrame):
  """
  代表的な戦略の生存確率を表示する。
  """
  print("\n" + "=" * 50)
  print("代表的な戦略の生存確率")
  print("=" * 50)

  # 実験1 (ダイナミックリバランスなし)
  print("\n[実験1: ダイナミックリバランスなし]")
  conditions = [
      (0.02, 0.02, "2%定額支出アップ"),
      (0.03, 0.00, "上限3%, 下限0%"),
      (0.05, -0.015, "上限5%, 下限-1.5%"),
      (0.02, -0.02, "上限2%, 下限-2%")
  ]

  exp1_data = []
  for up, low, label in conditions:
    res = df[(df["is_dynamic_rebalance"] == 0) &
             (np.isclose(df["upper_limit"], up)) &
             (np.isclose(df["lower_limit"], low))]
    if not res.empty:
      s30 = res["30"].values[0]
      s50 = res["50"].values[0]
      exp1_data.append({"ラベル": label, "30年": f"{s30:.1%}", "50年": f"{s50:.1%}"})
  print(pd.DataFrame(exp1_data))

  # 実験2 (ダイナミックリバランスあり)
  print("\n[実験2: ダイナミックリバランスあり (50年)]")
  exp2_conditions = [
      (0.03, 0.00, "上限3%, 下限0%"),
      (0.05, -0.015, "上限5%, 下限-1.5%"),
      (0.02, -0.02, "上限2%, 下限-2%")
  ]
  exp2_data = []
  for up, low, label in exp2_conditions:
    res = df[(df["is_dynamic_rebalance"] == 1) &
             (np.isclose(df["upper_limit"], up)) &
             (np.isclose(df["lower_limit"], low))]
    if not res.empty:
      s50 = res["50"].values[0]
      exp2_data.append({"ラベル": label, "50年": f"{s50:.1%}"})
  print(pd.DataFrame(exp2_data))


def main():
  csv_path = "data/dynamic_spending_grid_comp.csv"
  if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    return

  df = pd.read_csv(csv_path)

  # 代表的な統計を表示
  print_representative_stats(df)

  # ヒートマップを生成する対象年数
  target_years = ["30", "50"]
  # ダイナミックリバランスの有無
  dynamic_rebalance_options = [0, 1]

  # 1. フィッティングの実行 (is_dynamic_rebalance=0 のみ)
  run_fitting(df, ["20", "30", "40", "50"])

  # 2. ヒートマップの生成
  print("\nヒートマップを生成中...")
  for target_year in target_years:
    for is_dyn in dynamic_rebalance_options:
      # データの抽出
      mask = (df["is_dynamic_rebalance"] == is_dyn)
      plot_df = df[mask][["upper_limit", "lower_limit", target_year]].copy()
      plot_df.columns = ["upper_limit", "lower_limit", "survival_rate"]  # type: ignore

      # 表示用に値を調整 (0.0〜1.0 -> 0%〜100%)
      plot_df["survival_rate_pct"] = plot_df["survival_rate"] * 100

      # Altair ヒートマップの作成
      # https://altair-viz.github.io/gallery/layered_heatmap_text.html

      base = alt.Chart(plot_df).encode(
          x=alt.X('lower_limit:O',
                  title='下限 (lower_limit)',
                  axis=alt.Axis(format='.1%')),
          y=alt.Y('upper_limit:O',
                  title='上限 (upper_limit)',
                  sort='descending',
                  axis=alt.Axis(format='.1%')),
      )

      # ヒートマップ部分
      heatmap = base.mark_rect().encode(
          color=alt.Color('survival_rate:Q',
                          title='生存確率',
                          scale=alt.Scale(scheme='redyellowgreen',
                                          domain=[plot_df["survival_rate"].min(),
                                                  1.0]))
      )

      # テキスト部分 (背景色に応じて文字色を変更)
      text_color_threshold = plot_df["survival_rate"].quantile(0.3)

      text = base.mark_text(baseline='middle').encode(
          text=alt.Text('survival_rate_pct:Q', format='.1f'),
          color=alt.condition(
              alt.datum.survival_rate > text_color_threshold,
              alt.value('black'),
              alt.value('white')
          )
      )

      dyn_label = "あり" if is_dyn == 1 else "なし"
      chart = (heatmap + text).properties(
          title=f'{target_year}年後の生存確率 (%) (ダイナミックリバランス: {dyn_label})',
          width=400,
          height=300
      )

      # 保存
      output_path = f"docs/imgs/dynamic_spending/grid_heatmap_{target_year}yr_survival_dyn_reb_{is_dyn}.svg"
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      chart.save(output_path)
      print(f"✅ {output_path} に保存しました。")


if __name__ == "__main__":
  main()
