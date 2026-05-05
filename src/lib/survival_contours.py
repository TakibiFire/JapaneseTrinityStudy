"""
生存確率のグリッドデータから等高線（指定確率を達成するライン）を抽出し、
グラフ化や簡略化された公式（Rule of Thumb）を生成するためのライブラリ。
PCHIP（Piecewise Cubic Hermite Interpolating Polynomial）を用いて、
単調性を保証しながら滑らかな曲線を生成する。
"""

import os
from typing import Any, Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import LinearRegression


def get_contour_anchor_points(
    df: pd.DataFrame,
    target_prob: float,
    target_year: str = "35"
) -> List[Tuple[float, float, float]]:
  """
  指定された目標生存確率を達成する、正確な (初期支出率, 支出レベル, 初期資産) のポイントを抽出する。
  支出レベル(initial_annual_cost)ごとに、確率と支出率の関係を PCHIP で補間し、
  目標確率となる支出率を逆算する。

  Args:
    df: 分析対象のデータフレーム。以下の列が必要。
      - initial_annual_cost: 初期年間支出額 (Spend)
      - spending_rule: 初期支出率 (Rule, %)
      - target_yearで指定される生存確率の列
    target_prob: 目標となる生存確率 (例: 0.95)
    target_year: ターゲットとする生存確率の列名 (例: "35")

  Returns:
    List[Tuple[float, float, float]]: (Spending Rule, Spend, Initial Money) のリスト。
      条件を満たさない（目標確率がデータ範囲外の）Spendティアは除外される。
  """
  anchor_points: List[Tuple[float, float, float]] = []

  for spend_val, group in df.groupby("initial_annual_cost"):
    # groupby のキーは float であることを明示する
    spend = float(spend_val)  # type: ignore
    group = group.sort_values("spending_rule")
    rules = group["spending_rule"].to_numpy(dtype=float)
    probs = group[target_year].to_numpy(dtype=float)

    # 支出率(Rule)が上がると確率(Prob)は下がるため、Probは単調減少する。
    # PCHIPはx軸が単調増加であることを要求するため、配列を反転させる。
    x = probs[::-1]
    y = rules[::-1]

    # フラットな領域（確率が1.0のまま等）を避けるため、一意のxのみ抽出する
    x_unique, indices = np.unique(x, return_index=True)
    y_unique = y[indices]

    if len(x_unique) >= 2 and x_unique.min() <= target_prob <= x_unique.max():
      f_point = PchipInterpolator(x_unique, y_unique)
      exact_rule = float(f_point(target_prob))
      m_money = float(spend / (exact_rule / 100.0))
      anchor_points.append((exact_rule, spend, m_money))

  # Spend の昇順にソートして返す
  anchor_points.sort(key=lambda p: p[1])
  return anchor_points


def generate_smooth_contour_data(
    anchor_points: List[Tuple[float, float, float]],
    target_label: str,
    num_points: int = 100
) -> List[Dict[str, Any]]:
  """
  アンカーポイントを基に、描画用の滑らかで高密度なデータポイントを生成する。
  Spend(支出レベル)に対するRule(支出率)の曲線をPCHIPで補間する。

  Args:
    anchor_points: get_contour_anchor_points で取得したアンカーポイントのリスト
    target_label: 結果の辞書に付与するラベル (例: "95%")
    num_points: 生成するデータポイントの数

  Returns:
    List[Dict[str, Any]]: 描画用データフレームに変換できる辞書のリスト
  """
  if len(anchor_points) < 2:
    return []

  spend_vals = np.array([p[1] for p in anchor_points])
  rule_vals = np.array([p[0] for p in anchor_points])

  # Spend は単調増加である前提
  f_curve = PchipInterpolator(spend_vals, rule_vals)

  dense_spends = np.linspace(spend_vals.min(), spend_vals.max(), num_points)
  dense_rules = f_curve(dense_spends)

  plot_data = []
  for s, r in zip(dense_spends, dense_rules):
    m = s / (r / 100.0)
    plot_data.append({
        "target_prob": target_label,
        "annual_spend_man": float(s),
        "spending_rule": float(r),
        "initial_money": float(m)
    })

  return plot_data


def save_contour_charts(df_plot: pd.DataFrame,
                        target_probs: List[float],
                        img_dir: str,
                        prefix: str = "") -> None:
  """
  生成された高密度データから、生存確率達成ラインの3種のグラフを作成・保存する。

  Args:
    df_plot: generate_smooth_contour_data の結果をまとめたデータフレーム
    target_probs: 描画対象の目標生存確率のリスト (ソート順指定のため)
    img_dir: 保存先ディレクトリ
    prefix: 保存ファイル名のプレフィックス
  """
  if df_plot.empty:
    print("曲線を描画できるデータがありませんでした。")
    return

  os.makedirs(img_dir, exist_ok=True)
  prob_order = [f"{p*100:g}%" for p in target_probs]

  # 1. 支出率 (S) vs 支出レベル (Spend)
  chart1 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('spending_rule:Q',
              title='初期支出率 (%)',
              scale=alt.Scale(domain=[2.8, 7.0])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, df_plot["annual_spend_man"].max() * 1.1])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order))
  ).properties(title="生存確率達成ライン (初期支出率 vs 支出レベル)", width=600, height=400)

  path1 = os.path.join(img_dir, f"{prefix}survival_rule_vs_spend.svg")
  chart1.save(path1)
  print(f"✅ {path1} に保存しました。")

  # 2. 総資産 (M) vs 支出レベル (Spend)
  chart2 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('initial_money:Q',
              title='総資産 (万円)',
              scale=alt.Scale(domain=[0, 30000])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, df_plot["annual_spend_man"].max() * 1.1])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order))
  ).properties(title="生存確率達成ライン (総資産 vs 支出レベル)", width=600, height=400)

  path2 = os.path.join(img_dir, f"{prefix}survival_asset_vs_spend.svg")
  chart2.save(path2)
  print(f"✅ {path2} に保存しました。")

  # 3. 総資産 (M) vs 初期支出率 (Rule)
  chart3 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('initial_money:Q',
              title='総資産 (万円)',
              scale=alt.Scale(domain=[0, 30000])),
      y=alt.Y('spending_rule:Q',
              title='初期支出率 (%)',
              scale=alt.Scale(domain=[2.8, 7.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order))
  ).properties(title="生存確率達成ライン (総資産 vs 初期支出率)", width=600, height=400)

  path3 = os.path.join(img_dir, f"{prefix}survival_asset_vs_rule.svg")
  chart3.save(path3)
  print(f"✅ {path3} に保存しました。")


def generate_rule_of_thumb(
    df: pd.DataFrame,
    target_probs: List[float],
    target_year: str = "35"
) -> None:
  """
  アンカーポイントに対して線形回帰を行い、人間が理解しやすい簡略化された公式を生成する。
  Spend = a * M + b の形式を特定し、ターミナルに出力する。

  Args:
    df: 分析対象のデータフレーム
    target_probs: 公式を算出する対象の生存確率リスト
    target_year: ターゲットとする生存確率の列名
  """
  print(f"\n\n{'='*20} 初期支出額を求める公式 (Rule of Thumb) {'='*20}")
  print("| 目標生存確率 | Spendを求める公式 | R2 Score |")
  print("| --: | --- | --- |")

  for p in target_probs:
    anchor_points = get_contour_anchor_points(df, p, target_year)
    if len(anchor_points) < 2:
      continue

    m_vals = np.array([pt[2] for pt in anchor_points]).reshape(-1, 1)
    s_vals = np.array([pt[1] for pt in anchor_points])

    model = LinearRegression().fit(m_vals, s_vals)
    a = model.coef_[0]
    b = model.intercept_
    r2 = model.score(m_vals, s_vals)

    formula = f"M の {a*100:.2f}% + {b:.0f}万円"
    print(f"| {p*100:g}% | Spend = {formula} | {r2:.4f} |")
