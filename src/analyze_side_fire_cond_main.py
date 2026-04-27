"""
data/side_fire_cond_grid.csv の結果を分析・可視化するスクリプト。

内容:
1. 予測モデルの評価 (P vs Logit, Degree 2 vs 3)
2. ステップワイズ特徴量選択による影響度の分析
3. 2次元ヒートマップによる可視化
(30年生存確率と50年生存確率の両方を対象とする)
"""

import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 設定
CSV_PATH = "data/side_fire_cond_grid.csv"
IMG_DIR = "docs/imgs/side_fire"


def run_fitting_analysis(df: pd.DataFrame, target_col: str):
  """
  予測モデルの評価を行い STDOUT に出力する。
  """
  print(f"\n=== 1. 予測モデルの評価 (R2 Score) - {target_col}年生存確率 ===")

  y_target = df[target_col].values.astype(float)
  X_val = df["threshold_x"].values.astype(float)
  Y_val = df["max_year_y"].values.astype(float)
  Z_val = df["income_percent_z"].values.astype(float)

  # 対数オッズ空間
  y_clipped = np.clip(y_target, 0.001, 0.999)
  logit_y = np.log(y_clipped / (1 - y_clipped))

  # 特徴量セット
  feats = pd.DataFrame({
      "X": X_val,
      "Y": Y_val,
      "Z": Z_val,
      "invX": 1.0 / X_val,
      "invY": 1.0 / Y_val,
      "invZ": 1.0 / Z_val
  })

  def evaluate(name, poly_deg, interaction_only, use_logit):
    poly = PolynomialFeatures(degree=poly_deg,
                              interaction_only=interaction_only)
    X_poly = poly.fit_transform(feats)

    target = logit_y if use_logit else y_target
    model = LinearRegression()
    model.fit(X_poly, target)

    pred_raw = model.predict(X_poly)
    if use_logit:
      y_pred = 1 / (1 + np.exp(-pred_raw))
    else:
      y_pred = np.clip(pred_raw, 0, 1)

    r2 = r2_score(y_target, y_pred)
    print(f"{name:<40} | R2: {r2:.4f} | 特徴量数: {X_poly.shape[1]}")

  evaluate("確率Pに対する線形モデル", 1, False, False)
  evaluate("Logitに対する線形モデル", 1, False, True)
  evaluate("確率Pに対する2次モデル (交互作用のみ)", 2, True, False)
  evaluate("Logitに対する2次モデル (交互作用のみ)", 2, True, True)


def run_stepwise_analysis(df: pd.DataFrame, target_col: str):
  """
  ステップワイズ特徴量選択による影響度の分析を行い STDOUT に出力する。
  """
  print(f"\n=== 2. ステップワイズ特徴量選択 - {target_col}年生存確率 ===")

  y_target = df[target_col].values.astype(float)
  X_val = df["threshold_x"].values.astype(float)
  Y_val = df["max_year_y"].values.astype(float)
  Z_val = df["income_percent_z"].values.astype(float)

  base_dict = {
      "X": X_val,
      "Y": Y_val,
      "Z": Z_val,
      "invX": 1.0 / X_val,
      "invY": 1.0 / Y_val,
      "invZ": 1.0 / Z_val
  }

  # 候補プールの作成
  candidates = {}
  keys = list(base_dict.keys())
  for k in keys:
    candidates[k] = base_dict[k]
  for i in range(len(keys)):
    for j in range(i, len(keys)):
      k1, k2 = keys[i], keys[j]
      if (k1 == "X" and k2 == "invX") or (k1 == "Y" and
                                          k2 == "invY") or (k1 == "Z" and
                                                            k2 == "invZ"):
        continue
      name = f"{k1}*{k2}" if i != j else f"{k1}^2"
      candidates[name] = base_dict[k1] * base_dict[k2]

  candidate_df = pd.DataFrame(candidates)

  selected = []
  current_r2 = 0
  print(f"{'Step':>4} | {'追加された特徴量':<20} | {'R2 Score':>8} | {'向上幅':>8}")
  print("-" * 55)

  for step in range(1, 11):
    best_feat = None
    best_r2 = -1
    for feat in candidate_df.columns:
      if feat in selected:
        continue
      trial = selected + [feat]
      model = LinearRegression()
      model.fit(candidate_df[trial], y_target)
      r2 = r2_score(y_target, model.predict(candidate_df[trial]))
      if r2 > best_r2:
        best_r2 = r2
        best_feat = feat

    if best_feat:
      improvement = best_r2 - current_r2
      print(
          f"{step:4d} | {best_feat:<20} | {best_r2:8.4f} | {improvement:+8.4f}")
      selected.append(best_feat)
      current_r2 = best_r2
    else:
      break

  # 標準化係数
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(candidate_df[selected])
  final_model = LinearRegression()
  final_model.fit(X_scaled, y_target)

  print(f"\n--- 最終モデルの標準化係数 ({target_col}年) ---")
  coef_df = pd.DataFrame({"特徴量": selected, "標準化係数": final_model.coef_})
  coef_df["絶対値"] = coef_df["標準化係数"].abs()
  print(coef_df.sort_values("絶対値", ascending=False).to_string(index=False))


def create_heatmap(df: pd.DataFrame,
                   target_col: str,
                   title: str,
                   x_col: str,
                   x_title: str,
                   y_col: str,
                   y_title: str,
                   output_name: str,
                   x_sort: List = None,
                   y_sort: List = None):
  """
  ヒートマップを作成して保存する。
  """
  plot_df = df.copy()
  plot_df["survival_rate"] = plot_df[target_col]
  plot_df["survival_rate_pct"] = plot_df["survival_rate"] * 100

  base = alt.Chart(plot_df).encode(
      x=alt.X(f'{x_col}:O', title=x_title, sort=x_sort),
      y=alt.Y(f'{y_col}:O', title=y_title, sort=y_sort),
  )

  heatmap = base.mark_rect().encode(color=alt.Color(
      'survival_rate:Q',
      title='生存確率',
      scale=alt.Scale(scheme='redyellowgreen', domain=[0.4, 1.0])))

  text = base.mark_text(baseline='middle').encode(
      text=alt.Text('survival_rate_pct:Q', format='.1f'),
      color=alt.condition(alt.datum.survival_rate > 0.6, alt.value('black'),
                          alt.value('white')))

  chart = (heatmap + text).properties(title=title, width=400, height=300)

  # STDOUT出力
  print(f"\n--- {title} ---")
  pivot = plot_df.pivot(index=y_col, columns=x_col, values="survival_rate_pct")
  if y_sort:
    pivot = pivot.reindex(index=y_sort)
  if x_sort:
    pivot = pivot.reindex(columns=x_sort)
  print(pivot.to_string())

  output_path = os.path.join(IMG_DIR, output_name)
  os.makedirs(IMG_DIR, exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def main():
  if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} が見つかりません。")
    return

  df_all = pd.read_csv(CSV_PATH)

  # Baseline (x=1.01 または scenario="Baseline") の値を取得して表示
  baseline_row = df_all[df_all["threshold_x"] > 1.0]
  if not baseline_row.empty:
    print("\n=== Baseline (労働なし) の生存確率 ===")
    print(f"30年生存確率: {baseline_row['30'].values[0]*100:.1f}%")
    print(f"50年生存確率: {baseline_row['50'].values[0]*100:.1f}%")

  # ワークシナリオのみを抽出
  df = df_all[df_all["threshold_x"] <= 1.0].copy()

  # ラベルの日本語化
  df["threshold_label"] = df["threshold_x"].map(
      lambda x: f"x{1.0/x:.1f}"
      if 1.0 / x != int(1.0 / x) else f"x{int(1.0/x)}")
  df["income_label"] = (df["income_percent_z"] * 100).map(lambda x: f"{x:.0f}%")
  df["year_label"] = df["max_year_y"].map(lambda x: f"{x}年")

  x_order = df["threshold_label"].unique().tolist()  # ソート順はデータ通り
  z_order = ["25%", "50%", "75%", "100%"]
  y_order = ["10年", "20年", "30年", "40年", "50年"]

  for year_target in ["30", "50"]:
    # 冗長なデータ (最大労働期間 Y > 評価期間) を除外
    df_filtered = df[df["max_year_y"] <= int(year_target)].copy()

    # 1 & 2. 分析の実行
    run_fitting_analysis(df_filtered, year_target)
    run_stepwise_analysis(df_filtered, year_target)

    # 3. 可視化
    current_y_order = [
        y for y in y_order if int(y.replace("年", "")) <= int(year_target)
    ]

    # ヒートマップ1: 期間(Y) vs 収入(Z) [閾値 X=10% (x10) 固定]
    df_x10 = df_filtered[np.isclose(df_filtered["threshold_x"], 0.10)].copy()
    create_heatmap(df_x10,
                   target_col=year_target,
                   title=f"条件付き労働の生存確率(%) [トリガー=x10固定, {year_target}年後]",
                   x_col="income_label",
                   x_title="労働収入レベル (支出比)",
                   y_col="year_label",
                   y_title="最大労働期間",
                   output_name=f"heatmap_yz_at_x10_{year_target}yr.svg",
                   x_sort=z_order,
                   y_sort=current_y_order[::-1])

    # ヒートマップ2: 閾値(1/X) vs 収入(Z) [期間 Y=20年 固定]
    # Y=20固定のヒートマップは、全てのターゲット期間で有効
    df_y20 = df_filtered[df_filtered["max_year_y"] == 20].copy()
    if not df_y20.empty:
      create_heatmap(df_y20,
                     target_col=year_target,
                     title=f"条件付き労働の生存確率(%) [最大期間=20年固定, {year_target}年後]",
                     x_col="threshold_label",
                     x_title="トリガー閾値 (資産/支出)",
                     y_col="income_label",
                     y_title="労働収入レベル (支出比)",
                     output_name=f"heatmap_xz_at_y20_{year_target}yr.svg",
                     x_sort=x_order,
                     y_sort=z_order[::-1])


if __name__ == "__main__":
  main()
