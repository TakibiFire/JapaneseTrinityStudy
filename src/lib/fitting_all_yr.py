"""
生存確率の予測モデル（フィッティング）に関する共通ライブラリ。
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


class FeatureSetType(Enum):
  """特徴量セットの種類を定義する Enum。"""
  # 初期支出率 (S) を用いた標準的な特徴量セット
  STANDARD = "standard"
  # 初期年間支出額 (Spend) を用いた特徴量セット
  ASSET_SPEND = "asset_spend"


def get_features(df: pd.DataFrame,
                 feature_set_type: FeatureSetType = FeatureSetType.STANDARD) -> pd.DataFrame:
  """
  特徴量セットを生成する。

  Args:
    df: 入力データフレーム。以下の列が必要：
      - 'initial_money': 初期資産 (万円)
      - 'spend_multiplier': 支出倍率 (1.0 = 標準)
      - 'spending_rule' (standard の場合): 支出率 (%)
      - 'initial_annual_cost' (asset_spend の場合): 初期年間支出額 (万円)
    feature_set_type: 生成する特徴量セットの種類。

  Returns:
    pd.DataFrame: 生成された特徴量（M, S, Mult, invM, invS, logM, logS 等）を含むデータフレーム。
  """
  M_val = df["initial_money"].to_numpy().astype(float)
  Mult_val = df["spend_multiplier"].to_numpy().astype(float)

  if feature_set_type == FeatureSetType.STANDARD:
    S_val = df["spending_rule"].to_numpy().astype(float)
    return pd.DataFrame({
        "M": M_val,
        "S": S_val,
        "Mult": Mult_val,
        "invM": 1.0 / np.maximum(M_val, 1.0),
        "invS": 1.0 / np.maximum(S_val, 0.1),
        "logM": np.log(np.maximum(M_val, 1.0)),
        "logS": np.log(np.maximum(S_val, 0.1))
    })
  elif feature_set_type == FeatureSetType.ASSET_SPEND:
    Spend_val = df["initial_annual_cost"].to_numpy().astype(float)
    return pd.DataFrame({
        "M": M_val,
        "Mult": Mult_val,
        "Spend": Spend_val,
        "invM": 1.0 / np.maximum(M_val, 1.0),
        "invSpend": 1.0 / np.maximum(Spend_val, 1.0),
        "logM": np.log(np.maximum(M_val, 1.0)),
        "logSpend": np.log(np.maximum(Spend_val, 1.0))
    })
  else:
    raise ValueError(f"Unknown feature_set_type: {feature_set_type}")


def run_fitting_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_set_type: FeatureSetType = FeatureSetType.STANDARD) -> List[Dict[str, Any]]:
  """
  各種回帰モデル（P, Logit / Degree 2, 3）の評価を行い、結果を標準出力に表示する。

  Args:
    df: 入力データフレーム。get_features で必要な列に加え、target_col で指定された列が必要。
    target_col: ターゲットとなる生存確率の列名 (例: "45", "35")。
    feature_set_type: 使用する特徴量セットの種類。

  Returns:
    List[Dict[str, Any]]: 各モデルの評価結果を含む辞書のリスト。
      辞書のキー:
        - "name": モデル名
        - "poly_deg": 多項式の次数
        - "interaction_only": 交互作用のみか
        - "use_logit": Logit 空間を使用しているか
        - "adj_r2": 調整済み決定係数 (Adjusted R2 Score)
  """
  print(f"\n\n{'='*20} 予測モデルの評価 (R2 Score) - {target_col}年生存確率 {'='*20}")

  y_target = df[target_col].to_numpy().astype(float)

  # 対数オッズ空間
  y_clipped = np.clip(y_target, 0.001, 0.999)
  logit_y = np.log(y_clipped / (1 - y_clipped))

  # 特徴量セット
  feats = get_features(df, feature_set_type)

  results: List[Dict[str, Any]] = []

  def evaluate(name: str, poly_deg: int, interaction_only: bool, use_logit: bool) -> None:
    """
    指定されたパラメータで線形回帰モデルを学習し、R2スコア等を表示・保存する。

    Args:
        name: 評価名
        poly_deg: 多項式の次数
        interaction_only: 交互作用のみか
        use_logit: Logit空間を使用するか
    """
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
    n = len(y_target)
    p = X_poly.shape[1] - 1  # 定数項を除く
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(
        f"{name:<40} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f} | 特徴量数: {X_poly.shape[1]}"
    )
    results.append({
        "name": name,
        "poly_deg": poly_deg,
        "interaction_only": interaction_only,
        "use_logit": use_logit,
        "adj_r2": adj_r2
    })

  evaluate("確率Pに対する2次モデル (交互作用のみ)", 2, True, False)
  evaluate("Logitに対する2次モデル (交互作用のみ)", 2, True, True)
  evaluate("確率Pに対する3次モデル (交互作用のみ)", 3, True, False)
  evaluate("Logitに対する3次モデル (交互作用のみ)", 3, True, True)

  return results


def run_stepwise_fitting_analysis(
    df: pd.DataFrame,
    target_col: str,
    max_adj_r2: float,
    poly_deg: int,
    interaction_only: bool,
    use_logit: bool,
    feature_set_type: FeatureSetType = FeatureSetType.STANDARD
) -> Tuple[LinearRegression, List[str], PolynomialFeatures]:
  """
  ステップワイズ特徴量選択を行い、生存確率の近似式を算出する。

  この関数は、候補となる多数の特徴量（多項式項）の中から、調整済み決定係数（Adj R2）を
  最も改善するものを一つずつ選び出し、モデルに追加していく。
  目標とする Adj R2（threshold）に達するか、最大ステップ数に達するまで繰り返す。
  これにより、予測精度を維持しつつ、特徴量数を抑えた解釈性の高いモデルを得ることができる。

  Args:
    df: 入力データフレーム。get_features で必要な列に加え、target_col で指定された列が必要。
    target_col: ターゲットとなる生存確率の列名。
    max_adj_r2: 全特徴量を使用した際の調整済みR2。
      この値を基準に、モデル構築を終了するしきい値（threshold）を決定する。
    poly_deg: 生成する多項式の次数。
    interaction_only: 交互作用項のみを生成するか。
    use_logit: 目的変数を Logit 空間で扱うか。
    feature_set_type: 使用する特徴量セットの種類。
      STANDARD の場合は max_adj_r2 の 99%、それ以外は 97% をしきい値とする。

  Returns:
    Tuple[LinearRegression, List[str], PolynomialFeatures]:
      - LinearRegression: 学習済みの最終回帰モデル
      - List[str]: 選択された特徴量の名前リスト
      - PolynomialFeatures: 使用された多項式変換器オブジェクト
  """
  print(f"\n\n{'='*20} ステップワイズ特徴量選択による生存確率の近似式算出 {'='*20}")
  print(
      f"ターゲット: {target_col}年生存確率, Logit使用: {use_logit}, Degree: {poly_deg}, InteractionOnly: {interaction_only}"
  )

  # 目標 Adj R2 の設定。feature_set_type によって係数を変える (元のスクリプトの挙動に合わせる)
  if feature_set_type == FeatureSetType.STANDARD:
    threshold = max_adj_r2 * 0.99
    print(f"目標 Adj R2 (99% of {max_adj_r2:.4f}): {threshold:.4f}")
  else:
    threshold = max_adj_r2 * 0.97
    print(f"目標 Adj R2 (97% of {max_adj_r2:.4f}): {threshold:.4f}")

  y_raw = df[target_col].to_numpy().astype(float)
  if use_logit:
    y_clipped = np.clip(y_raw, 0.0001, 0.9999)
    y_target = np.log(y_clipped / (1 - y_clipped))
  else:
    y_target = y_raw

  # 特徴量の準備
  base_feats = get_features(df, feature_set_type)

  poly = PolynomialFeatures(degree=poly_deg, interaction_only=interaction_only)
  X_all = poly.fit_transform(base_feats)
  feature_names = poly.get_feature_names_out(base_feats.columns)

  candidate_df = pd.DataFrame(X_all, columns=feature_names)
  if "1" in candidate_df.columns:
    candidate_df = candidate_df.drop(columns=["1"])

  selected: List[str] = []
  current_r2 = -1.0
  n = len(y_target)

  print(
      f"{'Step':>4} | {'追加された特徴量':<20} | {'R2 Score':>8} | {'Adj R2':>8} | {'向上幅':>8}"
  )
  print("-" * 65)

  for step in range(1, 41):
    best_feat = None
    best_r2 = -1e9
    best_adj_r2 = -1e9
    for feat in candidate_df.columns:
      if feat in selected:
        continue
      trial = selected + [feat]
      model = LinearRegression()
      model.fit(candidate_df[trial], y_target)

      pred_raw = model.predict(candidate_df[trial])
      if use_logit:
        y_pred = 1 / (1 + np.exp(-pred_raw))
      else:
        y_pred = np.clip(pred_raw, 0, 1)

      r2 = r2_score(y_raw, y_pred)
      p = len(trial)
      adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

      if adj_r2 > best_adj_r2:
        best_r2 = r2
        best_adj_r2 = adj_r2
        best_feat = feat

    if best_feat:
      improvement = best_r2 - current_r2 if current_r2 != -1.0 else best_r2
      print(
          f"{step:4d} | {best_feat:<20} | {best_r2:8.4f} | {best_adj_r2:8.4f} | {improvement:+8.4f}"
      )
      selected.append(best_feat)
      current_r2 = best_r2

      if best_adj_r2 >= threshold:
        print(f"目標 Adj R2 に達したため終了します。")
        break
    else:
      break

  # 最終モデル
  final_model = LinearRegression()
  final_model.fit(candidate_df[selected], y_target)

  print(f"\n--- 最終モデルの係数 (ターゲット: {target_col}年) ---")
  print(f"切片 (Intercept): {final_model.intercept_:.6f}")
  coef_df = pd.DataFrame({"特徴量": selected, "係数": final_model.coef_})
  print(coef_df.to_string(index=False))

  # 97% / 90% の算出 (feature_set_type==FeatureSetType.STANDARD の場合のみ元のスクリプトで出力されていた)
  if feature_set_type == FeatureSetType.STANDARD:

    def print_formula(target_val: float, label: str) -> None:
      parts = [f"{final_model.intercept_:.6f}"]
      for f, c in zip(selected, final_model.coef_):
        parts.append(f"({c:+.6f} * {f})")
      formula = " + ".join(parts)
      print(f"\n--- {label} 生存確率の条件式 ---")
      if use_logit:
        logit_p = np.log(target_val / (1 - target_val))
        print(f"Logit(P) = {logit_p:.4f}")
        print(f"0 = {formula} - {logit_p:.4f}")
      else:
        print(f"P = {target_val:.4f}")
        print(f"0 = {formula} - {target_val:.4f}")

    print_formula(0.97, "97%")
    print_formula(0.90, "90%")

  return final_model, selected, poly


def run_survival_curve_analysis(
    df: pd.DataFrame,
    model: LinearRegression,
    selected_feats: List[str],
    poly: PolynomialFeatures,
    use_logit: bool,
    target_probs: List[float]
) -> Tuple[pd.DataFrame, float]:
  """
  近似式を用いて特定の生存確率を達成する曲線を生成し、データフレームを返す。

  Args:
    df: 入力データフレーム。以下の列が必要：
      - 'spend_multiplier': 支出倍率
      - 'initial_annual_cost': 初期年間支出額 (万円)
    model: 学習済みの回帰モデル。
    selected_feats: 選択された特徴量の名前リスト。
    poly: 多項式変換器。
    use_logit: ターゲットが Logit かどうか。
    target_probs: 算出対象の生存確率のリスト (例: [0.95, 0.90])。

  Returns:
    Tuple[pd.DataFrame, float]:
      - pd.DataFrame: 各生存確率・支出率における必要資産や支出レベルのデータフレーム。
      - float: 基準となる支出額 (multiplier=1.0 の時の初期支出額)。
  """
  print(f"\n\n{'='*20} 6. 近似式による生存確率達成データの生成 {'='*20}")

  # 支出率の範囲 (2.8% から 7.0%)
  # グラフが綺麗になるよう、少し細かめに刻む
  fine_rules = np.arange(2.8, 7.01, 0.1)

  # 基準となる支出額 (multiplier=1.0 の時)
  base_cost = df[np.isclose(df["spend_multiplier"], 1.0)]["initial_annual_cost"].iloc[0]

  def get_pred_val(s: float, mult: float) -> float:
    """
    指定された支出率と支出倍率における生存確率（またはLogit）の予測値を算出する。
    """
    m_money = base_cost * mult / (s / 100.0)
    b_feats = pd.DataFrame({
        "M": [m_money],
        "S": [s],
        "Mult": [mult],
        "invM": [1.0 / max(m_money, 1.0)],
        "invS": [1.0 / max(s, 0.1)],
        "logM": [np.log(max(m_money, 1.0))],
        "logS": [np.log(max(s, 0.1))]
    })
    X_poly = poly.transform(b_feats)
    feature_names = poly.get_feature_names_out(b_feats.columns)
    c_df = pd.DataFrame(X_poly, columns=feature_names)
    X_sel = c_df[selected_feats]
    return float(model.predict(X_sel)[0])

  plot_data = []
  # 探索範囲
  m_min, m_max = 0.05, 10.0

  for target_p in target_probs:
    target_val = np.log(target_p / (1 - target_p)) if use_logit else target_p
    label = f"{target_p*100:g}%"

    for s in fine_rules:
      try:
        s_val = float(s)
        v_min = get_pred_val(s_val, m_min) - target_val
        v_max = get_pred_val(s_val, m_max) - target_val

        if v_min * v_max <= 0:
          opt_m = brentq(lambda m: get_pred_val(s_val, m) - target_val, m_min,
                         m_max)
          spend = opt_m * base_cost
          plot_data.append({
              "spending_rule": s_val,
              "multiplier": opt_m,
              "annual_spend_man": spend,
              "initial_money": spend / (s_val / 100.0),
              "target_prob": label
          })
      except Exception:
        pass

  df_plot = pd.DataFrame(plot_data)
  return df_plot, float(base_cost)


def save_survival_charts(
    df_plot: pd.DataFrame,
    base_cost: float,
    target_probs: List[float],
    img_dir: str
) -> None:
  """
  生成されたデータから生存確率達成ラインのグラフを作成して保存する。

  Args:
    df_plot: run_survival_curve_analysis で生成されたデータフレーム。
    base_cost: 基準となる支出額。
    target_probs: ターゲット生存確率のリスト。
    img_dir: 画像の保存先ディレクトリ。
  """
  if df_plot.empty:
    print("曲線を描画できるデータが見わたりませんでした。")
    return

  prob_order = [f"{p*100:g}%" for p in target_probs]

  # 1. 支出率 (S) vs 支出レベル (Spend)
  chart1 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('spending_rule:Q',
              title='初期支出率 (%)',
              scale=alt.Scale(domain=[2.8, 7.0])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, base_cost * 3.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (初期支出率 vs 支出レベル)", width=600, height=400)

  path1 = os.path.join(img_dir, "survival_rule_vs_spend.svg")
  chart1.save(path1)
  print(f"✅ {path1} に保存しました。")

  # 2-a: x=総資産, y=支出レベル
  chart2 = alt.Chart(df_plot).mark_line(point=True, clip=True).encode(
      x=alt.X('initial_money:Q',
              title='総資産 (万円)',
              scale=alt.Scale(domain=[0, 30000])),
      y=alt.Y('annual_spend_man:Q',
              title='支出レベル (万円/年)',
              scale=alt.Scale(domain=[0, base_cost * 3.0])),
      color=alt.Color('target_prob:N',
                      title='目標生存確率',
                      sort=prob_order,
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (総資産 vs 支出レベル)", width=600, height=400)

  path2 = os.path.join(img_dir, "survival_asset_vs_spend.svg")
  chart2.save(path2)
  print(f"✅ {path2} に保存しました。")

  # 2-b: x=総資産, y=初期支出率
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
                      scale=alt.Scale(domain=prob_order)),
      tooltip=['spending_rule', 'multiplier', 'annual_spend_man', 'initial_money', 'target_prob']
  ).properties(title="生存確率達成ライン (総資産 vs 初期支出率)", width=600, height=400)

  path3 = os.path.join(img_dir, "survival_asset_vs_rule.svg")
  chart3.save(path3)
  print(f"✅ {path3} に保存しました。")
