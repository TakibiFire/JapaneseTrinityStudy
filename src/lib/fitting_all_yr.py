"""
生存確率の予測モデル（フィッティング）に関する共通ライブラリ。
"""

from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


class FeatureSetType(Enum):
  """特徴量セットの種類を定義する Enum。"""
  # NOW: Need detailed comments.
  STANDARD = "standard"
  # NOW: Need detailed comments.
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
