"""
生存確率(P)、初期資産(M)、初期支出額(Spend)の関係を分析し、
簡略化された統一公式を生成・評価するためのライブラリ。

内容:
1. 2次形式モデル (Quadratic Unified)
2. 単純線形モデル (Simple Linear)
3. 双曲的モデル (Hyper-Alpha + Lin-Beta)
4. 直接有理関数モデル (Direct Rational Fit)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.lib.survival_contours import get_contour_anchor_points


def run_survival_formula_analysis(df_survival: pd.DataFrame,
                                  target_year: str = "35") -> dict:
  """
  生存確率のグリッドデータから、様々な近似モデルを構築し、その精度を比較・表示する。

  Args:
    df_survival: 生存確率のデータフレーム。
    target_year: 対象年数（リタイアからの経過年数）。

  Returns:
    Direct Rational Model の係数 (rc1 ~ rc5) を含む辞書。
  """
  # Target probabilities from 60% to 99% (initial data gathering)
  # Use linspace to avoid float precision issues
  base_probs = np.linspace(0.60, 0.99, 40)
  # Ensure special points are exactly included
  special_points = [0.98, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]
  target_probs = np.unique(np.concatenate([base_probs, special_points]))

  results = []
  data_points = []

  for p in target_probs:
    anchors = get_contour_anchor_points(df_survival, p, target_year)
    if len(anchors) < 2:
      continue

    m_vals = np.array([pt[2] for pt in anchors]).reshape(-1, 1)
    s_vals = np.array([pt[1] for pt in anchors])
    model = LinearRegression().fit(m_vals, s_vals)

    results.append({
        "p": p,
        "logit_p": np.log(p / (1 - p)),
        "alpha": model.coef_[0],
        "beta": model.intercept_,
        "r2": model.score(m_vals, s_vals)
    })

    for rule, spend, m in anchors:
      data_points.append({
          "p": p,
          "logit_p": np.log(p / (1 - p)),
          "M": m,
          "Spend": spend
      })

  res_df = pd.DataFrame(results)
  df_fit = pd.DataFrame(data_points)

  # Filter range to 60% ~ 98% as requested
  P_MIN, P_MAX = 0.60, 0.98
  res_df_f = res_df[(res_df["p"] >= P_MIN) & (res_df["p"] <= P_MAX)].copy()
  df_fit_f = df_fit[(df_fit["p"] >= P_MIN) & (df_fit["p"] <= P_MAX)].copy()
  L = res_df_f["logit_p"].to_numpy()

  print(f"\n\n{'='*20} 生存確率近似モデルの分析 ({P_MIN*100:g}%-{P_MAX*100:g}%) {'='*20}")

  # --- [Idea 1] Quadratic fit for alpha and beta ---
  L2 = L**2
  X_quad = np.column_stack([L2, L, np.ones_like(L)])
  mod_a_q = LinearRegression(fit_intercept=False).fit(X_quad, res_df_f["alpha"])
  qa1, qa2, qa3 = mod_a_q.coef_
  mod_b_q = LinearRegression(fit_intercept=False).fit(X_quad, res_df_f["beta"])
  qb1, qb2, qb3 = mod_b_q.coef_

  def solve_logit_quad(m, spend):
    A_coeff = qa1 * m + qb1
    B_coeff = qa2 * m + qb2
    C_coeff = qa3 * m + qb3 - spend
    disc = B_coeff**2 - 4 * A_coeff * C_coeff
    if disc < 0:
      return np.nan
    root1 = (-B_coeff + np.sqrt(disc)) / (2 * A_coeff)
    root2 = (-B_coeff - np.sqrt(disc)) / (2 * A_coeff)
    if 0.3 <= root1 <= 4.5:
      return root1
    if 0.3 <= root2 <= 4.5:
      return root2
    return root1

  # --- [Idea 2] Simple Linear-in-logit Approximation ---
  X_lin = df_fit_f[["M"]].copy()
  X_lin["logit_p_M"] = df_fit_f["logit_p"] * df_fit_f["M"]
  X_lin["logit_p"] = df_fit_f["logit_p"]
  y_lin = df_fit_f["Spend"]
  model_lin = LinearRegression().fit(X_lin, y_lin)
  cm, km, kc, cc = model_lin.coef_[0], model_lin.coef_[1], model_lin.coef_[
      2], model_lin.intercept_

  # --- [Idea 3] Hyperbolic alpha + Linear beta ---
  inv_alpha = 1.0 / res_df_f["alpha"]
  mod_a_h = LinearRegression().fit(L.reshape(-1, 1), inv_alpha)
  ka, ia = mod_a_h.coef_[0], mod_a_h.intercept_
  mod_b_l = LinearRegression().fit(L.reshape(-1, 1), res_df_f["beta"])
  kb, ib = mod_b_l.coef_[0], mod_b_l.intercept_

  def solve_logit_h1(m, spend):
    A = -ka * kb
    B = ka * spend - ka * ib - ia * kb
    C = ia * spend - ia * ib - m
    disc = B**2 - 4 * A * C
    if disc < 0:
      return np.nan
    return (-B + np.sqrt(disc)) / (2 * A)

  # --- [Idea 4] Direct Rational Fit ---
  X_r = pd.DataFrame({
      "Spend": df_fit_f["Spend"],
      "M": df_fit_f["M"],
      "1": 1.0,
      "L*Spend": -df_fit_f["logit_p"] * df_fit_f["Spend"],
      "L*M": -df_fit_f["logit_p"] * df_fit_f["M"]
  })
  mod_r = LinearRegression(fit_intercept=False).fit(X_r, df_fit_f["logit_p"])
  rc1, rc2, rc3, rc4, rc5 = mod_r.coef_

  # --- Evaluations and Tables ---
  df_fit_f["p_quad"] = df_fit_f.apply(
      lambda r: solve_logit_quad(r["M"], r["Spend"]), axis=1)
  df_fit_f["p_lin"] = (df_fit_f["Spend"] -
                       (cm * df_fit_f["M"] + cc)) / (km * df_fit_f["M"] + kc)
  df_fit_f["p_h1"] = df_fit_f.apply(
      lambda r: solve_logit_h1(r["M"], r["Spend"]), axis=1)
  df_fit_f["p_rat"] = (rc1 * df_fit_f["Spend"] + rc2 * df_fit_f["M"] + rc3) / (
      rc4 * df_fit_f["Spend"] + rc5 * df_fit_f["M"] + 1)

  models = [("Quadratic (Unified)", df_fit_f["p_quad"],
             lambda l: qa1 * l**2 + qa2 * l + qa3,
             lambda l: qb1 * l**2 + qb2 * l + qb3),
            ("Simple Linear", df_fit_f["p_lin"], lambda l: km * l + cm,
             lambda l: kc * l + cc),
            ("Hyper-Alpha + Lin-Beta", df_fit_f["p_h1"], lambda l: 1 /
             (ka * l + ia), lambda l: kb * l + ib),
            ("Direct Rational", df_fit_f["p_rat"], lambda l: (rc2 - l * rc5) /
             (l * rc4 - rc1), lambda l: (rc3 - l) / (l * rc4 - rc1))]

  for name, pred, a_func, b_func in models:
    print(f"\n--- {name} ---")
    valid_pred = df_fit_f.dropna(subset=[pred.name])
    r2 = r2_score(valid_pred['logit_p'], valid_pred[pred.name])
    print(f"R2 (Logit space): {r2:.6f}")

    if name == "Quadratic (Unified)":
      print(f"A(M) = ({qa1:.6f}*M + {qb1:.6f})")
      print(f"B(M) = ({qa2:.6f}*M + {qb2:.6f})")
      print(f"C(M, Spend) = ({qa3:.6f}*M + {qb3:.6f} - Spend)")
    elif name == "Simple Linear":
      print(
          f"logit(P) ≈ ((M * {-cm*100:.2f}% + {-cc:.1f}) - Spend) / (M * {km:.4f} + {kc:.2f})"
      )
    elif name == "Hyper-Alpha + Lin-Beta":
      print(f"alpha = 1 / ({ka:.6f} * L + {ia:.6f})")
      print(f"beta = {kb:.4f} * L + {ib:.1f}")
    elif name == "Direct Rational":
      # Output with more precision for small coefficients
      print(
          f"logit(P) = ({rc1:.8f}*Spend + {rc2:.8f}*M + {rc3:.8f}) / ({rc4:.8f}*Spend + {rc5:.8e}*M + 1)"
      )
      print(
          f"Spend = (M * ({rc2:.8f} - L * {rc5:.8e}) + ({rc3:.8f} - L)) / (L * {rc4:.8f} - {rc1:.8f})"
      )
      print(
          f"M = (Spend * ({rc1:.8f} - L * {rc4:.8f}) + {rc3:.8f} - L) / (L * {rc5:.8e} - {rc2:.8f})"
      )

    print("| P | Target Alpha | Model Alpha | Target Beta | Model Beta |")
    for p in [0.98, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]:
      lp = np.log(p / (1 - p))
      t_row = res_df_f[np.isclose(res_df_f["p"], p, atol=1e-5)]
      if t_row.empty:
        continue
      ta, tb = t_row.iloc[0]["alpha"], t_row.iloc[0]["beta"]
      print(
          f"| {p*100:g}% | {ta*100:.2f}% | {a_func(lp)*100:.2f}% | {tb:.1f} | {b_func(lp):.1f} |"
      )

  return {
      "rc1": rc1,
      "rc2": rc2,
      "rc3": rc3,
      "rc4": rc4,
      "rc5": rc5,
  }
