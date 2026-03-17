"""
data/withdrawal_rate_comp.csv を読み込み、
各支出率と各経過年数において生存確率を最大化するオルカン比率を特定し、
表形式で出力するスクリプト。
さらに、資産寿命 N_ruin を境界とした Piecewise モデルを用いて
最適なオルカン比率を予測する近似式を計算します。
"""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.optimize as opt


def main():
  # CSVデータの読み込み
  csv_path = "data/withdrawal_rate_comp.csv"
  if not os.path.exists(csv_path):
    print(
        f"Error: {csv_path} が見つかりません。先に withdrawal_rate_comp_main.py を実行してください。"
    )
    return

  df = pd.read_csv(csv_path)

  # 経過年数のカラム（"1"〜"50"）を特定
  year_cols = [str(y) for y in range(1, 51)]

  # 支出率ごとに処理
  spend_ratios = sorted(df['spend_ratio'].unique())

  ratio_summary_rows = []
  prob_summary_rows = []

  for spend_ratio in spend_ratios:
    df_spend = df[df['spend_ratio'] == spend_ratio].copy()

    # 行データを作成
    ratio_row: Dict[str, Any] = {"spend_ratio": spend_ratio}
    prob_row: Dict[str, Any] = {"spend_ratio": spend_ratio}

    for year in year_cols:
      # その年において生存確率を最大化するインデックスを取得
      max_val = df_spend[year].max()
      best_rows = df_spend[df_spend[year] == max_val]

      # 同率1位が複数ある場合は、後の処理のために1つ選ぶ（ここでは最初のもの）
      best_row = best_rows.iloc[0]

      # オルカン比率と生存確率を格納
      ratio_row[year] = best_row['orukan_ratio']
      prob_row[year] = max_val

    ratio_summary_rows.append(ratio_row)
    prob_summary_rows.append(prob_row)

  # データフレームを作成
  summary_df = pd.DataFrame(ratio_summary_rows)
  summary_df.set_index("spend_ratio", inplace=True)
  summary_df.index.name = "spend_ratio / year"

  prob_df = pd.DataFrame(prob_summary_rows)
  prob_df.set_index("spend_ratio", inplace=True)
  prob_df.index.name = "spend_ratio / year"

  print("\n各支出率と経過年数において生存確率を最大化する「最適オルカン比率」:")
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  pd.set_option('display.max_rows', None)
  print(summary_df)

  print("\n各支出率と経過年数における「最大生存確率」:")
  print(prob_df)

  # CSVとして保存
  output_ratio_csv = "data/optimal_orukan_ratio.csv"
  summary_df.to_csv(output_ratio_csv)
  output_prob_csv = "data/max_survival_probability.csv"
  prob_df.to_csv(output_prob_csv)
  print(f"✅ {output_ratio_csv} および {output_prob_csv} に保存しました。")
  print("\n")

  # --- 資産寿命（無リスク資産のみ）の計算 ---
  def get_ruin_year(S: float) -> float:
    # 実際のシミュレーション設定に基づく定数
    r_base = 0.04
    tax = 0.20315
    inflation = 0.02

    r_eff = r_base * (1.0 - tax)
    i_ln = np.log(1.0 + inflation)
    delta = r_eff - i_ln

    if S <= delta:
      return 999.0  # 支出が実質利回り以下なら理論上無限

    # P(t) = 0 となる t を解く
    return np.log(1.0 - delta / S) / (-delta)

  def get_features_for_piece(S: float, N_raw: float,
                             N_ruin: float) -> Dict[str, float]:
    n = N_raw / 50.0  # 正規化
    m = (N_raw - N_ruin) / 50.0  # 資産寿命からの超過年数 (Region 2用)

    # 共通の特徴量
    feats = {
        "S": S,
        "S^2": S**2,
        "n": n,
        "n^2": n**2,
        "1/S": 1.0 / S,
        "1/n": 1.0 / n,
        "n*S": n * S,
        "n/S": n / S,
        "S/n": S / n,
        "log(S)": np.log(S),
        "log(n)": np.log(max(n, 0.001)),
        "log(n*S)": np.log(max(n * S, 0.0001)),
        "exp(S)": np.exp(S),
        "exp(n)": np.exp(n),
        "exp(-n)": np.exp(-n),
        "1/(n*S)": 1.0 / (max(n * S, 0.0001))
    }

    # Region 2用の特徴量を追加
    if m > 0:
      feats.update({
          "m": m,
          "m^2": m**2,
          "1/m": 1.0 / max(m, 0.001),
          "m*S": m * S,
          "m/S": m / S,
          "S/m": S / max(m, 0.001),
          "log(m)": np.log(max(m, 0.0001)),
          "exp(m)": np.exp(m),
          "exp(-m)": np.exp(-m),
          "sqrt(m)": np.sqrt(max(m, 0))
      })
    return feats

  def solve_piece(points, name, num_terms=4):
    if not points:
      return None, None

    # 各点において N_ruin を再計算
    X_dict_list = [
        get_features_for_piece(S, float(N), get_ruin_year(S))
        for S, N, v in points
    ]
    Y_arr = np.array([v for S, N, v in points])
    feature_names = list(X_dict_list[0].keys())

    selected: List[str] = []
    best_w: np.ndarray = np.zeros(1)

    def loss_func(w, X, Y):
      raw = X.dot(w)
      clamped = np.clip(raw, 0.0, 1.0)
      # L2 正則化を追加 (係数が巨大化するのを防ぐ)
      return np.sum((clamped - Y)**2) + 0.01 * np.sum(w**2)

    print(f"\n[{name}] のステップワイズ選択:")
    for step in range(num_terms):
      best_feature = ""
      best_error = float('inf')
      current_best_w = np.zeros(1)

      # Region 2 の場合、n 系の変数よりも m 系の変数を優先的に試すようにしてもよいが、
      # ここでは全候補から選ぶ
      for f in feature_names:
        if f in selected:
          continue

        current_fs = selected + [f]
        # X行列の構築 (先頭は定数項の1.0)
        X_mat = np.zeros((len(X_dict_list), len(current_fs) + 1))
        X_mat[:, 0] = 1.0
        for i, d in enumerate(X_dict_list):
          for j, cf in enumerate(current_fs):
            # 特徴量が辞書にない場合は0（Region 1 で m 系が呼ばれた場合など）
            X_mat[i, j + 1] = d.get(cf, 0.0)

        # 最小二乗法で初期値を得る
        w0, _, _, _ = np.linalg.lstsq(X_mat, Y_arr, rcond=None)

        # scipyでクリッピングを考慮した最適化
        res = opt.minimize(loss_func,
                           w0,
                           args=(X_mat, Y_arr),
                           method='L-BFGS-B')
        error = res.fun

        if error < best_error:
          best_error = error
          best_feature = f
          current_best_w = res.x

      selected.append(best_feature)
      best_w = current_best_w
      print(f" Step {step+1}: {best_feature:<15} (Loss: {best_error:.4f})")
    return selected, best_w

  def get_training_points(data_df, filter_boundary=False):
    before = []
    after = []
    for spend_ratio in data_df.index:
      row = data_df.loc[spend_ratio]
      vals = row.values
      N_ruin = get_ruin_year(float(spend_ratio))
      for i in range(len(vals)):
        y_val = i + 1
        v = float(vals[i])
        # 100% または 0% 付近の平坦なデータを除外して学習（勾配のある場所を重点的に学習）
        if filter_boundary and (v >= 0.999 or v <= 0.001):
          continue
        if y_val <= N_ruin:
          before.append((float(spend_ratio), y_val, v))
        else:
          after.append((float(spend_ratio), y_val, v))
    return before, after

  # --- オルカン比率のモデル ---
  print("--- 最適オルカン比率の Piecewise モデル計算 ---")
  points_ratio_before, points_ratio_after = get_training_points(
      summary_df, filter_boundary=True)
  feat_ratio_before, w_ratio_before = solve_piece(
      points_ratio_before, "Ratio Region 1 (N <= N_ruin)")
  feat_ratio_after, w_ratio_after = solve_piece(points_ratio_after,
                                                "Ratio Region 2 (N > N_ruin)")

  # --- 生存確率のモデル ---
  print("\n--- 最大生存確率の Piecewise モデル計算 ---")
  # 生存確率の場合、100% (1.0) の地点は予測において支配的なので、境界フィルタリングを適用
  points_prob_before, points_prob_after = get_training_points(
      prob_df, filter_boundary=True)

  feat_prob_before, w_prob_before = solve_piece(points_prob_before,
                                                "Prob Region 1 (N <= N_ruin)")
  feat_prob_after, w_prob_after = solve_piece(points_prob_after,
                                              "Prob Region 2 (N > N_ruin)")

  def predict_piecewise(S, N, fs, ws, default_before=1.0):
    N_ruin = get_ruin_year(S)
    if fs is None or ws is None:
      return default_before if N <= N_ruin else 0.0

    f_dict = get_features_for_piece(S, N, N_ruin)
    features_list = [1.0] + [f_dict.get(f, 0.0) for f in fs]
    raw = float(np.dot(ws, features_list))
    return max(0.0, min(1.0, raw))

  def predict_ratio(S, N):
    return predict_piecewise(
        S,
        N,
        feat_ratio_before if N <= get_ruin_year(S) else feat_ratio_after,
        w_ratio_before if N <= get_ruin_year(S) else w_ratio_after,
        default_before=1.0)

  def predict_prob(S, N):
    return predict_piecewise(
        S,
        N,
        feat_prob_before if N <= get_ruin_year(S) else feat_prob_after,
        w_prob_before if N <= get_ruin_year(S) else w_prob_after,
        default_before=1.0)

  def format_formula(fs, ws):
    if fs is None:
      return "1.0000 (Fixed)"
    form = f"{float(ws[0]):+.4f}"
    for i, f in enumerate(fs):
      form += f" {float(ws[i+1]):+.4f} * {f}"
    return form

  print("\n得られた近似式 (最適オルカン比率):")
  print(f"g_ratio(S, n) = {format_formula(feat_ratio_before, w_ratio_before)}")
  print(f"h_ratio(S, m) = {format_formula(feat_ratio_after, w_ratio_after)}")

  print("\n得られた近似式 (最大生存確率):")
  print(f"g_prob(S, n)  = {format_formula(feat_prob_before, w_prob_before)}")
  print(f"h_prob(S, m)  = {format_formula(feat_prob_after, w_prob_after)}")

  # サンプル比較
  print("\n--- サンプル比較 (実測値 vs 予測値) ---")
  sample_points = [(0.025, 20), (0.03, 30), (0.04, 15), (0.05, 30),
                   (0.066667, 20)]
  print(
      f"{'S':>8} | {'N':>3} | {'Ratio実':>7} | {'Ratio予':>7} | {'Prob実':>7} | {'Prob予':>7}"
  )
  print("-" * 70)
  for S, N in sample_points:
    if S not in summary_df.index:
      continue
    r_act = summary_df.loc[S, str(N)]
    r_pre = predict_ratio(S, float(N))
    p_act = prob_df.loc[S, str(N)]
    p_pre = predict_prob(S, float(N))
    print(
        f"{S:8.3f} | {N:3d} | {r_act:7.2f} | {r_pre:7.2f} | {p_act:7.2f} | {p_pre:7.2f}"
    )

  # --- 誤差テーブル (Piecewise) ---
  print("\n--- 誤差テーブル (Piecewise: 最適オルカン比率) ---")
  diff_rows = []
  for spend_ratio in summary_df.index:
    diff_row_data: Dict[str, Any] = {"spend_ratio": spend_ratio}
    for year in year_cols:
      actual_val = summary_df.loc[spend_ratio, year]
      predicted_val = predict_ratio(float(spend_ratio), float(year))
      diff_row_data[year] = round(predicted_val - actual_val, 2)
    diff_rows.append(diff_row_data)
  diff_df = pd.DataFrame(diff_rows)
  diff_df.set_index("spend_ratio", inplace=True)
  diff_df.index.name = "spend_ratio / year"
  print(diff_df)

  print("\n--- 誤差テーブル (Piecewise: 最大生存確率) ---")
  diff_rows_prob = []
  for spend_ratio in prob_df.index:
    row_err = {"spend_ratio": spend_ratio}
    for year in year_cols:
      actual_val = prob_df.loc[spend_ratio, year]
      predicted_val = predict_prob(float(spend_ratio), float(year))
      row_err[year] = round(predicted_val - actual_val, 2)
    diff_rows_prob.append(row_err)
  diff_df_prob = pd.DataFrame(diff_rows_prob)
  diff_df_prob.set_index("spend_ratio", inplace=True)
  diff_df_prob.index.name = "spend_ratio / year"
  print(diff_df_prob)
  print("\n")


if __name__ == "__main__":
  main()
