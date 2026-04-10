"""
このスクリプトは、data/withdrawal_rate_grid_comp.csv を読み込み、
各支出率と各経過年数において生存確率を最大化するオルカン比率を特定し、
表形式およびグラフ（SVG）で出力します。
また、得られた近似式（Piecewise モデル）に基づく最適な比率の推移も可視化します。

出力ファイル:
- data/optimal_orukan_ratio.csv: 各条件での最適オルカン比率
- data/max_survival_probability.csv: 各条件での最大生存確率
- docs/imgs/dynamic_rebalance/optimal_orukan_ratio.svg: 最適比率の推移グラフ (実測)
- docs/imgs/dynamic_rebalance/fitted_optimal_orukan_ratio.svg: 最適比率の推移グラフ (近似式)

近似式の計算手法:
1. 資産寿命 N_ruin (無リスク資産のみの場合) を境界としてデータを2つのリージョンに分割。
2. 各リージョンに対して、支出率 S と経過年数 n (正規化済み) を用いた非線形特徴量を生成。
3. ステップワイズ選択により、生存確率（または最適比率）を最もよく説明する特徴量を4つ選択。
4. L2正則化付きの最小二乗法および L-BFGS-B 最適化を用いて、[0, 1] にクリップされた近似式の係数を決定。
   - フィッティング時、N_ruin 付近の精度を高めるために境界付近のデータに高い重みを設定しています。

最適オルカン比率の決定（ノイズ除去）:
- 標本誤差によるガタつきを抑えるため、最大生存確率から一定範囲内（0.001以内）にある比率の中から、前年の比率に最も近いものを選択することで、時間軸方向の滑らかさを確保しています。

設定詳細 (N_ruin 計算用):
- 無リスク資産利回り: 4%
- 税率: 20.315%
- インフレ率: 1.77%
- 解析対象期間: 60年
- 支出率: 2% 〜 6.67%
- 目標生存確率の正規化定数: 60年
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import scipy.optimize as opt


def main():
  # CSVデータの読み込み
  csv_path = "data/withdrawal_rate_grid_comp.csv"
  if not os.path.exists(csv_path):
    print(
        f"Error: {csv_path} が見つかりません。先に src/withdrawal_rate_grid_comp.py を実行してください。"
    )
    return

  df = pd.read_csv(csv_path)

  # 経過年数のカラム（"1"〜"60"）を特定
  years_range = 60
  year_cols = [str(y) for y in range(1, years_range + 1)]

  # 支出率ごとに処理
  spend_ratios = sorted(df['spend_ratio'].unique())

  ratio_summary_rows = []
  prob_summary_rows = []

  for spend_ratio in spend_ratios:
    df_spend = df[df['spend_ratio'] == spend_ratio].copy()

    # 行データを作成
    ratio_row: Dict[str, Any] = {"spend_ratio": spend_ratio}
    prob_row: Dict[str, Any] = {"spend_ratio": spend_ratio}

    last_ratio = 1.0  # 初期値は100%から開始すると仮定

    for year in year_cols:
      # その年において最大生存確率を取得
      max_val = df_spend[year].max()

      # 最大値から許容範囲内（0.001）の比率を抽出
      tolerance = 0.001
      best_candidates = df_spend[df_spend[year] >= (max_val - tolerance)].copy()

      # 前年の比率に最も近いものを選択
      best_candidates['dist'] = (best_candidates['orukan_ratio'] -
                                 last_ratio).abs()
      best_row = best_candidates.sort_values(
          by=['dist', 'orukan_ratio']).iloc[0]

      # オルカン比率と生存確率を格納
      last_ratio = best_row['orukan_ratio']
      ratio_row[year] = last_ratio
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
  print(summary_df.head())

  print("\n各支出率と経過年数における「最大生存確率」:")
  print(prob_df.head())

  # CSVとして保存
  os.makedirs("data", exist_ok=True)
  output_ratio_csv = "data/optimal_orukan_ratio.csv"
  summary_df.to_csv(output_ratio_csv)
  output_prob_csv = "data/max_survival_probability.csv"
  prob_df.to_csv(output_prob_csv)
  print(f"✅ {output_ratio_csv} および {output_prob_csv} に保存しました。")

  # --- 資産寿命（無リスク資産のみ）の計算 ---
  def get_ruin_year(S: float) -> float:
    # 実際のシミュレーション設定に基づく定数
    r_base = 0.04
    tax = 0.20315
    inflation = 0.0177  # Simulation parameters

    r_eff = r_base * (1.0 - tax)
    i_ln = np.log(1.0 + inflation)
    delta = r_eff - i_ln

    if S <= delta:
      return 999.0  # 支出が実質利回り以下なら理論上無限

    # P(t) = 0 となる t を解く
    return np.log(1.0 - delta / S) / (-delta)

  def get_features_for_piece(S: float, N_raw: float,
                             N_ruin: float) -> Dict[str, float]:
    n = N_raw / 60.0  # 正規化 (Duration is 60 years)
    m = (N_raw - N_ruin) / 60.0  # 資産寿命からの超過年数 (Region 2用)

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

  def solve_piece(
      points: List[Tuple[float, float, float]],
      name: str,
      num_terms: int = 5,
      weights: Optional[np.ndarray] = None) -> Tuple[List[str], np.ndarray]:
    if not points:
      return [], np.zeros(0)

    # 各点において N_ruin を再計算
    X_dict_list = [
        get_features_for_piece(S, float(N), get_ruin_year(S))
        for S, N, v in points
    ]
    Y_arr = np.array([v for S, N, v in points])

    if weights is None:
      weights = np.ones(len(Y_arr))

    feature_names = list(X_dict_list[0].keys())

    selected: List[str] = []
    best_w: np.ndarray = np.zeros(1)

    def loss_func(w, X, Y, W):
      raw = X.dot(w)
      clamped = np.clip(raw, 0.0, 1.0)
      # L2 正則化を追加 (係数が巨大化するのを防ぐ) + 重み付き二乗誤差
      # ※ Y=1.0 または Y=0.0 の境界データに対して、clamped を用いることで、
      #    raw >= 1.0 または raw <= 0.0 の場合は誤差0 (acceptable) として扱い、
      #    過剰なペナルティを防ぐ。
      return np.sum(W * (clamped - Y)**2) + 0.01 * np.sum(w**2)

    print(f"\n[{name}] のステップワイズ選択:")
    for step in range(num_terms):
      best_feature = ""
      best_error = float('inf')
      current_best_w = np.zeros(1)

      for f in feature_names:
        if f in selected:
          continue

        current_fs = selected + [f]
        # X行列の構築 (先頭は定数項の1.0)
        X_mat = np.zeros((len(X_dict_list), len(current_fs) + 1))
        X_mat[:, 0] = 1.0
        for i, d in enumerate(X_dict_list):
          for j, cf in enumerate(current_fs):
            X_mat[i, j + 1] = d.get(cf, 0.0)

        # 最小二乗法で初期値を得る
        w0, _, _, _ = np.linalg.lstsq(X_mat, Y_arr, rcond=None)

        # scipyでクリッピングを考慮した最適化
        res = opt.minimize(loss_func,
                           w0,
                           args=(X_mat, Y_arr, weights),
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

  def get_training_data(data_df: pd.DataFrame, filter_boundary: bool = False):
    before, after = [], []
    weights_b, weights_a = [], []
    for spend_ratio in data_df.index:
      row = data_df.loc[spend_ratio]
      vals = row.values
      N_ruin = get_ruin_year(float(spend_ratio))
      for i in range(len(vals)):
        y_val = i + 1
        v = float(vals[i])
        
        # 境界データ (1.0 や 0.0) の除外フラグ
        if filter_boundary and (v >= 0.999 or v <= 0.001):
          continue
        
        # N_ruin 付近のデータに重みを付ける (V字の底を正確に捉えるため)
        dist = abs(y_val - N_ruin)
        # N_ruin 付近の重みを指数関数的に強化 (1.0 + 10.0*exp(-dist))
        # これにより多数の中間データに埋もれることなく、N_ruin の境界条件を重視する。
        weight = 1.0 + 10.0 * np.exp(-dist / 1.0)

        if y_val <= N_ruin:
          before.append((float(spend_ratio), float(y_val), v))
          weights_b.append(weight)
        else:
          after.append((float(spend_ratio), float(y_val), v))
          weights_a.append(weight)
    return before, after, np.array(weights_b), np.array(weights_a)

  # --- オルカン比率のモデル ---
  print("--- 最適オルカン比率の Piecewise モデル計算 ---")
  # 1.0(100%) や 0.0(0%) といった境界データは以前は除外していましたが、
  # np.clip による loss_func が「1.0以上の予測は許容(loss=0)」として正しく機能するため、
  # 境界データも含めて全て学習に用いる。これにより Region の切り替わりが正確になる。
  points_ratio_before, points_ratio_after, w_ratio_b, w_ratio_a = get_training_data(
      summary_df, filter_boundary=False)
  feat_ratio_before, w_ratio_before = solve_piece(
      points_ratio_before, "Ratio Region 1 (N <= N_ruin)", weights=w_ratio_b)
  feat_ratio_after, w_ratio_after = solve_piece(points_ratio_after,
                                                "Ratio Region 2 (N > N_ruin)",
                                                weights=w_ratio_a)

  # --- 生存確率のモデル ---
  print("\n--- 最大生存確率の Piecewise モデル計算 ---")
  # 生存確率の場合も同様に境界データを含めて学習し、連続性を保つ。
  points_prob_before, points_prob_after, w_prob_b, w_prob_a = get_training_data(
      prob_df, filter_boundary=False)

  feat_prob_before, w_prob_before = solve_piece(points_prob_before,
                                                "Prob Region 1 (N <= N_ruin)",
                                                weights=w_prob_b)
  feat_prob_after, w_prob_after = solve_piece(points_prob_after,
                                              "Prob Region 2 (N > N_ruin)",
                                              weights=w_prob_a)

  def predict_piecewise(S, N, fs, ws, default_before=1.0):
    N_ruin = get_ruin_year(S)
    if not fs:
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
    if not fs:
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

  # --- 可視化 (実測値 vs 近似式) ---
  img_dir = "docs/imgs/dynamic_rebalance"
  os.makedirs(img_dir, exist_ok=True)

  # 1. 実測値のグラフ
  plot_df_raw = summary_df.reset_index().melt(id_vars="spend_ratio / year",
                                              var_name="year",
                                              value_name="optimal_ratio")
  plot_df_raw.columns = ["spend_ratio", "year", "optimal_ratio"]  # type: ignore
  plot_df_raw["year"] = plot_df_raw["year"].astype(int)
  plot_df_raw["支出率"] = (plot_df_raw["spend_ratio"] *
                        100).map(lambda x: f"{x:.2f}%")

  chart_raw = alt.Chart(plot_df_raw).mark_line().encode(
      x=alt.X("year:Q", title="目標寿命 (年)"),
      y=alt.Y("optimal_ratio:Q",
              title="最適オルカン比率",
              scale=alt.Scale(domain=[0, 1])),
      color=alt.Color("支出率:N",
                      sort=alt.SortField("spend_ratio", order="ascending"),
                      scale=alt.Scale(scheme='category10')),
      tooltip=[
          alt.Tooltip("支出率:N"),
          alt.Tooltip("year:Q", title="目標寿命 (年)"),
          alt.Tooltip("optimal_ratio:Q", title="最適比率", format=".2f")
      ]).properties(title="支出率別の最適オルカン比率の推移 (実測)", width=600, height=400)

  chart_file_raw = os.path.join(img_dir, "optimal_orukan_ratio.svg")
  chart_raw.save(chart_file_raw)
  print(f"✅ {chart_file_raw} に保存しました。")

  # 2. 近似式のグラフ
  fitted_rows = []
  for S in spend_ratios:
    for y in range(1, 61):
      fitted_rows.append({
          "spend_ratio": S,
          "year": y,
          "optimal_ratio": predict_ratio(S, float(y))
      })
  plot_df_fitted = pd.DataFrame(fitted_rows)
  plot_df_fitted["支出率"] = (plot_df_fitted["spend_ratio"] *
                           100).map(lambda x: f"{x:.2f}%")

  chart_fitted = alt.Chart(plot_df_fitted).mark_line().encode(
      x=alt.X("year:Q", title="目標寿命 (年)"),
      y=alt.Y("optimal_ratio:Q",
              title="最適オルカン比率 (近似式)",
              scale=alt.Scale(domain=[0, 1])),
      color=alt.Color("支出率:N",
                      sort=alt.SortField("spend_ratio", order="ascending"),
                      scale=alt.Scale(scheme='category10')),
      tooltip=[
          alt.Tooltip("支出率:N"),
          alt.Tooltip("year:Q", title="目標寿命 (年)"),
          alt.Tooltip("optimal_ratio:Q", title="最適比率 (近似)", format=".2f")
      ]).properties(title="近似式による最適オルカン比率の推移 (60年)", width=600, height=400)

  chart_file_fitted = os.path.join(img_dir, "fitted_optimal_orukan_ratio.svg")
  chart_fitted.save(chart_file_fitted)
  print(f"✅ {chart_file_fitted} に保存しました。")

  # サンプル比較 (STDOUT)
  print("\n--- サンプル比較 (実測値 vs 予測値) ---")
  sample_points = [(0.02, 50), (0.03, 30), (0.04, 15), (0.05, 30),
                   (0.066667, 16)]
  print(
      f"{'S':>8} | {'N':>3} | {'Ratio実':>7} | {'Ratio予':>7} | {'Prob実':>7} | {'Prob予':>7}"
  )
  print("-" * 70)
  for S_val, N_val in sample_points:
    # Use np.isclose to find matching spend_ratio index reliably
    match_s = next((idx for idx in summary_df.index if np.isclose(idx, S_val)),
                   None)
    if match_s is None:
      continue
    r_act = summary_df.at[match_s, str(N_val)]
    r_pre = predict_ratio(match_s, float(N_val))
    p_act = prob_df.at[match_s, str(N_val)]
    p_pre = predict_prob(match_s, float(N_val))
    print(
        f"{match_s:8.3f} | {N_val:3d} | {r_act:7.2f} | {r_pre:7.2f} | {p_act:7.2f} | {p_pre:7.2f}"
    )


if __name__ == "__main__":
  main()
