"""
生存確率（P_surv）モデルのフィッティング結果を JSON ファイルから読み込んで可視化するスクリプト。

使用例:
python src/visualize_dp_model_p.py \
  --json data/optimal_strategy_dp/experiments/re60_pen70_95_ar1_residual_legacy_no_shortcut_n1000.json \
  --ages 61,65,70,75,80,85,90,94
"""

import argparse
import json

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate


def get_chart_opt_p(age, json_data):
  """
  指定された年齢の生存確率モデルを可視化する。
  
  Args:
    age: 対象年齢。
    json_data: モデルパラメータが含まれる辞書。
     
  Returns:
    alt.Chart: Altair のチャートオブジェクト。
  """
  age_key = str(age)
  if age_key not in json_data:
    print(f"Age {age} not found in JSON.")
    return None

  age_data = json_data[age_key]
  if 'p_survival_model' not in age_data:
    print(f"Age {age} does not have p_survival_model.")
    return None

  p_model = age_data['p_survival_model']
  r_pts = np.array(p_model['r_points'])
  p_pts = np.array(p_model['p_points'])

  r_min = p_model['r_min_p']
  r_max = p_model['r_max_p']
  p_min = age_data.get('p_min', 0.0)
  p_max = age_data.get('p_max', 1.0)

  # 表示用の R グリッドを作成
  r_start = r_min * 0.5
  r_end = r_max * 2.0
  r_grid = np.geomspace(r_start, r_end, 1000)

  # PCHIP 補間による予測
  p_pred_raw = pchip_interpolate(r_pts, p_pts, r_grid)
  p_pred_raw = np.clip(p_pred_raw, 0.0, 1.0)

  # ガードの適用
  p_pred_guarded = p_pred_raw.copy()
  p_pred_guarded[r_grid <= r_min] = p_max
  p_pred_guarded[r_grid >= r_max] = p_min

  # プロット用データの準備
  plot_raw = pd.DataFrame({
      'r': r_grid,
      'p': p_pred_raw,
      'type': 'PCHIP Raw'
  })
  plot_guarded = pd.DataFrame({
      'r': r_grid,
      'p': p_pred_guarded,
      'type': 'Guarded'
  })
  # アンカーポイント
  plot_anchors = pd.DataFrame({
      'r': r_pts,
      'p': p_pts,
      'type': 'Anchors'
  })

  chart_lines = alt.Chart(pd.concat([plot_raw, plot_guarded])).mark_line(
      clip=True).encode(
          x=alt.X('r:Q',
                  scale=alt.Scale(type='log', domain=[r_start, r_end],
                                  clamp=True),
                  title='R (Spending Rate)'),
          y=alt.Y('p:Q', title='P_surv', scale=alt.Scale(domain=[-0.1, 1.1])),
          color=alt.Color('type:N',
                          scale=alt.Scale(domain=['PCHIP Raw', 'Guarded'],
                                          range=['#4c78a8', '#e45756'])))

  chart_anchors = alt.Chart(plot_anchors).mark_point(
      size=60, color='black', symbol='diamond').encode(x='r:Q', y='p:Q')

  # R_min/R_max の垂直線を追加
  rules_df = pd.DataFrame({
      'r': [r_min, r_max],
      'color': ['blue', 'blue'],
      'name': ['R_min_P', 'R_max_P']
  })
  rules = alt.Chart(rules_df).mark_rule(opacity=0.5, strokeDash=[4, 4]).encode(
      x='r:Q', color=alt.Color('color:N', scale=None))

  return (chart_lines + chart_anchors + rules).properties(
      title=f'Age {age} P_surv Fit', width=400, height=250)


def parse_ages(age_str: str) -> list[int]:
  """
  年齢指定文字列をパースする（例: '61,65-70' -> [61, 65, 66, 67, 68, 69, 70]）。
  """
  ages = []
  for part in age_str.split(','):
    if '-' in part:
      start_s, end_s = part.split('-')
      start, end = int(start_s), int(end_s)
      if start <= end:
        ages.extend(range(start, end + 1))
      else:
        ages.extend(range(start, end - 1, -1))
    else:
      ages.append(int(part))
  return ages


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description='Recreate production fits for P_surv model.')
  parser.add_argument('--ages', type=str, help='Ages to process (e.g., 61,65-70)')
  parser.add_argument('--json',
                      type=str,
                      required=True,
                      help='Path to production JSON model')
  args = parser.parse_args()

  with open(args.json, 'r') as f:
    json_data = json.load(f)

  if args.ages:
    ages = parse_ages(args.ages)
  else:
    # データの存在する年齢を自動取得してソート
    ages = sorted([int(k) for k in json_data.keys() if k.isdigit()],
                  reverse=True)
    # デフォルトでは最新の6件程度にする
    ages = ages[:6]

  charts = []
  for age in ages:
    c = get_chart_opt_p(age, json_data)
    if c:
      charts.append(c)

  if charts:
    # 2列に並べる
    rows = []
    for i in range(0, len(charts), 2):
      rows.append(alt.hconcat(*charts[i:i + 2]))
    grid = alt.vconcat(*rows).resolve_scale(color='shared')
    output_path = 'temp/visualize_dp_model_p.svg'
    grid.save(output_path)
    print(f"Saved {output_path}")
