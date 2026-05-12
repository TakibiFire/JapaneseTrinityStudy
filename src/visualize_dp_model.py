"""
最適資産配分（A_opt）および生存確率（P_surv）モデルのフィッティング結果を
JSON ファイルから読み込んで可視化するスクリプト。

使用例:
python src/visualize_dp_model.py \
  --json data/optimal_strategy_dp/re60_pen70_95.json \
  --ages 60,65,70,75,80,85,94
"""

import argparse
import json
import os

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate


def get_chart_opt_a(age, json_data):
  """
  指定された年齢の最適資産配分モデルを可視化する。
  
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
  if 'a_opt_model' not in age_data:
    print(f"Age {age} does not have a_opt_model.")
    return None

  a_model = age_data['a_opt_model']
  r_pts = np.array(a_model['r_points'])
  a_pts = np.array(a_model['a_points'])

  r_min = a_model['r_min_a']
  r_max = a_model['r_max_a']

  # グラフの表示範囲のための P 境界（ない場合は A 境界を使用）
  p_model = age_data.get('p_survival_model', {})
  r_min_p = p_model.get('r_min_p', r_min)
  r_max_p = p_model.get('r_max_p', r_max)

  # 表示用の R グリッドを作成
  r_start = min(r_min, r_min_p) * 0.5
  r_end = max(r_max, r_max_p) * 2.0
  r_grid = np.geomspace(r_start, r_end, 1000)

  # PCHIP 補間による予測
  a_pred_raw = pchip_interpolate(r_pts, a_pts, r_grid)
  a_pred_raw = np.clip(a_pred_raw, 0.0, 1.0)

  # ガードの適用 (r_min, r_max の外側は 1.0 固定)
  a_pred_guarded = a_pred_raw.copy()
  a_pred_guarded[r_grid <= r_min] = 1.0
  a_pred_guarded[r_grid >= r_max] = 1.0

  # プロット用データの準備
  plot_raw = pd.DataFrame({'r': r_grid, 'a': a_pred_raw, 'type': 'PCHIP Raw'})
  plot_guarded = pd.DataFrame({
      'r': r_grid,
      'a': a_pred_guarded,
      'type': 'Guarded'
  })

  chart_lines = alt.Chart(pd.concat(
      [plot_raw, plot_guarded])).mark_line(clip=True).encode(
          x=alt.X('r:Q',
                  scale=alt.Scale(type='log',
                                  domain=[r_start, r_end],
                                  clamp=True),
                  title='R (Spending Rate)'),
          y=alt.Y('a:Q',
                  title='Opt A (Equity Ratio)',
                  scale=alt.Scale(domain=[0, 1.1])),
          color=alt.Color('type:N',
                          scale=alt.Scale(domain=['PCHIP Raw', 'Guarded'],
                                          range=['#4c78a8', '#e45756'])))

  # R_min/R_max の垂直線を追加
  rules_df = pd.DataFrame({
      'r': [r_min_p, r_max_p, r_min, r_max],
      'color': ['blue', 'blue', 'red', 'red'],
      'name': ['R_min_P', 'R_max_P', 'R_min_A', 'R_max_A']
  })
  rules = alt.Chart(rules_df).mark_rule(opacity=0.5, strokeDash=[4, 4]).encode(
      x='r:Q', color=alt.Color('color:N', scale=None))

  return (chart_lines + rules).properties(title=f'Age {age} Opt A Fit',
                                          width=400,
                                          height=250)


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
  plot_raw = pd.DataFrame({'r': r_grid, 'p': p_pred_raw, 'type': 'PCHIP Raw'})
  plot_guarded = pd.DataFrame({
      'r': r_grid,
      'p': p_pred_guarded,
      'type': 'Guarded'
  })
  # アンカーポイント
  plot_anchors = pd.DataFrame({'r': r_pts, 'p': p_pts, 'type': 'Anchors'})

  chart_lines = alt.Chart(pd.concat(
      [plot_raw, plot_guarded])).mark_line(clip=True).encode(
          x=alt.X('r:Q',
                  scale=alt.Scale(type='log',
                                  domain=[r_start, r_end],
                                  clamp=True),
                  title='R (Spending Rate)'),
          y=alt.Y('p:Q', title='P_surv', scale=alt.Scale(domain=[-0.1, 1.1])),
          color=alt.Color('type:N',
                          scale=alt.Scale(domain=['PCHIP Raw', 'Guarded'],
                                          range=['#4c78a8', '#e45756'])))

  chart_anchors = alt.Chart(plot_anchors).mark_point(size=60,
                                                     color='black',
                                                     shape='diamond').encode(
                                                         x='r:Q', y='p:Q')

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
      description='Visualize DP model fits (A and P).')
  parser.add_argument('--ages',
                      type=str,
                      help='Ages to process (e.g., 61,65-70)')
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

  rows = []
  for age in ages:
    chart_a = get_chart_opt_a(age, json_data)
    chart_p = get_chart_opt_p(age, json_data)

    if chart_a and chart_p:
      # A と P を横に並べる
      rows.append(alt.hconcat(chart_a, chart_p))
    elif chart_a:
      rows.append(chart_a)
    elif chart_p:
      rows.append(chart_p)

  if rows:
    # 各行を縦に並べる
    grid = alt.vconcat(*rows).resolve_scale(color='shared')
    json_basename = os.path.splitext(os.path.basename(args.json))[0]
    output_path = f'temp/visualize_{json_basename}.svg'
    grid.save(output_path)
    print(f"Saved {output_path}")
