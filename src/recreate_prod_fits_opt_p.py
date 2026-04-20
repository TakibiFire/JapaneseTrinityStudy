import argparse
import json
import os

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate


def get_chart(age):
  path = f'temp/age_{age}.json'
  if not os.path.exists(path):
    print(f"File {path} not found.")
    return None

  with open(path, 'r') as f:
    data = json.load(f)

  # Production state (New PCHIP model)
  p_model = data['models']['p_survival']
  r_anchors = np.array(p_model['r_points'])
  p_anchors = np.array(p_model['p_points'])

  r_min = data['config']['r_min_p']
  r_max = data['config']['r_max_p']
  p_max = data['config']['p_max']
  p_min = data['config']['p_min']

  all_points = pd.DataFrame(data['all_points'])

  # 1. Recreate Production Model (PCHIP Spline)
  r_grid = np.geomspace(max(all_points['r'].min(), 0.005), all_points['r'].max(),
                        1000)

  p_pred = pchip_interpolate(r_anchors, p_anchors, r_grid)

  # Apply production guards
  p_pred_guarded = p_pred.copy()
  p_pred_guarded[r_grid <= r_min] = p_max
  p_pred_guarded[r_grid >= r_max] = p_min

  # Print out sampled values to STDOUT to inspect the fit
  print(f"\n--- Age {age} Model Stats ---")
  print(f"Guards: R_min={r_min:.4f}, R_max={r_max:.4f}")
  print(f"Number of anchor points: {len(r_anchors)}")

  print("\nAnchor Points & Predicted Values:")
  print("Anchor R | Observed P | Predicted P")
  # Use the training points for P to compare
  train_p = pd.DataFrame(data['training_points_p'])
  for i, row in train_p.iterrows():
    rv = float(row['r'])
    pv_obs = float(row['p_survival'])
    if rv <= r_min:
      pv_pred = p_max
    elif rv >= r_max:
      pv_pred = p_min
    else:
      pv_pred = float(pchip_interpolate(r_anchors, p_anchors, rv))
    print(f"R={rv:7.4f} | Obs={pv_obs:.6f} | Pred={pv_pred:.6f}")

  # DataFrames for plotting
  plot_prod = pd.DataFrame({
      'r': r_grid,
      'p': p_pred_guarded,
      'type': 'Prod (PCHIP + Guard)'
  })
  plot_actual = pd.DataFrame({
      'r': all_points['r'],
      'p': all_points['p_survival'],
      'type': 'Actual'
  })

  plot_df = pd.concat([plot_prod, plot_actual])

  chart_base = alt.Chart(plot_df).encode(
      x=alt.X('r:Q',
              scale=alt.Scale(type='log',
                              domain=[r_min * 0.5, r_max * 2.0],
                              clamp=True),
              title='R'),
      y=alt.Y('p:Q', title='P_surv'),
      color=alt.Color('type:N',
                      scale=alt.Scale(
                          domain=['Prod (PCHIP + Guard)', 'Actual'],
                          range=['#4c78a8', '#f58518'])))

  lines = chart_base.mark_line(opacity=0.8,
                               clip=True).transform_filter("datum.type != 'Actual'")
  points = chart_base.mark_circle(size=40,
                                  clip=True).transform_filter("datum.type == 'Actual'")

  return (lines + points).properties(title=f'Age {age} PCHIP Fit',
                                     width=350,
                                     height=200)


def parse_ages(age_str: str) -> list[int]:
  """
  Parse a string of ages like '91,93-95' into a list of integers [91, 93, 94, 95].
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
  parser.add_argument('--ages', type=str, help='Ages to process (e.g., 91,93-95)')
  args = parser.parse_args()

  if args.ages:
    ages = parse_ages(args.ages)
  else:
    # Usually we process 95, 94, 93, 92, 91
    ages = [95, 94, 93, 92, 91]

  charts = []
  for age in ages:
    c = get_chart(age)
    if c:
      charts.append(c)

  if charts:
    rows = []
    for i in range(0, len(charts), 2):
      rows.append(alt.hconcat(*charts[i:i + 2]))
    grid = alt.vconcat(*rows).resolve_scale(color='shared')
    grid.save('temp/recreate_prod_fits.svg')
    print("\nSuccessfully saved temp/recreate_prod_fits.svg")
