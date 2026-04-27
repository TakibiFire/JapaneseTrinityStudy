import argparse
import json
import os

import altair as alt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate


def get_chart_opt_a(age):
  path = f'temp/age_{age}.json'
  if not os.path.exists(path):
    print(f"File {path} not found.")
    return None

  with open(path, 'r') as f:
    data = json.load(f)

  a_model = data['models']['a_optimal']
  r_pts = np.array(a_model['r_points'])
  a_pts = np.array(a_model['a_points'])

  r_min = data['config']['r_min_a']
  r_max = data['config']['r_max_a']
  r_min_p = data['config']['r_min_p']
  r_max_p = data['config']['r_max_p']

  all_points = pd.DataFrame(data['all_points'])

  # Grid for prediction
  r_grid = np.geomspace(max(all_points['r'].min(), 0.005),
                        all_points['r'].max(), 1000)

  # Raw PCHIP prediction
  a_pred_raw = pchip_interpolate(r_pts, a_pts, r_grid)
  a_pred_raw = np.clip(a_pred_raw, 0.0, 1.0)

  # Guarded prediction (r_min guard)
  a_pred_guarded = a_pred_raw.copy()
  a_pred_guarded[r_grid <= r_min] = 1.0
  a_pred_guarded[r_grid >= r_max] = 1.0

  # Print out sampled values to STDOUT to inspect the fit
  print(f"\n--- Age {age} A_opt Model Stats ---")
  print(f"A Guards: R_min_A={r_min:.4f}, R_max_A={r_max:.4f}")
  print(f"P Guards: R_min_P={r_min_p:.4f}, R_max_P={r_max_p:.4f}")

  print("\nSampled R | Predicted A (Guarded)")
  # Sample points from the grid
  sample_indices = np.linspace(0, len(r_grid) - 1, 15, dtype=int)
  for idx in sample_indices:
    r_val = r_grid[idx]
    a_val = a_pred_guarded[idx]
    print(f"R={r_val:7.4f} | A={a_val:.6f}")

  print("\nActual Points | [Min A, Max A] | Predicted A (Guarded)")
  # Sample actual points (e.g. up to 20 points around the transition)
  actual_sampled = all_points[(all_points['r'] >= r_min_p * 0.5) &
                              (all_points['r'] <= r_max_p * 2.0)]
  for i, row in actual_sampled.reset_index().iterrows():
    rv = float(row['r'])
    # Calculate prediction for this specific RV
    if rv <= r_min or rv >= r_max:
      ap_guarded = 1.0
    else:
      ap_guarded = float(np.clip(pchip_interpolate(r_pts, a_pts, rv), 0.0, 1.0))

    print(
        f"R={rv:7.4f} | [{row['a_opt_min']:.2f}, {row['a_opt_max']:.2f}] | Pred={ap_guarded:.6f}"
    )

  # DataFrames for plotting
  plot_raw = pd.DataFrame({
      'r': r_grid,
      'a': a_pred_raw,
      'type': 'Prod (PCHIP Raw)'
  })
  plot_guarded = pd.DataFrame({
      'r': r_grid,
      'a': a_pred_guarded,
      'type': 'Prod (Guarded)'
  })

  # Indifference range (a_opt_min ~ a_opt_max)
  range_df = all_points[['r', 'a_opt_min', 'a_opt_max']].copy()

  # Actual a_opt points
  plot_actual = pd.DataFrame({
      'r': all_points['r'],
      'a': all_points['a_opt_max'],
      'type': 'Actual Max A'
  })

  chart_range = alt.Chart(range_df).mark_area(
      opacity=0.2, color='red', clip=True).encode(x=alt.X(
          'r:Q',
          scale=alt.Scale(type='log',
                          domain=[r_min_p * 0.5, r_max_p * 2.0],
                          clamp=True),
          title='R'),
                                                  y='a_opt_min:Q',
                                                  y2='a_opt_max:Q')

  chart_lines = alt.Chart(pd.concat(
      [plot_raw, plot_guarded])).mark_line(clip=True).encode(
          x=alt.X('r:Q',
                  scale=alt.Scale(type='log',
                                  domain=[r_min_p * 0.5, r_max_p * 2.0],
                                  clamp=True)),
          y=alt.Y('a:Q', title='Opt A'),
          color=alt.Color('type:N',
                          scale=alt.Scale(
                              domain=['Prod (PCHIP Raw)', 'Prod (Guarded)'],
                              range=['#4c78a8', '#e45756'])))

  chart_points = alt.Chart(plot_actual).mark_circle(size=40,
                                                    color='black',
                                                    clip=True).encode(x='r:Q',
                                                                      y='a:Q')

  # Add R_min/R_max vertical lines
  rule_min_p = alt.Chart(pd.DataFrame({'r': [r_min_p]
                                      })).mark_rule(strokeDash=[4, 4],
                                                    color='blue',
                                                    opacity=0.5).encode(x='r:Q')
  rule_max_p = alt.Chart(pd.DataFrame({'r': [r_max_p]
                                      })).mark_rule(strokeDash=[4, 4],
                                                    color='blue',
                                                    opacity=0.5).encode(x='r:Q')
  rule_min_a = alt.Chart(pd.DataFrame({'r': [r_min]
                                      })).mark_rule(strokeDash=[2, 2],
                                                    color='red',
                                                    opacity=0.5).encode(x='r:Q')
  rule_max_a = alt.Chart(pd.DataFrame({'r': [r_max]
                                      })).mark_rule(strokeDash=[2, 2],
                                                    color='red',
                                                    opacity=0.5).encode(x='r:Q')

  return (chart_range + chart_lines + chart_points + rule_min_p + rule_max_p +
          rule_min_a + rule_max_a).properties(
              title=f'Age {age} Opt A Fit (PCHIP)', width=400, height=250)


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
      description='Recreate production fits for A_opt model.')
  parser.add_argument('--ages',
                      type=str,
                      help='Ages to process (e.g., 91,93-95)')
  args = parser.parse_args()

  if args.ages:
    ages = parse_ages(args.ages)
  else:
    # Usually we process 95, 94, 93, 92, 91
    ages = [95, 94, 93, 92, 91]

  charts = []
  for age in ages:
    c = get_chart_opt_a(age)
    if c:
      charts.append(c)

  if charts:
    # Group in 2 columns
    rows = []
    for i in range(0, len(charts), 2):
      rows.append(alt.hconcat(*charts[i:i + 2]))
    grid = alt.vconcat(*rows).resolve_scale(color='shared')
    grid.save('temp/recreate_prod_fits_opt_a.svg')
    print("Saved temp/recreate_prod_fits_opt_a.svg")
