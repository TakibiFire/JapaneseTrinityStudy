import os
import re
import sys
from dataclasses import replace
from typing import Any, Dict, List, Union, cast

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

from src.core import simulate_strategy
from src.lib.dp_predictor import DPOptimalStrategyPredictor, WinThresholdType
from src.lib.retired_spending import (SpendingType,
                                      get_annual_retired_spending_values)
from src.lib.scenario_builder import (PredefinedStock, PredefinedZeroRisk,
                                      SpendAwareDPRebalance,
                                      create_experiment_setup)
from src.lib.world_setup import re60_pen70_95

YEARS = 35
START_AGE = 60
SEED = 43
N_SIM_EVAL = 1000
SPENDING_RULE = 7

LHS = {
    "model_path":
        "data/optimal_strategy_dp/experiments/re60_pen70_95_tblegacy_miny0_n1000.json",
    "name":
        "Legacy (miny=0)/V2_90",
    "win_threshold_type":
        WinThresholdType.V2_90
}
RHS = {
    "model_path":
        "data/optimal_strategy_dp/experiments/re60_pen70_95_tblegacy_miny5_n1000.json",
    "name":
        "Legacy (miny=5)/V2_90",
    "win_threshold_type":
        WinThresholdType.V2_90
}


def compile_and_run():
  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      60, 1)[0]

  exp_setup = re60_pen70_95(n_sim=N_SIM_EVAL, seed=SEED)
  exp_setup.name = "experimental_dp_eval"
  init_money = base_spend_annual / (SPENDING_RULE / 100.0)

  exp_setup.add_experiment(
      name=f"LHS: {LHS['name']}",
      overwrite_strategy=replace(
          exp_setup.strategy,
          initial_money=float(init_money),
          rebalance=SpendAwareDPRebalance(
              risky_asset=PredefinedStock.ORUKAN_155,
              zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
              model_name=LHS["model_path"],
              win_threshold_type=LHS["win_threshold_type"])))

  exp_setup.add_experiment(
      name=f"RHS: {RHS['name']}",
      overwrite_strategy=replace(
          exp_setup.strategy,
          initial_money=float(init_money),
          rebalance=SpendAwareDPRebalance(
              risky_asset=PredefinedStock.ORUKAN_155,
              zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
              model_name=RHS["model_path"],
              win_threshold_type=RHS["win_threshold_type"])))

  compiled = create_experiment_setup(exp_setup, record_annual_spend=False)
  return compiled[1], compiled[2]


def parse_log(log_str):
  # cur_ann_spend=2.30, prev_ann_spend=502.35, rem_years=..., cpi=1.0000, prev_cpi=1.0000, pY_N=..., P_pred=..., W_N=...
  m_age = re.search(r"Age (\d+)", log_str)
  m_spend = re.search(r"cur_ann_spend=([\d\.]+)", log_str)
  m_prev = re.search(r"prev_ann_spend=([\d\.]+)", log_str)
  m_net = re.search(r"total_net=([\d\.]+)", log_str)
  m_post = re.search(r"post_tax_net=([\d\.]+)", log_str)
  m_cpi = re.search(r"cpi=([\d\.]+)", log_str)
  m_prev_cpi = re.search(r"prev_cpi=([\d\.]+)", log_str)
  m_price = re.search(r"ORUKAN_155: ([\d\.]+)", log_str)
  m_a = re.search(r"'ORUKAN_155': [^\(]*\(([\d\.]+)\)", log_str)
  if not m_a:
    m_a = re.search(r"'ORUKAN_155': ([\d\.]+)", log_str)

  m_pyn = re.search(r"pY_N=([\d\.]+)", log_str)
  m_ppred = re.search(r"P_pred=([\d\.]+)", log_str)
  m_wn = re.search(r"W_N=([\d\.]+)", log_str)

  age = int(m_age.group(1)) if m_age else 0
  net = float(m_net.group(1)) if m_net else 0.0
  post = float(m_post.group(1)) if m_post else 0.0
  spend = float(m_spend.group(1)) if m_spend else 0.0
  prev_spend = float(m_prev.group(1)) if m_prev else 0.0
  cpi = float(m_cpi.group(1)) if m_cpi else 1.0

  pred_y = float(m_pyn.group(1)) if m_pyn else 0.0
  p_val = float(m_ppred.group(1)) if m_ppred else 0.0
  w_n = float(m_wn.group(1)) if m_wn else 0.0

  return {
      'age': age,
      'spend': spend,
      'pred_y': pred_y,
      'net': net,
      'post': post,
      'prev_spend': prev_spend,
      'cpi': cpi,
      'price': float(m_price.group(1)) if m_price else 0.0,
      'a': float(m_a.group(1)) if m_a else 0.0,
      'p': p_val,
      'w_n': w_n
  }


def main():
  exp_v1, exp_v2 = compile_and_run()

  pred_v1 = DPOptimalStrategyPredictor(
      LHS["model_path"], win_threshold_type=LHS["win_threshold_type"])
  pred_v2 = DPOptimalStrategyPredictor(
      RHS["model_path"], win_threshold_type=RHS["win_threshold_type"])

  res_v1 = simulate_strategy(exp_v1.strategy, exp_v1.monthly_prices,
                             exp_v1.monthly_cashflows)
  res_v2 = simulate_strategy(exp_v2.strategy, exp_v2.monthly_prices,
                             exp_v2.monthly_cashflows)

  surv_v1 = 1.0 - (res_v1.sustained_months < YEARS * 12).mean()
  surv_v2 = 1.0 - (res_v2.sustained_months < YEARS * 12).mean()
  print(f"Overall Survival Rates: LHS={surv_v1:.4f}, RHS={surv_v2:.4f}")

  # LHS が勝つケース (LHS が完走し、RHS が破産)
  win_v1 = np.where((res_v1.sustained_months == YEARS * 12) &
                    (res_v2.sustained_months < YEARS * 12))[0]
  # RHS が勝つケース (RHS が完走し、LHS が破産)
  win_v2 = np.where((res_v2.sustained_months == YEARS * 12) &
                    (res_v1.sustained_months < YEARS * 12))[0]

  selected_v1 = win_v1[:2].tolist()
  selected_v2 = win_v2[:2].tolist()
  selected = selected_v1 + selected_v2

  if not selected:
    selected = [0, 1, 2, 3]
    print(f"No paths with different outcomes. Selected indices: {selected}")
  else:
    print(f"Selected LHS win paths: {selected_v1}")
    print(f"Selected RHS win paths: {selected_v2}")

  print("\nColumn Definitions:")
  print("  X_N    : Total net assets at the start of the age (post-tax estimate).")
  print("  pY_N   : Predicted net withdrawal for the upcoming year (used for DP planning).")
  print("  Y_prev : Actual net withdrawal of the previous year (observed by the simulator).")
  print("  W_N    : Winning Threshold (Minimum assets required for a 'safe' outcome).")
  print("  A      : Optimal risky asset (ORUKAN) allocation ratio [0.0 - 1.0].")
  print("  P_pred : Predicted survival probability from the current state.")

  res_v1_d = simulate_strategy(exp_v1.strategy,
                               exp_v1.monthly_prices,
                               exp_v1.monthly_cashflows,
                               debug_indices=selected)
  res_v2_d = simulate_strategy(exp_v2.strategy,
                               exp_v2.monthly_prices,
                               exp_v2.monthly_cashflows,
                               debug_indices=selected)

  for p_idx in selected:
    reason = "LHS Win" if p_idx in selected_v1 else "RHS Win"
    print(f"\n[Path {p_idx} - {reason}]")
    print(f"{'Age':>3} | {'CPI':>7} | {LHS['name']:^61} | {RHS['name']:^61}")
    print(
        f"{'':>3} | {'':>7} | {'X_N':>8} {'pY_N':>7} {'Y_prev':>7} {'W_N':>8} {'A':>5} {'P_pred':>6} | {'X_N':>8} {'pY_N':>7} {'Y_prev':>7} {'W_N':>8} {'A':>5} {'P_pred':>6}"
    )
    print("-" * 150)

    debug_v1 = res_v1_d.debug_results
    debug_v2 = res_v2_d.debug_results

    if debug_v1 is None or debug_v2 is None:
      print("Error: debug_results is None")
      continue

    data_v1 = {
        parse_log(l)['age']: parse_log(l)
        for l in debug_v1.get(p_idx, [])
        if "Rebalance" in l
    }
    data_v2 = {
        parse_log(l)['age']: parse_log(l)
        for l in debug_v2.get(p_idx, [])
        if "Rebalance" in l
    }

    for age in range(60, 95):
      d1, d2 = data_v1.get(age), data_v2.get(age)
      if not d1 and not d2:
        continue
      
      cpi_val = 0.0
      if d1:
        cpi_val = d1['cpi']
      elif d2:
        cpi_val = d2['cpi']

      v1_str = f"{d1['post']:8.1f} {d1['pred_y']:7.1f} {d1['spend']:7.1f} {d1['w_n']:8.1f} {d1['a']:5.2f} {d1['p']:6.3f}" if d1 else "        Bankrupt/N.A.                                            "
      v2_str = f"{d2['post']:8.1f} {d2['pred_y']:7.1f} {d2['spend']:7.1f} {d2['w_n']:8.1f} {d2['a']:5.2f} {d2['p']:6.3f}" if d2 else "        Bankrupt/N.A.                                            "
      print(f"{age:>3} | {cpi_val:7.4f} | {v1_str} | {v2_str}")


if __name__ == "__main__":
  main()
