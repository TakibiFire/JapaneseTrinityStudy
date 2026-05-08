import os
import re
import sys
from dataclasses import replace
from typing import Any, Dict, List, Union, cast

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import simulate_strategy
from src.lib.dp_predictor import DPOptimalStrategyPredictor
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
SPENDING_RULE = 8.0
MODEL_BASE = "re60_pen70_95_n1000"

def compile_and_run():
  base_spend_annual = get_annual_retired_spending_values(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      60, 1)[0]

  exp_setup = re60_pen70_95(n_sim=N_SIM_EVAL, seed=SEED)
  exp_setup.name = "experimental_dp_eval"
  init_money = base_spend_annual / (SPENDING_RULE / 100.0)
  model_path = f"data/optimal_strategy_dp/experiments/{MODEL_BASE}.json"

  exp_setup.add_experiment(name="OFF", overwrite_strategy=replace(
      exp_setup.strategy, initial_money=float(init_money),
      rebalance=SpendAwareDPRebalance(risky_asset=PredefinedStock.ORUKAN_155, zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                                     model_name=model_path, disable_win_threshold=True)))

  exp_setup.add_experiment(name="ON", overwrite_strategy=replace(
      exp_setup.strategy, initial_money=float(init_money),
      rebalance=SpendAwareDPRebalance(risky_asset=PredefinedStock.ORUKAN_155, zero_risk_asset=PredefinedZeroRisk.ZERO_RISK_4PCT,
                                     model_name=model_path, disable_win_threshold=False)))

  compiled = create_experiment_setup(exp_setup, record_annual_spend=False)
  return compiled[1], compiled[2]

def parse_log(log_str, predictor):
  # cur_ann_spend=2.30, prev_ann_spend=502.35, rem_years=...
  m_age = re.search(r"Age (\d+)", log_str)
  m_spend = re.search(r"cur_ann_spend=([\d\.]+)", log_str)
  m_prev = re.search(r"prev_ann_spend=([\d\.]+)", log_str)
  m_net = re.search(r"total_net=([\d\.]+)", log_str)
  m_post = re.search(r"post_tax_net=([\d\.]+)", log_str)
  m_price = re.search(r"ORUKAN_155: ([\d\.]+)", log_str)
  m_a = re.search(r"'ORUKAN_155': [^\(]*\(([\d\.]+)\)", log_str)
  if not m_a: m_a = re.search(r"'ORUKAN_155': ([\d\.]+)", log_str)
  
  age = int(m_age.group(1)) if m_age else 0
  net = float(m_net.group(1)) if m_net else 0.0
  post = float(m_post.group(1)) if m_post else 0.0
  spend = float(m_spend.group(1)) if m_spend else 0.0
  prev_spend = float(m_prev.group(1)) if m_prev else 0.0
  
  # P の再計算 (expected_growth 等)
  expected_growth = predictor.get_spend_multiplier(age - 1, age)
  s_rate = (spend * expected_growth / post) if post > 1e-6 else 1.0
  p_val = float(predictor.predict_p_surv(age, s_rate))
  
  # W_N の計算
  w_n = predictor.calculate_winning_threshold(age, prev_spend)
  
  return {
      'age': age, 'spend': spend, 'net': net, 'post': post,
      'price': float(m_price.group(1)) if m_price else 0.0,
      'a': float(m_a.group(1)) if m_a else 0.0,
      'p': p_val,
      'w_n': float(w_n)
  }

def main():
  exp_off, exp_on = compile_and_run()
  model_path = f"data/optimal_strategy_dp/experiments/{MODEL_BASE}.json"
  pred_off = DPOptimalStrategyPredictor(model_path, disable_win_threshold=True)
  pred_on = DPOptimalStrategyPredictor(model_path, disable_win_threshold=False)

  res_off = simulate_strategy(exp_off.strategy, exp_off.monthly_prices, exp_off.monthly_cashflows)
  res_on = simulate_strategy(exp_on.strategy, exp_on.monthly_prices, exp_on.monthly_cashflows)

  diff = np.where((res_off.sustained_months < YEARS * 12) & (res_on.sustained_months == YEARS * 12))[0]
  if len(diff) == 0: return
  selected = diff[:3].tolist()
  
  res_off_d = simulate_strategy(exp_off.strategy, exp_off.monthly_prices, exp_off.monthly_cashflows, debug_indices=selected)
  res_on_d = simulate_strategy(exp_on.strategy, exp_on.monthly_prices, exp_on.monthly_cashflows, debug_indices=selected)

  for p_idx in selected:
    print(f"\n[Path {p_idx}]")
    print(f"{'Age':>3} | {'P_Orukan':>10} | {'Heuristic ON (disable=False)':^45} | {'Heuristic OFF (disable=True)':^45}")
    print(f"{'':>3} | {'':>10} | {'X_N':>8} {'Y_N':>7} {'W_N':>8} {'A':>5} {'P_pred':>6} | {'X_N':>8} {'Y_N':>7} {'W_N':>8} {'A':>5} {'P_pred':>6}")
    print("-" * 155)
    
    data_on = {parse_log(l, pred_on)['age']: parse_log(l, pred_on) for l in res_on_d.debug_results[p_idx] if "Rebalance" in l}
    data_off = {parse_log(l, pred_off)['age']: parse_log(l, pred_off) for l in res_off_d.debug_results[p_idx] if "Rebalance" in l}
    
    for age in range(60, 95):
        d1, d2 = data_on.get(age), data_off.get(age)
        if not d1 and not d2: continue
        pr = f"{d1['price']:10.4f}" if d1 else f"{d2['price']:10.4f}"
        
        v_on = f"{d1['net']:8.1f} {d1['spend']:7.1f} {d1['w_n']:8.1f} {d1['a']:5.2f} {d1['p']:6.3f}" if d1 else "        Bankrupt/N.A.                "
        v_off = f"{d2['net']:8.1f} {d2['spend']:7.1f} {'inf':>8} {d2['a']:5.2f} {d2['p']:6.3f}" if d2 else "        Bankrupt/N.A.                "
        print(f"{age:>3} | {pr} | {v_on} | {v_off}")

if __name__ == "__main__":
  main()
