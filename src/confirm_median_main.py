import numpy as np
from scipy import stats


def confirm_median_convergence():
  # Parameters for ACWI (Monthly) from model_fitting_results_v3.txt
  mu_log = 0.006393421399374782
  std_log = 0.048285 # Log-Normal std
  
  # Johnson SU params (Updated from V3 output)
  jsu_params = (0.5985794609028992, 1.5979947040822444, 0.033828733503047, 0.05883263137905962)
  
  n_sims = 100000
  years = 50
  months = years * 12
  
  print(f"--- Simulating {years} years ({months} months) x {n_sims} paths ---")
  print(f"Target monthly log-mean: {mu_log:.6f}")
  
  # 1. Log-Normal (Normal in log-space)
  log_rets_norm = np.random.normal(mu_log, std_log, size=(n_sims, months))
  final_wealth_norm = np.exp(np.sum(log_rets_norm, axis=1))
  median_norm = np.median(final_wealth_norm)
  cagr_norm = median_norm**(1/years) - 1
  
  # 2. Johnson SU
  log_rets_jsu = stats.johnsonsu.rvs(*jsu_params, size=(n_sims, months))
  final_wealth_jsu = np.exp(np.sum(log_rets_jsu, axis=1))
  median_jsu = np.median(final_wealth_jsu)
  cagr_jsu = median_jsu**(1/years) - 1
  
  print("\nResults (50-year Final Wealth Median):")
  print(f"Log-Normal: {median_norm:.4f} (CAGR: {cagr_norm*100:.2f}%)")
  print(f"Johnson SU: {median_jsu:.4f} (CAGR: {cagr_jsu*100:.2f}%)")
  
  diff_pct = (median_jsu / median_norm - 1) * 100
  print(f"\nDifference: {diff_pct:.2f}%")
  
  # Also check mean (arithmetic) to show they DO differ there
  mean_norm = np.mean(final_wealth_norm)
  mean_jsu = np.mean(final_wealth_jsu)
  print("\nResults (50-year Final Wealth Mean - Arithmetic):")
  print(f"Log-Normal: {mean_norm:.4f}")
  print(f"Johnson SU: {mean_jsu:.4f}")

if __name__ == "__main__":
  confirm_median_convergence()
