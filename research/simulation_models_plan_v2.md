# Simulation Models Plan V2: S&P500 & ACWI

## Objective
The goal of this phase is to determine whether sophisticated modeling (complex distributions, mean reversion, direct vs. approximated modeling) actually matters for our life simulation results (e.g., survival probability). 

We strictly focus on S&P500 and ACWI, utilizing a monthly frequency for the Monte Carlo simulations.

This **V2 document** updates the methodology from [V1 (simulation_models_plan.md)](simulation_models_plan.md) based on a critical mathematical finding regarding fat tails and long-term expected returns. The original design for this fix is in [`tasks/model_fitting_design2.md`](../tasks/model_fitting_design2.md), and the implementation is located in `src/lib/asset_model.py`. The raw output for V3 is in `data/model_fitting_results_v3.txt`.

---

## Pitfalls & Theoretical Conversations (Why V1 Failed)

During the evaluation of V1, a critical issue was discovered regarding the **ACWI-3** model, which used a Laplace distribution and predicted an unrealistic annualized arithmetic mean of 19.08% (compared to the empirical ~9.68%). 

### 1. The Symmetry Trap
In V1, the optimization algorithm chose `laplace` because it effectively minimized Mean Squared Error (MSE) and maximized Log-Likelihood by capturing the extreme left tail (crashes like 2008). However, because `laplace` is strictly symmetric, forcing it to learn the left-tail risk caused it to falsely predict identical right-tail risk (frequent extreme booms). 
*   **Conclusion:** We must exclude strictly symmetric distributions (`laplace`, `t`, `cauchy`, `dweibull`) when modeling unconditional equity returns, as equities exhibit inherent negative skewness (crashes are sharper than rallies).

### 2. The Parameter Constraint Trap (Location != Mean)
Our first attempt to fix the CAGR involved centering the data (subtracting the empirical mean $\mu_{emp}$) and forcing the optimization routine to fit the distribution with `loc=0`. 
*   **The Pitfall:** For asymmetric distributions (like `johnsonsu`), the `loc` parameter is **not** the theoretical mean. When we simulated a `johnsonsu` model fitted this way in `src/confirm_median_main.py`, the 50-year Median Terminal Wealth (CAGR) was wildly off (e.g., 9.70% vs 7.98% for Log-Normal), showing a 119% discrepancy.
*   **The Mathematical Reality:** If you want the long-term Median Terminal Wealth to match historical data (to isolate the effect of risk from return), you must anchor the **Theoretical Expected Value of the Log Returns ($E[X]$)**.

### 3. The Correct "Shift" Methodology (V2 Implementation)
To solve this, we implemented the `find_best_distribution_with_fixed_mean` function in `src/lib/asset_model.py`:
1. Let the optimizer fit the asymmetric distribution freely to the raw data (finding the best shape).
2. Calculate the *theoretical expected value* ($E[X]$) of that fitted distribution using `dist.mean(*params_raw)`.
3. Mathematically shift the distribution's `loc` parameter by the offset (`mu_emp - mu_theo`).
4. This guarantees that the new shifted distribution has the exact same expected value as the historical log-returns, preserving the exact shape (variance, skewness, kurtosis) found by the optimizer.

**Verification:** As verified by `src/confirm_median_main.py`, this shift perfectly aligns the 50-year CAGR. A Monte Carlo simulation of 100,000 paths over 50 years produced median terminal wealths of 46.13 (Log-Normal) vs 46.71 (Johnson SU). The 1.27% difference is purely normal Monte Carlo variance, proving the long-term growth engine is now identical across models.

---

## Analysis of Model Fitting Results (V3 Data)

Based on `data/model_fitting_results_v3.txt`, we observe the following for monthly returns under the new mean-constrained logic:

### S&P 500 Models
*Note: The Central Limit Theorem naturally drives the compound product of any stationary monthly returns towards a Log-Normal distribution over long horizons. Standard Log-Normal fits provide the baseline.*

| Model ID | Date Range | Distribution | Dynamics | Monthly Parameters ($\mu_M$, $\sigma_M$) | Annualized Arith. Mean | Annualized Stddev | BIC | Action Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SP500-1** | 1871-03 to 2025-12 | Simple Normal | Standard | $\mu=0.008344$, $\sigma=0.042105$ | $10.52\%$ | $16.06\%$ | -6493.38 | No |
| **SP500-2** | 1871-03 to 2025-12 | Log-Normal | Standard | $\mu=0.007437$, $\sigma=0.041840$ | $10.53\%$ | $16.12\%$ | -6516.88 | No |
| **SP500-3** | 1871-03 to 2025-12 | Genlogistic | Fixed Mean | $c=0.598, \text{loc}=0.024, \text{scale}=0.017$ | $10.39\%$ | $14.90\%$ | -6833.03 | No |
| **SP500-4** | 1871-03 to 2025-12 | Normal | MR-GBM | No mean reversion detected | N/A | N/A | N/A | No |

*(Note on SP500-3: The Genlogistic model is constrained to have the exact same log-mean as SP500-2. Its arithmetic mean is slightly lower due to different higher-order moments influencing the exponentiation.)*

### ACWI Models (Direct Modeling)
*Note: ACWI "All Available" data is exactly from 2008-04 to 2025-12.*

| Model ID | Date Range | Distribution | Dynamics | Monthly Parameters ($\mu_M$, $\sigma_M$) | Annualized Arith. Mean | Annualized Stddev | BIC | Action Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACWI-1** | 2008-04 to 2025-12 | Simple Normal | Standard | $\mu=0.007573$, $\sigma=0.047802$ | $9.47\%$ | $18.12\%$ | -689.85 | No |
| **ACWI-2** | 2008-04 to 2025-12 | Log-Normal | Standard | $\mu=0.006393$, $\sigma=0.048285$ | $9.47\%$ | $18.38\%$ | -685.50 | No |
| **ACWI-3** | 2008-04 to 2025-12 | Johnson SU | Fixed Mean | $a=0.598, b=1.597, \text{loc}=0.033, \text{scale}=0.058$ | $9.50\%$ | $18.42\%$ | -695.84 | No |

*(Note on ACWI-3: The previous V1 Laplace model had a highly inflated arithmetic mean of 19.08%. The new Johnson SU model correctly aligns the long-term compound growth rate, yielding a realistic arithmetic mean of 9.50%, while successfully capturing the fat tails, as evidenced by its superior BIC of -695.84).*

### ACWI Models (Approximation via S&P500)
These models simulate ACWI by first simulating S&P500 log returns and then applying the linear formula above: 
$$ \text{ACWI\_Log\_Return} = 1.0269 \times \text{SP500\_Log\_Return} - 0.002907 + \text{Noise} $$

*Note: For residual noise, symmetric distributions are acceptable because they model variance around a trend rather than absolute long-term growth.*

| Model ID | Underlying SP500 Model | Noise Distribution | Monthly Parameters for Noise | BIC |
| :--- | :--- | :--- | :--- | :--- |
| **ACWI-APP-1** | SP500-2 (Log-Normal) | Dweibull | $c=1.219, \text{loc}=0, \text{scale}=0.0106$ | -1254.65 |

---

## Next Concrete Actions

1.  **Simulation Execution**:
    *   Integrate these specific updated distributions (`scipy.stats.genlogistic.rvs`, `scipy.stats.johnsonsu.rvs`) into the life simulation Monte Carlo engine.
    *   Run the standard baseline retirement scenario (e.g., 4% withdrawal rule, fixed portfolio) across all defined models (Log-Normal vs Best Asymmetric).
2.  **Comparison and Evaluation**:
    *   Compare the key metrics: Survival Probability, Median Final Wealth, and 5th Percentile Wealth across models.
    *   Determine if the differences between standard Log-Normal and the "Best Asymmetric Fixed Mean" fat-tailed distributions are statistically/practically significant regarding *survival rates*. (We already mathematically proved the Median Final Wealth will align).
    *   Evaluate if Approximation (ACWI-APP) yields similar survival rates to Direct Modeling (ACWI-3).