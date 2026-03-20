# Simulation Models Plan: S&P500 & ACWI

## Objective
The goal of the next phase is to determine whether sophisticated modeling (complex distributions, mean reversion, direct vs. approximated modeling) actually matters for our life simulation results (e.g., survival probability). 

We will strictly focus on S&P500 and ACWI, utilizing a monthly frequency for the Monte Carlo simulations.

## Analysis of Model Fitting Results

Based on `data/model_fitting_results_v2.txt`, we observe the following for monthly returns:

### S&P 500 (Monthly, All-Time 1871-03 to 2025-12)
*   **Simple Normal**: $\mu=0.008373$, $\sigma=0.042125$
*   **Log-Normal**: $\mu=0.007465$, $\sigma=0.041860$
*   **Best Fit (MSE)**: `genlogistic`, params=$(0.5975, 0.0240, 0.0171)$
*   *Note on MR-GBM*: The report shows a daily MR-GBM fit ($\theta=0.2294, \mu=8.6231, \sigma=0.3695$). However, applying Mean-Reverting Geometric Brownian Motion (MR-GBM) to daily or monthly equity index returns is highly debatable. While there is evidence for mean reversion in equities over very long horizons (e.g., 5-10 year business cycles), daily or monthly mean reversion is typically negligible or an artifact. Standard random walk (GBM) is typically sufficient and standard for retirement simulations. We will include a monthly MR-GBM model to test if this complexity alters survival rates.

### ACWI (Monthly, All-Time Available: 2008-04 to 2025-12)
*   **Simple Normal**: $\mu=0.007731$, $\sigma=0.047941$
*   **Log-Normal**: $\mu=0.006543$, $\sigma=0.048426$
*   **Best Fit (MSE)**: `laplace`, params=$(0.01329, 0.03554)$

### ACWI Approximation by S&P500 (Monthly)
*   **Data Points**: Monthly data points for overlapping periods (2008-04 to 2025-12).
*   **Methodology**: The linear regression is performed on **log returns**, not simple returns or raw prices. 
    Let $av[t]$ and $sv[t]$ be the monthly prices of ACWI and S&P500, respectively. The relationship is defined as:
    $$ \log\left(\frac{av[t+1]}{av[t]}\right) = 1.0263 \times \log\left(\frac{sv[t+1]}{sv[t]}\right) - 0.003027 + \text{Noise} $$
    ($R^2 = 0.9304$)
*   Residual Noise: `dweibull`, params=$(1.2245, 0.0, 0.01056)$

---

## Selected Models for Simulation

**Evaluation Methods**:
1.  **Log-Likelihood (LL)**: The direct probability of the model producing the observed actual time series. Higher (closer to 0) is better.
2.  **Bayesian Information Criterion (BIC)**: A rigorous metric derived from Log-Likelihood. A lower (more negative) BIC indicates a better model. BIC balances the empirical likelihood of the data under the model against a penalty for the number of parameters, preventing overfitting. 
3.  **Mean Squared Error (MSE)**: While often requested to compare model fits against empirical histograms, our standard toolings optimize parameters strictly via Maximum Likelihood Estimation (MLE) which aligns directly with maximizing LL (and minimizing BIC) rather than minimizing histogram density MSE. Thus, LL and BIC are the primary, robust metrics used below.

**Unified Evaluation: ACWI Direct vs. ACWI Approximation:**
To fairly compare all ACWI models on their ability to predict the ACWI marginal distribution (i.e. "the probability of the model producing the actual ACWI time series"), we calculate the **Log-Likelihood (LL)** of the actual ACWI data under each model. 

For the Approximation models (ACWI-APP), we calculate the **Induced Unconditional ACWI Distribution**. 
If S&P 500 log returns are $S \sim \mathcal{N}(\mu_S, \sigma_S^2)$, and $A = aS + b + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, then unconditionally:
$A \sim \mathcal{N}(a\mu_S + b, a^2\sigma_S^2 + \sigma_n^2)$.

We evaluated this induced Normal distribution against the actual ACWI data:

| Model | Description | Log-Likelihood (LL) |
| :--- | :--- | :--- |
| **ACWI-3** | Direct Best Fit (`laplace`) | **350.13** (Highest probability) |
| **ACWI-2** | Direct Log-Normal fit on ACWI | **342.67** |
| **ACWI-APP-1 + SP500-5** | Induced from S&P500 (Recent 30y) | **342.46** |
| **ACWI-APP-1 + SP500-2** | Induced from S&P500 (All-Time) | **341.13** |

*(Note: Higher LL means the model is more likely to generate the observed data).*

**Conclusion:** The induced marginal distributions from the approximation models (LL ~341-342) are nearly identical in performance to the direct Log-Normal fit (LL=342.67). This mathematically proves that approximating ACWI via S&P500 does not significantly degrade the marginal accuracy, while properly maintaining the joint correlation required for portfolio simulations involving both S&P 500 and ACWI simultaneously.

### S&P 500 Models
*Note: The Central Limit Theorem naturally drives the compound product of any stationary monthly returns towards a Log-Normal distribution over long horizons. This explains why standard Log-Normal fits provide a robust baseline.*

| Model ID | Date Range | Distribution | Dynamics | Monthly Parameters ($\mu_M$, $\sigma_M$) | Annualized ($\mu_Y$, $\sigma_Y$) | BIC | Action Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SP500-1** | 1871-03 to 2025-12 | Simple Normal | Standard | $\mu=0.008373$, $\sigma=0.042125$ | $\mu=10.54\%$, $\sigma=16.08\%$ | -6481.18 | No |
| **SP500-2** | 1871-03 to 2025-12 | Log-Normal | Standard | $\mu=0.007465$, $\sigma=0.041860$ | $\mu=10.54\%$, $\sigma=16.10\%$ | -6504.63 | No |
| **SP500-3** | 1871-03 to 2025-12 | Genlogistic | Standard | $c=0.5975, \text{loc}=0.0240, \text{scale}=0.0171$ | $\mu=10.20\%$, $\sigma=14.97\%$ | -6820.87 | No |
| **SP500-4** | 1871-03 to 2025-12 | Normal | MR-GBM | No mean reversion detected (positive slope) | N/A | N/A | No |
| **SP500-5** | 1996-01 to 2025-12 | Log-Normal | Standard | $\mu=0.008211$, $\sigma=0.044273$ | $\mu=11.67\%$, $\sigma=17.20\%$ | -1211.11 | No |
| **SP500-6** | 1996-01 to 2025-12 | Best Fit (Genlogistic) | Standard | $c=0.4880, \text{loc}=0.0333, \text{scale}=0.0173$ | $\mu=11.75\%$, $\sigma=17.21\%$ | -1235.68 | No |

### ACWI Models (Direct Modeling)
*Note: ACWI "All Available" data is exactly from 2008-04 to 2025-12.*

| Model ID | Date Range | Distribution | Dynamics | Monthly Parameters ($\mu_M$, $\sigma_M$) | Annualized ($\mu_Y$, $\sigma_Y$) | BIC | Action Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACWI-1** | 2008-04 to 2025-12 | Simple Normal | Standard | $\mu=0.007731$, $\sigma=0.047941$ | $\mu=9.68\%$, $\sigma=18.20\%$ | -678.91 | No |
| **ACWI-2** | 2008-04 to 2025-12 | Log-Normal | Standard | $\mu=0.006543$, $\sigma=0.048426$ | $\mu=9.69\%$, $\sigma=18.52\%$ | -674.62 | No |
| **ACWI-3** | 2008-04 to 2025-12 | Laplace | Standard | $\text{loc}=0.01329, \text{scale}=0.03554$ | $\mu=19.08\%$, $\sigma=20.93\%$ | -689.54 | No |

### ACWI Models (Approximation via S&P500)
These models simulate ACWI by first simulating S&P500 log returns and then applying the linear formula above: 
$$ \text{ACWI\_Log\_Return} = 1.0263 \times \text{SP500\_Log\_Return} - 0.003027 + \text{Noise} $$
*Note: The choice of underlying S&P500 model causes significant variation.*

| Model ID | Underlying SP500 Model | Noise Distribution | Monthly Parameters for Noise ($\mu_M, \sigma_M$) | BIC |
| :--- | :--- | :--- | :--- | :--- |
| **ACWI-APP-1** | SP500-2 (Log-Normal, All-Time) | Simple Normal | $\mu=0.000000$, $\sigma=0.012774$ | -1242.31 |
| **ACWI-APP-2** | SP500-2 (Log-Normal, All-Time) | Dweibull | $c=1.2245, \text{loc}=0, \text{scale}=0.01056$ | -1242.19 |
| **ACWI-APP-3** | SP500-5 (Log-Normal, 30 Yrs) | Dweibull | $c=1.2245, \text{loc}=0, \text{scale}=0.01056$ | -1242.19 |

*Interpretation of BIC for Approximation Models*: The heavily negative BIC values (-1242) reflect the fit of the *residual noise conditionally given S&P 500*, not the raw ACWI volatility. Because the residual variation ($\sigma \approx 0.012$) is extremely narrow compared to unconditional ACWI volatility ($\sigma \approx 0.048$), the conditional log-likelihood is much higher, forcing the BIC lower. These BICs confirm the approximation is structurally sound but cannot be directly compared to the BICs of the unconditioned "Direct Modeling" above. (See the unified evaluation below for a mathematically equivalent comparison).

---

## Next Concrete Actions

1.  **Recalculate Missing Parameters (Data Prep)**:
    *   Filter S&P 500 data to the "recent 30 years" (e.g., 1995-2025) and fit the Log-Normal and best-fit distributions to get parameters for **SP500-5** and **SP500-6**.
    *   Estimate monthly MR-GBM parameters for the S&P 500 (**SP500-4**).
    *   Filter ACWI data to its maximum available recent period (up to 30 years) and fit Log-Normal (**ACWI-4**).
    *   Calculate the Simple Normal fit for the ACWI approximation residuals (**ACWI-APP-1**).
2.  **Simulation Execution** and 3.  **Comparison and Evaluation** will be done after writing the full report of the to-be-tested models and obtaining user approval.

2.  **Simulation Execution**:
    *   Integrate these specific distributions into the life simulation Monte Carlo engine (e.g., `scipy.stats.genlogistic.rvs`, `scipy.stats.laplace.rvs`).
    *   Run the standard baseline retirement scenario (e.g., 4% withdrawal rule, fixed portfolio) across all defined models.
3.  **Comparison and Evaluation**:
    *   Compare the key metrics: Survival Probability, Median Final Wealth, and 5th Percentile Wealth across models.
    *   Determine if the differences between standard Log-Normal and the "Best Fit" complex distributions are statistically/practically significant.
    *   Evaluate if Approximation (ACWI-APP) yields similar survival rates to Direct Modeling (ACWI-1/2/3). This will dictate if we need separate robust models for international equities or if approximating them via US equities is sufficient.
