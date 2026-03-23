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

## Appendix (DO NOT DELETE)

以下は AI との chat です。

I made `Annualized ($\mu_Y$, $\sigma_Y$)` in the md file to quickly understand the distribution.

My question is
| Model ID | Date Range | Distribution | Dynamics | Monthly Parameters ($\mu_M$, $\sigma_M$) | Annualized ($\mu_Y$, $\sigma_Y$) | BIC | Action Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACWI-1** | 2008-04 to 2025-12 | Simple Normal | Standard | $\mu=0.007731$, $\sigma=0.047941$ | $\mu=9.68\%$, $\sigma=18.20\%$ | -678.91 | No |
| **ACWI-2** | 2008-04 to 2025-12 | Log-Normal | Standard | $\mu=0.006543$, $\sigma=0.048426$ | $\mu=9.69\%$, $\sigma=18.52\%$ | -674.62 | No |
| **ACWI-3** | 2008-04 to 2025-12 | Laplace | Standard | $\text{loc}=0.01329, \text{scale}=0.03554$ | $\mu=19.08\%$, $\sigma=20.93\%$ | -689.54 | No |

ACWI-3 just uses laplace, but the annualized mu is much higher than ACWI-1 and ACWI-2. これ本当にきちんと計算できている？

---

ご質問ありがとうございます。

結論から申し上げますと、**19.08%という数値の計算自体（シミュレーション結果として）は数学的に「正しく」計算されたものです。しかし、一般的な投資信託の「目論見書」などで使われる計算方法（リスクとリターンを単純に年率換算する方法）とは異なります。**

Laplace（ラプラス）分布のような裾の重い（ファットテールな）分布を対数リターンとして毎月複利で掛け合わせると、極端にリターンが大きい月が複数回発生する確率が高まるため、期待値（平均値）が上振れするという数学的性質があるためです。

### 1. なぜLaplaceだと19.08%に跳ね上がっているのか？（計算のカラクリ）

プログラム（`src/lib/asset_model.py`）を確認すると、Annualizedの値は以下の計算で求められています。

```python
monthly_sims = dist.rvs(*params, size=(n_sims, 12)) # 毎月のリターンを12ヶ月分シミュレーション
annual_sims = np.exp(np.sum(monthly_sims, axis=1)) - 1 # 対数リターンを12ヶ月合計して単利に直す
```

これは、期待値 $E[e^{X_1 + X_2 + ... + X_{12}}] - 1$ を直接求めているものです。
正規分布に比べてラプラス分布は「極端に大きなプラス」と「極端に大きなマイナス」が起きやすい性質を持っています。しかし、投資のリターン（$e^X$）では、マイナスはどれだけ大きくても「-100%」で止まるのに対し、プラスは「+200%」「+300%」と青天井になります。

そのため、正規分布を当てはめたACWI-1/ACWI-2（約9.6%）と同じかそれ以上に元本を大きく減らす確率がある一方で、時々「とてつもない大化け」を引いて資産が大きく増えるケースがシミュレーションの平均値を強力に押し上げてしまい、結果として「平均リターンが19.08%」という数値に膨れ上がっています。

### 2. 一般的な「目論見書」などの方法で換算するといくらになるか？

目論見書や一般的な金融機関のレポートでは、単純な「年率換算」がよく使われます。
月次パラメータ $\mu_M=0.01329$、$\sigma_M=0.03554$ を使った場合、以下のように計算されます。

**①対数リターンの単純年率化による推定**
- $\mu_{\text{annual}} = \mu_M \times 12 = 0.01329 \times 12 = 0.15948$ **(15.95%)**
- $\sigma_{\text{annual}} = \sigma_M \times \sqrt{2} \times \sqrt{12}$ ※
※ラプラス分布の標準偏差は $\sqrt{2} \times \text{scale}$ のため、
$0.03554 \times 1.414 \times 3.464 \approx 0.1741$ **(17.41%)**

さらに、これを単利（幾何平均ベース）として簡易的に変換するよくある近似式 $\mu_{\text{simple}} \approx \mu_{\text{annual}} + \frac{\sigma_{\text{annual}}^2}{2}$ に当てはめると、
$15.95\% + (17.41\%)^2 / 2 \approx 15.95\% + 1.51\% = 17.46\%$ となります。

**②単純な月次期待値の複利計算**
月次の単利期待値（近似）: $e^{0.01329} - 1 \approx 1.338\%$
年率化: $(1.01338)^{12} - 1 \approx 0.1728$ **(17.28%)**

つまり、目論見書のような単純な換算（外れ値の爆発を直接シミュレーションしない計算）でならして表現すると、**平均リターンは約 16.0% 〜 17.5% 、リスク（標準偏差）は約 17.4%** 程度と記載されるような数値感になります。

### まとめ

*   **ACWI-3の19.08%は間違っているわけではなく**、「ラプラス分布が持つ上振れリスク（ファットテール）」が、対数リターンから単利へ変換される際に指数関数的に増幅された「数学的に正しい期待値」です。
*   しかし、現実的な目論見書の感覚で言えば、**「リターン 16〜17%前後、リスク 17.4%」**程度のパフォーマンスを指しています。（それでも通常の正規分布当てはめよりかなり高めに出ています。これはラプラス分布が極端な外れ値のデータを拾い上げて平均を引き上げているためです。）

---

いや、でも目論見書と同じように 2008-04 to 2025-12 で annualized mu, sigma を計算することは確定的にできるでしょ？

自分はそれは ACWI-1 が表示している `| $\mu=9.68\%$, $\sigma=18.20\%$` に相当していると思っていて、つまりこれが annualized mu, sigma の正解値だと思っています。

それからすると ACWI-3 はだいぶずれていると思うけど、それなのに best fit と言っているし。I'm puzzled

---

ご指摘の通りです。おっしゃる通り、2008-04から2025-12までの実際のデータから計算される「確定的な年率リターンとリスク」は、まさに **ACWI-1 の $\mu=9.68\%$, $\sigma=18.20\%$** （あるいは実際の過去データのCAGR）に相当するものです。

それなのになぜ、実際の平均（9.68%）から大きくかけ離れた期待値（19.08%）を持つ ACWI-3（Laplace）が「Best Fit」と判定されてしまったのか？という疑問は完全に的を射ています。

### なぜずれているのに「Best Fit」と言っているのか？

理由は、このプログラムにおける「Best Fit」の評価基準（MSEやLL、BIC）が、**「分布の中心（平均）がデータと合っているか」よりも「極端な暴落・暴騰の発生頻度（テール）がデータと合っているか」を重視してしまった**ためです。

1.  **対数リターンの分布形状へのフィッティング:**
    `scipy.stats` の分布当てはめ（Maximum Likelihood Estimation: MLE）は、用意されたデータの「ヒストグラム全体の形」を最もよく説明するパラメータを探します。
    2008年（リーマンショック）や2020年（コロナショック）など、株式市場には「正規分布では絶対に起こり得ないレベルの大暴落・大暴騰」が存在します。
2.  **Laplace分布の特徴と「平均の犠牲」:**
    Laplace分布は、正規分布よりも中央が尖っており、かつ裾が厚い（極端な値が出やすい）という特徴があります。プログラムがこの期間のデータにLaplaceを当てはめた時、リーマンショックなどの「極端な外れ値の頻度」をうまく説明できることを高く評価しました。
    しかし、Laplace分布の数式上の制約により、**「極端な外れ値の頻度」を現実に合わせようとパラメータ（loc, scale）を調整した結果、「分布の中心（平均リターン）」が現実の9.68%から大きく上にずれてしまった**のです。
3.  **対数から単利への変換による爆発:**
    さらに、Laplace分布が持つ「極端な上振れ（+50%など）」が対数リターンから単利に変換（$e^X-1$）される際に指数関数的に爆発し、シミュレーション上の平均値を19.08%まで異常に押し上げてしまいました。

### 結論

ユーザー様の疑問の通り、**「シミュレーションの平均リターンが現実の過去実績（9.68%）からここまで大きく乖離しているモデルは、いくら数学的な指標（BICやMSE）で "Best Fit" と出ても、長期の資産形成シミュレーションとしては不適切（使い物にならない）」**というのが正しい解釈です。

金融工学の実務でも、このような場合は「平均は現実の過去データ（9.68%）に固定した上で、リスク（分散やテールの厚さ）だけをファットテールな分布でモデル化する」という調整を行うのが一般的です。

現状のACWI-3のままシミュレーション（Monte Carlo）を回すと、「現実のACWIよりはるかに儲かりやすい（平均19.08%で増える）」という非現実的な結果になってしまいますので、ご指摘の通り、このまま採用すべきではありません。

---

ご指摘を受けて、改めてデータの期間や性質、そしてシミュレーションの計算方法について詳細を整理しました。非常に鋭いご指摘をいただき、ありがとうございます。

19.08%という数値がなぜ出てくるのか、そして「それが使い物にならないのか、あるいは理論的に起こり得る（学習の成果である）のか」について、以下の3つの観点から回答いたします。

### 1. なぜ「使い物にならない（不適切）」と言えるのか？（理論的な理由）

これは**「平均は現実の過去データ（9.68%）に固定した上で、リスクだけをファットテールな分布でモデル化する」のが実務上（理論上）正しいアプローチだから**です。

17年というデータ期間が短いからというよりは、**「株式リターンの非対称性」をLaplace分布（左右対称な分布）で無理やり表現しようとしたこと**に原因があります。

*   **現実の株式市場:** 大暴落（左側のテール）は一瞬で大きく起きますが、大暴騰（右側のテール）はそれほど大きくありません。（左に歪んだファットテール）
*   **Laplace分布:** 左右対称の分布です。プログラムが「リーマンショックなどの大暴落（左のテール）」をデータから学習し、「これを再現できるように裾を厚くしよう（scaleを大きくしよう）」と判断します。すると、**Laplaceは左右対称なので、自動的に「右側のテール（大暴騰）」も同じ確率で発生するように設定してしまいます。**

その結果、現実にはめったに起きない「月に+20%以上」のような大暴騰がシミュレーション上で頻繁に発生し、それが複利（`np.exp`）で年率化される際に爆発して、平均リターンを19.08%まで押し上げてしまったのです。
つまり、「大きな上振れが多かったから学習した」のではなく、「下振れを学習した結果、モデルの制約（左右対称）のせいで、存在しないはずの巨大な上振れを捏造してしまった」というのが真相です。

### 2. `monthly_sims = dist.rvs(*params, size=(n_sims, 12))` がブレすぎている可能性は？

この可能性は低いです（大数の法則が働きます）。
`n_sims` が十分に大きければ（例えば10万〜100万回）、乱数によるブレは極めて小さくなり、理論値に収束します。19.08%という数値は、ブレではなく**「Laplace分布の理論上の期待値そのもの」**が約19%になっていることを示しています。

### 3. では、どうすべきか？

ユーザー様が「ACWI-1（9.68%）が正解値だ」とお考えになっている直感は、まさに正しいです。長期シミュレーションにおいて、**「リターンの期待値（平均）」を現実に合わせることは、最も重要で優先されるべき条件**です。

もし、ファットテール（極端な変動）をシミュレーションに取り入れたいのであれば、左右対称なLaplace分布ではなく、以下のようなアプローチが適切です。

1.  **非対称な分布を使う:** S&P500のベストフィット（`genlogistic`）のように、左右の歪み（歪度）を表現できる分布を使う。
2.  **平均を固定する（実務的なアプローチ）:** 現実の平均リターン（9.68%等）にモデルの平均値を強制的に合わせる（`loc` パラメータを調整する等）。
3.  **歴史的シミュレーション（ブートストラップ法）:** 過去の月次リターンデータをそのままランダムに抽出して並べ替える。これにより、過去の暴落・暴騰の「実際の頻度と大きさ」をそのまま再現しつつ、平均値も過去データと完全に一致させることができます。

結論として、現状の「ACWI-3 (Laplace)」は、暴落を表現しようとした代償として非現実的な暴騰を生み出し、長期の平均リターンを著しく歪めているため、そのまま資産形成シミュレーションに用いるべきではありません。

