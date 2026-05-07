/**
 * 60歳リタイア用 DP 最適戦略計算機 (JavaScript 版)
 * 
 * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 補間を実装し、
 * Python の scipy.interpolate.pchip_interpolate と同様の挙動を再現します。
 */

class PchipInterpolator {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.n = x.length;
    if (this.n < 2) throw new Error("At least 2 points are required for interpolation.");
    
    this.h = new Array(this.n - 1);
    this.delta = new Array(this.n - 1);
    for (let i = 0; i < this.n - 1; i++) {
      this.h[i] = x[i + 1] - x[i];
      this.delta[i] = (y[i + 1] - y[i]) / this.h[i];
    }

    this.d = new Array(this.n);
    // Endpoints
    this.d[0] = this.computeEndpointDerivative(this.h[0], this.h[1], this.delta[0], this.delta[1]);
    this.d[this.n - 1] = this.computeEndpointDerivative(this.h[this.n - 2], this.h[this.n - 3], this.delta[this.n - 2], this.delta[this.n - 3]);

    // Internal points
    for (let i = 1; i < this.n - 1; i++) {
      if (this.delta[i - 1] * this.delta[i] <= 0) {
        this.d[i] = 0;
      } else {
        const w1 = 2 * this.h[i] + this.h[i - 1];
        const w2 = this.h[i] + 2 * this.h[i - 1];
        this.d[i] = (w1 + w2) / (w1 / this.delta[i - 1] + w2 / this.delta[i]);
      }
    }
  }

  computeEndpointDerivative(h0, h1, d0, d1) {
    // One-sided derivative at the boundary
    const s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (s * d0 <= 0) {
      return 0;
    } else if (d0 * d1 <= 0 && Math.abs(s) > Math.abs(3 * d0)) {
      return 3 * d0;
    }
    return s;
  }

  interpolate(xi) {
    if (xi <= this.x[0]) return this.y[0];
    if (xi >= this.x[this.n - 1]) return this.y[this.n - 1];

    // Binary search for the interval
    let low = 0, high = this.n - 2;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      if (this.x[mid] <= xi && xi <= this.x[mid + 1]) {
        const i = mid;
        const h = this.h[i];
        const t = (xi - this.x[i]) / h;
        const t2 = t * t;
        const t3 = t2 * t;
        
        // Hermite basis functions
        const h00 = 2 * t3 - 3 * t2 + 1;
        const h10 = t3 - 2 * t2 + t;
        const h01 = -2 * t3 + 3 * t2;
        const h11 = t3 - t2;

        return h00 * this.y[i] + h10 * h * this.d[i] + h01 * this.y[i + 1] + h11 * h * this.d[i + 1];
      } else if (this.x[mid] > xi) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return this.y[this.n - 1];
  }
}

class DPOptimalStrategyPredictor {
  constructor(data) {
    this.data = data;
    this.cpi_annual_mu = data.cpi_annual_mu || 0;
    this.cpi_annual_sigma = data.cpi_annual_sigma || 0;
    this.a_opt_interpolators = {};
    this.p_surv_interpolators = {};
    
    for (const age in data) {
      if (isNaN(parseInt(age))) continue;
      const ageData = data[age];
      if (ageData.a_opt_model && ageData.a_opt_model.r_points.length >= 2) {
        this.a_opt_interpolators[age] = new PchipInterpolator(ageData.a_opt_model.r_points, ageData.a_opt_model.a_points);
      }
      if (ageData.p_survival_model && ageData.p_survival_model.r_points.length >= 2) {
        this.p_surv_interpolators[age] = new PchipInterpolator(ageData.p_survival_model.r_points, ageData.p_survival_model.p_points);
      }
    }
  }

  getUnexpectedCpiJump(zScore = 2.326) {
    const denom = 1.0 + this.cpi_annual_mu;
    if (denom <= 0) return 1.0;
    return (1.0 + this.cpi_annual_mu + zScore * this.cpi_annual_sigma) / denom;
  }

  getSpendMultiplier(ageFrom, ageTo) {
    const yFrom = this.data[ageFrom]?.avg_y_withdraw;
    const yTo = this.data[ageTo]?.avg_y_withdraw;
    if (!yFrom || !yTo || yFrom <= 1e-6) return 1.0;
    return yTo / yFrom;
  }

  predictAOpt(age, sRate) {
    const model = this.data[age]?.a_opt_model;
    if (!model) return 1.0;
    if (sRate <= model.r_min_a || sRate >= model.r_max_a) return 1.0;
    if (!this.a_opt_interpolators[age]) return model.a_points[0] || 1.0;
    return this.a_opt_interpolators[age].interpolate(sRate);
  }

  predictPSurv(age, sRate) {
    const model = this.data[age]?.p_survival_model;
    const ageData = this.data[age];
    if (!model) return 0.0;
    if (sRate <= model.r_min_p) return ageData.p_max || 1.0;
    if (sRate >= model.r_max_p) return ageData.p_min || 0.0;
    if (!this.p_surv_interpolators[age]) return model.p_points[0] || 0.0;
    return this.p_surv_interpolators[age].interpolate(sRate);
  }

  calculateWinningThreshold(age, lastYWithdraw, zScore = 2.326) {
    const mN = this.data[age]?.m_winning_multiplier || 0;
    if (mN <= 0) return Infinity;
    
    const expectedGrowth = this.getSpendMultiplier(age - 1, age);
    const worstCaseYN = lastYWithdraw * expectedGrowth * this.getUnexpectedCpiJump(zScore);
    return mN * worstCaseYN;
  }

  getAOptWithWinningThreshold(age, initialWealth, lastYWithdraw, zScoreWinning = 2.326, zScoreNextSpend = 0.0) {
    const wN = this.calculateWinningThreshold(age, lastYWithdraw, zScoreWinning);
    
    if (initialWealth > wN) {
      return (initialWealth - wN) / initialWealth;
    }

    let expectedGrowth = this.getSpendMultiplier(age - 1, age);
    if (zScoreNextSpend !== 0) {
      expectedGrowth *= this.getUnexpectedCpiJump(zScoreNextSpend);
    }
    const expectedYN = lastYWithdraw * expectedGrowth;
    const sRate = expectedYN / initialWealth;
    return this.predictAOpt(age, sRate);
  }
}

// グローバルな予測器インスタンスを保持するオブジェクト
const predictors = {};
const mValues = ["0.75", "1", "1.2", "1.5", "2", "3"];
const mFiles = {
  "0.75": 'data/all_60yr/re60_pen70_95_m0_75.json',
  "1": 'data/all_60yr/re60_pen70_95_m1.json',
  "1.2": 'data/all_60yr/re60_pen70_95_m1_2.json',
  "1.5": 'data/all_60yr/re60_pen70_95_m1_5.json',
  "2": 'data/all_60yr/re60_pen70_95_m2.json',
  "3": 'data/all_60yr/re60_pen70_95_m3.json'
};

async function initPredictors() {
  const promises = mValues.map(async (m) => {
    try {
      const response = await fetch(mFiles[m]);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      predictors[m] = new DPOptimalStrategyPredictor(data);
      calculateAll();
    } catch (e) {
      console.error(`Failed to load model M=${m}:`, e);
    }
  });
  await Promise.all(promises);
}

function calculateAll() {
  const age = parseInt(document.getElementById('age').value);
  const wealth = parseFloat(document.getElementById('wealth').value);
  const lastSpend = parseFloat(document.getElementById('lastSpend').value);

  if (isNaN(age) || isNaN(wealth) || isNaN(lastSpend)) return;

  mValues.forEach(m => {
    const predictor = predictors[m];
    const aEl = document.getElementById(`a_val_${m.replace('.', '_')}`);
    const pEl = document.getElementById(`p_val_${m.replace('.', '_')}`);
    
    if (!predictor) return;
    if (!aEl || !pEl) return;

    if (!predictor.data[age]) {
      aEl.textContent = 'N/A';
      pEl.textContent = 'N/A';
      return;
    }

    // A の計算
    const aOpt = predictor.getAOptWithWinningThreshold(age, wealth, lastSpend);
    
    // P の計算 (sRate = expectedYN / initialWealth)
    const expectedGrowth = predictor.getSpendMultiplier(age - 1, age);
    const expectedYN = lastSpend * expectedGrowth;
    const sRate = expectedYN / wealth;
    const pSurv = predictor.predictPSurv(age, sRate);

    if (aEl) aEl.textContent = (aOpt * 100).toFixed(1) + '%';
    if (pEl) pEl.textContent = (pSurv * 100).toFixed(1) + '%';
  });
}

// 起動時
document.addEventListener('DOMContentLoaded', async () => {
  await initPredictors();
  calculateAll();
  
  // イベントリスナー
  ['age', 'wealth', 'lastSpend'].forEach(id => {
    document.getElementById(id).addEventListener('input', calculateAll);
  });
});
