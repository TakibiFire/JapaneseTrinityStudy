/**
 * 汎用 DP 最適戦略計算機 (JavaScript 版)
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

    let low = 0, high = this.n - 2;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      if (this.x[mid] <= xi && xi <= this.x[mid + 1]) {
        const i = mid;
        const h = this.h[i];
        const t = (xi - this.x[i]) / h;
        const t2 = t * t;
        const t3 = t2 * t;
        
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

  calculateWinningThreshold(age, expectedYN, zScore = 2.326) {
    const mN = this.data[age]?.m_winning_multiplier || 0;
    if (mN <= 0) return Infinity;
    const worstCaseYN = expectedYN * this.getUnexpectedCpiJump(zScore);
    return mN * worstCaseYN;
  }

  getAOptWithWinningThreshold(age, initialWealth, expectedYN, zScoreWinning = 2.326) {
    const wN = this.calculateWinningThreshold(age, expectedYN, zScoreWinning);
    if (initialWealth > wN) {
      return (initialWealth - wN) / initialWealth;
    }
    const sRate = expectedYN / initialWealth;
    return this.predictAOpt(age, sRate);
  }
}

let calcConfig = null;
let formulaConfig = null;
let predictors = {};
let mValues = [];
let startAge = 60;

async function init() {
  const params = new URLSearchParams(window.location.search);
  const dataKey = params.get('data') || 'all_60yr';
  
  try {
    // dp_calc.json から計算機設定を読み込む
    const cResponse = await fetch(`data/${dataKey}/dp_calc.json`);
    if (!cResponse.ok) throw new Error("Failed to load dp_calc.json");
    calcConfig = await cResponse.json();

    startAge = calcConfig.start_age || 60;
    const targetAge = calcConfig.target_age || 95;

    document.getElementById('html-title').textContent = `${startAge}歳リタイア用 最適オルカン配分計算機`;
    document.getElementById('title').textContent = `${startAge}歳リタイア用 最適オルカン配分計算機`;
    document.getElementById('age-base-header').textContent = `${startAge}歳時の取り崩し額`;
    document.getElementById('age-label').textContent = `現在の年齢 (${startAge}〜${targetAge - 1}歳)`;

    const ageEl = document.getElementById('age');
    ageEl.value = startAge;
    ageEl.min = startAge;
    ageEl.max = targetAge - 1;

    mValues = Object.keys(calcConfig.models).sort((a, b) => parseFloat(a) - parseFloat(b));
    
    // テーブル行の生成
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';
    mValues.forEach(m => {
      const mId = m.replace('.', '_');
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="m-label">x${m}</td>
        <td id="m_val_${mId}">${calcConfig.base_spends[m]}万</td>
        <td class="val-a" id="a_val_${mId}">---</td>
        <td class="val-p" id="p_val_${mId}">---</td>
      `;
      tbody.appendChild(row);
    });

    // デフォルトの予定支出額を設定 (x1.0 の値)
    const expectedSpendEl = document.getElementById('expectedSpend');
    if (calcConfig.base_spends["1.0"]) {
      expectedSpendEl.value = calcConfig.base_spends["1.0"];
    } else if (calcConfig.base_spends["1"]) {
      expectedSpendEl.value = calcConfig.base_spends["1"];
    }

    // モデルの読み込み
    const promises = mValues.map(async (m) => {
      try {
        const mResponse = await fetch(`data/${dataKey}/${calcConfig.models[m]}`);
        const mData = await mResponse.json();
        predictors[m] = new DPOptimalStrategyPredictor(mData);
      } catch (e) {
        console.error(`Failed to load model M=${m}:`, e);
      }
    });
    
    await Promise.all(promises);
    calculateAll();

  } catch (e) {
    console.error("Initialization failed", e);
  }
}

function calculateAll() {
  const age = parseInt(document.getElementById('age').value);
  const wealth = parseFloat(document.getElementById('wealth').value);
  const expectedSpend = parseFloat(document.getElementById('expectedSpend').value);

  if (isNaN(age) || isNaN(wealth) || isNaN(expectedSpend) || mValues.length === 0) return;

  mValues.forEach(m => {
    const predictor = predictors[m];
    const mId = m.replace('.', '_');
    const aEl = document.getElementById(`a_val_${mId}`);
    const pEl = document.getElementById(`p_val_${mId}`);
    const mValEl = document.getElementById(`m_val_${mId}`);
    
    if (!predictor) return;
    
    const levelMultiplier = parseFloat(m);
    const targetExpectedYN = expectedSpend * levelMultiplier;

    if (mValEl) {
      // 今年(N歳)の予定支出から、開始年齢時点の支出に逆算して表示する
      const multiplierBaseToN = predictor.getSpendMultiplier(startAge, age);
      const spendAtBase = targetExpectedYN / multiplierBaseToN;
      mValEl.textContent = Math.round(spendAtBase) + '万';
    }

    if (!predictor.data[age]) {
      if (aEl) aEl.textContent = 'N/A';
      if (pEl) pEl.textContent = 'N/A';
      return;
    }

    // A の計算
    const aOpt = predictor.getAOptWithWinningThreshold(age, wealth, targetExpectedYN);
    
    // P の計算
    const sRate = targetExpectedYN / wealth;
    const pSurv = predictor.predictPSurv(age, sRate);

    if (aEl) aEl.textContent = (aOpt * 100).toFixed(1) + '%';
    if (pEl) pEl.textContent = (pSurv * 100).toFixed(1) + '%';
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await init();
  ['age', 'wealth', 'expectedSpend'].forEach(id => {
    document.getElementById(id).addEventListener('input', calculateAll);
  });
});
