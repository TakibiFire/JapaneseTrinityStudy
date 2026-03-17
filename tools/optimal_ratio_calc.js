/**
 * 最適オルカン比率計算機 (JavaScript 版)
 * 
 * 使い方:
 * 1. ブラウザのコンソール (F12) を開く
 * 2. 下記のコードを貼り付けて Enter
 * 3. calculateOptimalRatio(支出率, 年数) を呼び出す
 *    例: calculateOptimalRatio(0.04, 30) // 支出4%, 30年
 */

function calculateOptimalStrategy(S, N, logCallback = console.log) {
  const r_base = 0.04;
  const tax = 0.20315;
  const inflation = 0.02;

  // 実質利回り (税引後利回り - インフレ率)
  const r_eff = r_base * (1.0 - tax);
  const i_ln = Math.log(1.0 + inflation);
  const delta = r_eff - i_ln;

  // 1. 資産寿命 (N_ruin) の計算
  let n_ruin;
  if (S <= delta) {
    n_ruin = 999.0;
  } else {
    n_ruin = Math.log(1.0 - delta / S) / (-delta);
  }

  const n = N / 50.0;
  const m = (N - n_ruin) / 50.0;

  let ratio, prob;
  if (N <= n_ruin) {
    // Region 1: 資産寿命内
    ratio = -0.8088 - 0.3832 * Math.log(n * S) + 0.1134 * (1 / n) - 0.2017 * Math.log(S) - 1.4146 * Math.exp(S);
    prob = 1.0;
  } else {
    // Region 2: 資産寿命超
    ratio = +0.6431 + 0.1640 * Math.log(Math.max(m, 0.0001)) - 0.0194 * (1 / S) + 0.8301 * Math.exp(S) + 0.2235 * Math.sqrt(m);
    prob = -0.2858 + 0.0218 * (1 / (n * S)) + 0.2369 * Math.log(n) - 0.0325 * Math.log(Math.max(m, 0.0001)) + 0.0035 * (n / S);
  }

  const clampedRatio = Math.max(0, Math.min(1, ratio));
  const clampedProb = Math.max(0, Math.min(1, prob));

  logCallback(`--- 最適戦略の計算結果 ---`);
  logCallback(`支出率: ${(S * 100).toFixed(2)}%`);
  logCallback(`目標年数: ${N}年`);
  logCallback(`資産寿命 (無リスクのみ): ${n_ruin.toFixed(1)}年`);
  logCallback(`推奨オルカン比率: ${(clampedRatio * 100).toFixed(1)}%`);
  logCallback(`期待生存確率: ${(clampedProb * 100).toFixed(1)}%`);
  logCallback(`--------------------------`);

  return { ratio: clampedRatio, probability: clampedProb, ruinYear: n_ruin };
}
