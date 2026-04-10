/**
 * 最適オルカン比率計算機 (JavaScript 版)
 * 
 * 使い方:
 * 1. ブラウザのコンソール (F12) を開く
 * 2. 下記のコードを貼り付けて Enter
 * 3. calculateOptimalStrategy(支出率, 年数) を呼び出す
 *    例: calculateOptimalStrategy(0.04, 30) // 支出4%, 30年
 * 
 * アップデート内容:
 * - インフレ率 1.77% への対応
 * - 60年シミュレーションデータに基づく近似式の刷新
 * - N_ruin 付近の精度を向上させるための重み付きフィッティングを採用
 */

function calculateOptimalStrategy(S, N, logCallback = console.log) {
  const r_base = 0.04;
  const tax = 0.20315;
  const inflation = 0.0177;

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

  const n = N / 60.0;
  const m = (N - n_ruin) / 60.0;

  // ガード付きの対数関数
  const safeLog = (val, epsilon = 0.0001) => Math.log(Math.max(val, epsilon));

  let ratio, prob;
  if (N <= n_ruin) {
    // Region 1: 資産寿命内
    // g_ratio(S, n) = -0.8634 -0.7437 * log(n*S) +0.2169 * n^2 +0.0505 * 1/n -2.1119 * exp(S)
    ratio = -0.8634 - 0.7437 * safeLog(n * S) + 0.2169 * (n * n) + 0.0505 * (1 / Math.max(n, 0.001)) - 2.1119 * Math.exp(S);
    
    // g_prob(S, n)  = -0.0004 +0.0178 * 1/(n*S) -0.0006 * log(S) +0.0017 * exp(n) +0.0089 * n/S
    prob = -0.0004 + 0.0178 * (1 / Math.max(n * S, 0.0001)) - 0.0006 * safeLog(S) + 0.0017 * Math.exp(n) + 0.0089 * (n / S);
  } else {
    // Region 2: 資産寿命超
    // h_ratio(S, m) = -0.1543 +0.1227 * log(m) +0.6476 * log(S) -0.0538 * 1/(n*S) -1.4565 * log(n*S)
    ratio = -0.1543 + 0.1227 * safeLog(m) + 0.6476 * safeLog(S) - 0.0538 * (1 / Math.max(n * S, 0.0001)) - 1.4565 * safeLog(n * S);
    
    // h_prob(S, m)  = -0.8682 -0.2673 * sqrt(m) -0.3769 * log(S) +0.0003 * 1/m +0.0074 * 1/(n*S)
    prob = -0.8682 - 0.2673 * Math.sqrt(Math.max(m, 0)) - 0.3769 * safeLog(S) + 0.0003 * (1 / Math.max(m, 0.001)) + 0.0074 * (1 / Math.max(n * S, 0.0001));
  }

  const clampedRatio = Math.max(0, Math.min(1, ratio));
  const clampedProb = Math.max(0, Math.min(1, prob));

  logCallback(`--- 最適戦略の計算結果 ---`);
  logCallback(`支出率: ${(S * 100).toFixed(2)}%`);
  logCallback(`目標寿命: ${N}年`);
  logCallback(`資産寿命 (無リスクのみ): ${n_ruin.toFixed(1)}年`);
  logCallback(`推奨オルカン比率: ${(clampedRatio * 100).toFixed(1)}%`);
  logCallback(`期待生存確率: ${(clampedProb * 100).toFixed(1)}%`);
  logCallback(`--------------------------`);

  return { ratio: clampedRatio, probability: clampedProb, ruinYear: n_ruin };
}
