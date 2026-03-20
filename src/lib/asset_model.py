"""
アセットモデリングに関する統計的フィッティング、モデル評価、各種指標計算を行うライブラリ。
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

logger = logging.getLogger(__name__)

# フィッティングを試行する主要な連続分布のリスト
# 過去の広範な探索結果(data/top_distributions.txt)から抽出された上位分布のユニオン
ALL_DISTRIBUTIONS = [
    stats.alpha, stats.burr, stats.cauchy, stats.dgamma, stats.dweibull,
    stats.genlogistic, stats.gennorm, stats.hypsecant, stats.invgamma,
    stats.johnsonsu, stats.laplace, stats.loglaplace, stats.t, stats.norm
]


def process_returns(df: pd.DataFrame, freq: str) -> pd.DataFrame:
  """
  日次または月次のリターンを計算する
  
  Args:
    df (pd.DataFrame): 価格データを含むデータフレーム。Dateカラムをインデックスにする。
    freq (str): リターンの計算頻度 ('D' = 日次, 'ME' = 月次)
  """
  df = df.copy()
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.set_index('Date')

  if freq == 'M' or freq == 'ME':
    # 月末のデータを使用する
    df = df.resample('ME').last()

  returns = pd.DataFrame(index=df.index)
  for col in ['SP500', 'ACWI', 'BTC']:
    if col not in df.columns:
      continue
    valid_data = df[col].dropna()
    simple_ret = valid_data / valid_data.shift(1) - 1
    log_ret = np.log(valid_data / valid_data.shift(1))

    returns[f'{col}_simple'] = simple_ret
    returns[f'{col}_log'] = log_ret

  return returns


def find_best_distribution(data: pd.Series, bins: int = 100, top_n: int = 1):
  """
  与えられたデータに対してALL_DISTRIBUTIONS内のすべての連続分布をフィッティングし、
  ヒストグラムとのMean Squared Error (MSE)が小さい「上位」分布を探索する。

  Args:
    data (pd.Series): フィッティング対象の一次元データ（例：対数リターン）。NaNは自動で除外される。
    bins (int, optional): MSE計算時にヒストグラムを作成する際のビン数。デフォルトは100。
    top_n (int, optional): 返す上位モデルの数。デフォルトは1。

  Returns:
    list or None: 最適な分布の情報を格納した辞書のリスト。データが空またはすべてのフィッティングに失敗した場合はNone。
    キー構成:
    - 'name' (str): 分布名（例：'norm'）
    - 'params' (tuple): フィットされたパラメータ（shape, loc, scale 等）
    - 'mse' (float): ヒストグラムとPDF間の平均二乗誤差
    - 'loglik' (float): 対数尤度
    - 'aic' (float): 赤池情報量規準
    - 'bic' (float): ベイズ情報量規準
  """
  data = data.dropna()
  if len(data) == 0:
    return None

  # ヒストグラムの作成（確率密度）
  y, x = np.histogram(data, bins=bins, density=True)
  # ビンの中央値をX軸とする
  x = (x + np.roll(x, -1))[:-1] / 2.0

  results = []
  for distribution in ALL_DISTRIBUTIONS:
    try:
      with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # フィッティングの実行
        params = distribution.fit(data)
    except Exception as e:
      logger.warning(f"Failed to fit {distribution.name}: {e}")
      continue

    # PDFの生成とMSE等の計算
    try:
      with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        pdf = distribution.pdf(x, *params)
        # 異常値(NaNやInf)のチェック
        if np.any(np.isnan(pdf)) or np.any(np.isinf(pdf)):
          continue
        # MSE
        mse = np.mean((y - pdf)**2)

        loglik = distribution.logpdf(data, *params).sum()
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik

        results.append({
            'name': distribution.name,
            'params': params,
            'mse': mse,
            'loglik': loglik,
            'aic': aic,
            'bic': bic
        })
    except Exception as e:
      logger.warning(f"Failed to evaluate {distribution.name}: {e}")
      continue

  if not results:
    return None

  # MSEが最も小さいものを選択
  results.sort(key=lambda x: x['mse'])
  return results[:top_n]


def fit_normal_simple(data: pd.Series):
  """
  モデルA: 単純リターンを正規分布にフィッティングする。

  Args:
    data (pd.Series): 単純リターンのデータ。NaNは自動で除外される。

  Returns:
    dict: フィッティング結果。
      - 'mu' (float): 平均（loc）
      - 'std' (float): 標準偏差（scale）
      - 'loglik' (float): 対数尤度
      - 'aic' (float): 赤池情報量規準
      - 'bic' (float): ベイズ情報量規準
      - 'mse' (float): 標本と平均の平均二乗誤差（分散）
  """
  data = data.dropna()
  mu, std = stats.norm.fit(data)
  loglik = stats.norm.logpdf(data, loc=mu, scale=std).sum()
  k = 2
  n = len(data)
  aic = 2 * k - 2 * loglik
  bic = k * np.log(n) - 2 * loglik
  mse = np.mean((data - mu)**2)
  return {
      'mu': mu,
      'std': std,
      'loglik': loglik,
      'aic': aic,
      'bic': bic,
      'mse': mse
  }


def fit_normal_log(data: pd.Series):
  """
  モデルB: 対数リターンを正規分布にフィッティングする。

  Args:
    data (pd.Series): 対数リターンのデータ。NaNは自動で除外される。

  Returns:
    dict: フィッティング結果。キー構成は `fit_normal_simple` と同一。
  """
  data = data.dropna()
  mu, std = stats.norm.fit(data)
  loglik = stats.norm.logpdf(data, loc=mu, scale=std).sum()
  k = 2
  n = len(data)
  aic = 2 * k - 2 * loglik
  bic = k * np.log(n) - 2 * loglik
  mse = np.mean((data - mu)**2)
  return {
      'mu': mu,
      'std': std,
      'loglik': loglik,
      'aic': aic,
      'bic': bic,
      'mse': mse
  }


def calculate_mrgbm(data: pd.Series, dt: float):
  """
  平均回帰型幾何ブラウン運動（MR-GBM）のパラメータを推定する。
  対数価格に対してオルンシュタイン＝ウーレンベック過程を仮定し、OLSにより推定を行う。
  
  dX_t = theta * (mu - X_t) dt + sigma dW_t
  ここで X_t = log(P_t)

  Args:
    data (pd.Series): 価格データ。NaNは自動で除外される。
    dt (float): 時間刻み（例：日次データで年率換算する場合は 1/365.25）

  Returns:
    dict or None: 推定されたパラメータ。データが2点未満の場合や、
                  平均回帰性が確認できない（b >= 0）場合はNaNが返されるか、あるいはNone。
    キー構成:
    - 'theta' (float): 平均回帰の速度（強さ）
    - 'mu' (float): 対数価格の長期平均
    - 'sigma' (float): ボラティリティ
  """
  data = data.dropna()
  if len(data) < 2:
    return None

  # 対数価格系列とその差分
  x = np.log(data.values)
  x_t = x[:-1]
  dx = x[1:] - x_t

  # 回帰: X_{t+1} - X_t = a + b X_t + epsilon
  A = np.vstack([x_t, np.ones(len(x_t))]).T
  b, a = np.linalg.lstsq(A, dx, rcond=None)[0]

  # b = exp(-theta * dt) - 1 => theta = -log(b+1) / dt
  # b >= 0 の場合は平均回帰しない（発散する）ためNaNを返す
  # bが正の小さい値の場合も発散する可能性があるため、閾値を設けるかそのまま計算するか。
  # 実際にはランダムウォークに近いとbが0近辺になるが、少しでも負になれば平均回帰とみなす。
  # ただし今回は確実な平均回帰を見るためb>-1e-4程度を発散扱いとする
  if b >= -1e-4:
    return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}

  theta = -np.log(b + 1) / dt
  mu = a / (-b)

  # 残差からシグマを推定
  res = dx - (a + b * x_t)
  var_res = np.var(res)
  sigma = np.sqrt(var_res * 2 * theta / (1 - np.exp(-2 * theta * dt)))

  return {
      'theta': theta,
      'mu': mu,
      'sigma': sigma,
      'residuals': pd.Series(res, index=data.index[1:])
  }


def simulate_annual_stats_simple(dist, params, n_sims: int = 1000000):
  """
  与えられた確率分布（単利リターン）から12ヶ月分のデータをサンプリングし、
  年次単利リターンの平均と標準偏差をモンテカルロシミュレーションにより計算する。

  Args:
    dist: scipy.statsの分布オブジェクト（例: stats.norm）
    params: 分布のパラメータのタプル
    n_sims (int): シミュレーション回数

  Returns:
    tuple: (mean_annual_return, std_annual_return)
  """
  monthly_sims = dist.rvs(*params, size=(n_sims, 12))
  annual_sims = np.prod(1 + monthly_sims, axis=1) - 1
  return np.mean(annual_sims), np.std(annual_sims)


def simulate_annual_stats_log(dist, params, n_sims: int = 1000000):
  """
  与えられた確率分布（対数リターン）から12ヶ月分のデータをサンプリングし、
  年次単利リターンの平均と標準偏差をモンテカルロシミュレーションにより計算する。

  Args:
    dist: scipy.statsの分布オブジェクト（例: stats.norm）
    params: 分布のパラメータのタプル
    n_sims (int): シミュレーション回数

  Returns:
    tuple: (mean_annual_return, std_annual_return)
  """
  monthly_sims = dist.rvs(*params, size=(n_sims, 12))
  annual_sims = np.exp(np.sum(monthly_sims, axis=1)) - 1
  return np.mean(annual_sims), np.std(annual_sims)
