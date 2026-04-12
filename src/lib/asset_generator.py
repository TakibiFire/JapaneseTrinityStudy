"""
資産推移のシミュレーションエンジン。
分布と資産構成に基づき、一括で月次ベースの価格推移を生成する。

例:

assets = [
    Asset(name="SP500-1",
          dist=MonthlySimpleNormal(mu=0.008373, sigma=0.042125)),
    Asset(name="SP500-2", dist=YearlyLogNormal(mu=0.07, sigma=0.15)),
    Asset(name="SP500-3",
          dist=MonthlyDist(stats.genlogistic, params=(0.5975, 0.0240, 0.0171))),
    # Assetの依存関係: SP500-1 の月次リターンをベースに使う
    DerivedAsset(
        name="ACWI-APP-1", 
        base="SP500-1", 
        multiplier=1.0269, 
        noise_dist=MonthlySimpleNormal(mu=0.0, sigma=0.01)
    ),
    # Forex(為替)
    ForexAsset(name="USDJPY", dist=YearlyLogNormal(mu=0.0, sigma=0.10)),
    # 海外資産の日本円建て価値
    Asset(name="VT", dist=YearlyLogNormal(mu=0.06, sigma=0.14), forex="USDJPY")
]
"""

import dataclasses
import graphlib
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import lfilter, lfiltic


class Distribution(ABC):
  """
  月次リターンを生成するための分布の基底クラス。
  """

  @abstractmethod
  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    """
    指定されたshapeの月次リターンを生成する。

    Args:
      shape: (パス数, 月数) のタプル
      seed: 乱数シード

    Returns:
      月次リターンの2次元numpy配列
    """
    pass  # pragma: no cover


class MonthlySimpleNormal(Distribution):
  """
  単純リターンに対する正規分布 (月次パラメータを直接指定)。
  """

  def __init__(self, mu: float, sigma: float):
    self.mu = mu
    self.sigma = sigma

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(self.mu, self.sigma, size=shape)


class MonthlyLogNormal(Distribution):
  """
  対数リターンに対する正規分布 (月次パラメータを直接指定)。
  """

  def __init__(self, mu: float, sigma: float):
    self.mu = mu
    self.sigma = sigma

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=shape)
    return np.exp((self.mu - 0.5 * self.sigma**2) + self.sigma * Z) - 1.0


class YearlyLogNormal(Distribution):
  """
  対数リターンに対する正規分布 (年次パラメータを指定し、内部で月次に変換)。
  """

  def __init__(self, mu: float, sigma: float):
    self.mu = mu
    self.sigma = sigma

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=shape)
    # 年次から月次への変換
    dt = 1.0 / 12.0
    return np.exp((self.mu - 0.5 * self.sigma**2) * dt +
                  self.sigma * np.sqrt(dt) * Z) - 1.0


class YearlySimpleNormal(Distribution):
  """
  単純リターンに対する正規分布 (年次パラメータを指定し、内部で月次に変換)。

  注意:
  指定した年次パラメータから、月次ベースで単純リターンが正規分布に従うように生成します。
  ただし、理論上の欠陥として、単純正規分布を使用するとリターンが -100% (-1.0) を
  下回る確率が存在するため、資産価格がマイナス（ゼロ以下）になる可能性を排除できません
  （株価などの有限責任の資産価格では本来あり得ない挙動です）。
  特に長期や高ボラティリティのシミュレーションにおいて価格のマイナス化を防ぎたい場合で、
  なおかつ年次の「単純リターン（算術平均）」を指定したい場合は、
  幾何ブラウン運動に基づく `YearlyLogNormalArithmetic` の使用を推奨します。
  """

  def __init__(self, mu: float, sigma: float):
    self.mu = mu
    self.sigma = sigma

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 月次への変換: 12ヶ月複利で (1 + mu) になるように逆算
    monthly_mu = (1.0 + self.mu)**(1.0 / 12.0) - 1.0
    monthly_sigma = self.sigma / np.sqrt(12.0)
    return rng.normal(monthly_mu, monthly_sigma, size=shape)


class YearlyLogNormalArithmetic(Distribution):
  """
  対数リターンに対する正規分布。ただし、入力パラメータとして「算術平均（単純リターン）」と
  「算術標準偏差」を受け取り、内部で対数リターンのパラメータ（幾何ブラウン運動のドリフトと
  ボラティリティ）に変換して月次リターンを生成する。
  """

  def __init__(self, mu: float, sigma: float):
    # mu, sigma は算術（単純リターン）ベースの年次期待値とボラティリティ
    # これを対数リターンのパラメータ (mu_log, sigma_log) に変換する
    # 1 + mu = exp(mu_log + sigma_log^2 / 2)
    # sigma^2 / (1+mu)^2 = exp(sigma_log^2) - 1

    # 1. sigma_log の計算
    self.sigma_log = np.sqrt(np.log(1.0 + (sigma / (1.0 + mu))**2))

    # 2. mu_log の計算
    self.mu_log = np.log(1.0 + mu) - 0.5 * self.sigma_log**2

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=shape)
    # 年次から月次への変換
    dt = 1.0 / 12.0
    # ここでの mu_log はすでに -0.5*sigma^2 が引かれた単なる期待値ではないので、
    # 幾何ブラウン運動のドリフト項としてはそのまま使用する。
    # つまり、対数リターンの期待値は mu_log、分散は sigma_log^2。
    # E[ln(S_t/S_0)] = mu_log * t, Var[ln(S_t/S_0)] = sigma_log^2 * t
    return np.exp(self.mu_log * dt + self.sigma_log * np.sqrt(dt) * Z) - 1.0


class MonthlyDist(Distribution):
  """
  任意の scipy.stats 分布をラップするクラス (月次パラメータを直接指定)。

  注意:
  このクラスは生成した乱数を「そのまま単利リターン」として扱います。
  パラメータが対数リターン (Log Return) にフィットされたものである場合は、
  このクラスではなく `MonthlyLogDist` を使用してください。
  (対数リターンをそのまま単利として用いると、複利計算の過程で年率数%の
  下方へのズレが生じ、シミュレーション結果が破綻します)
  """

  def __init__(self, dist_func: Any, params: Tuple[Any, ...]):
    self.dist_func = dist_func
    self.params = params

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    return self.dist_func.rvs(*self.params, size=shape, random_state=seed)


class MonthlyLogDist(Distribution):
  """
  任意の scipy.stats 分布をラップするクラス。
  生成した乱数を「対数リターン」として扱い、出力時に単利リターンに変換します。

  このクラスは、過去の対数リターンデータにフィットさせた分布
  （例: stats.johnsonsu など）を用いてシミュレーションを行う際に使用します。
  出力される単利リターン R_t は、対数リターン r_t に対して R_t = exp(r_t) - 1 となります。
  """

  def __init__(self, dist_func: Any, params: Tuple[Any, ...]):
    self.dist_func = dist_func
    self.params = params

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    log_ret = self.dist_func.rvs(*self.params, size=shape, random_state=seed)
    return np.exp(log_ret) - 1.0


class MonthlyARLogNormal(Distribution):
  """
  月次対数リターンに対する AR(p) モデルに基づく正規分布。
  r_t = c + phi_1 * r_{t-1} + ... + phi_p * r_{t-p} + e_t
  e_t ~ N(0, sigma_e^2)
  出力は単利リターン (exp(r_t) - 1) に変換されます。
  """

  def __init__(self,
               c: float,
               phis: Sequence[float],
               sigma_e: float,
               initial_y: Optional[Sequence[float]] = None):
    """
    Args:
      c: 定数項
      phis: 自己相関係数のリスト [phi_1, phi_2, ..., phi_p]
      sigma_e: 誤差項の標準偏差
      initial_y: 初期値（対数リターン）のリスト [y_{t-p}, ..., y_{t-1}]
    """
    self.c = c
    self.phis = np.array(phis)
    self.p = len(phis)
    self.sigma_e = sigma_e
    self.initial_y = initial_y

  def generate(self, shape: Tuple[int, int], seed: int) -> np.ndarray:
    """
    指定されたパス数と月数でAR(p)過程に従うリターンを生成する。

    Args:
      shape: (パス数, 月数)
      seed: 乱数シード
    """
    rng = np.random.default_rng(seed)
    n_paths, n_months = shape

    # y_t - phi_1 y_{t-1} - ... = c + e_t
    a = np.concatenate(([1.0], -self.phis))
    b = np.array([1.0])

    if self.initial_y is not None:
      if len(self.initial_y) < self.p:
        raise ValueError(f"initial_y must have at least {self.p} elements.")
      # 直近の p 個を使用し、lfiltic 用に逆順にする
      y_hist = np.array(self.initial_y[-self.p:])[::-1]
      x_hist = np.full(self.p, self.c)
      zi_single = lfiltic(b, a, y=y_hist, x=x_hist)
      zi = np.tile(zi_single, (n_paths, 1))

      e = rng.normal(0, self.sigma_e, size=(n_paths, n_months))
      x = self.c + e
      r, _ = lfilter(b, a, x, axis=-1, zi=zi)
    else:
      # 初期値が指定されない場合は burn-in (100ヶ月) を用いて
      # 初期状態への依存を消し、定常分布からスタートさせる
      burn_in = 100
      total_months = n_months + burn_in
      e = rng.normal(0, self.sigma_e, size=(n_paths, total_months))
      x = self.c + e
      r_full = lfilter(b, a, x, axis=-1)
      r = r_full[:, burn_in:]

    return np.exp(r) - 1.0


@dataclasses.dataclass
class AssetConfig:
  """
  資産設定の基底クラス。
  """
  name: str
  trust_fee: float = 0.0  # 年率の信託報酬 (例: 0.1%なら0.001)
  leverage: float = 1.0  # レバレッジ倍率
  forex: Optional[str] = None  # 適用する為替レートの名前


@dataclasses.dataclass
class Asset(AssetConfig):
  """
  独立して生成される資産クラス。
  """
  dist: Optional[Distribution] = None


@dataclasses.dataclass
class DerivedAsset(AssetConfig):
  """
  他の資産(ベース)の推移に依存して生成される資産クラス。

  ベース資産のリターンに multiplier を掛け、noise_dist によるノイズを加算して
  派生資産のリターンを生成します。

  Attributes:
    base: ベースとなる資産の名前
    multiplier: ベースリターンに乗じる係数
    noise_dist: 追加するノイズ（残差）を生成する分布
    log_correlation: 
        ベース資産と派生資産の相関関係が「対数リターン」上で計算されたものかを指定します。
        True の場合：
          1. base_ret（単利）を対数リターン log(1 + base_ret) に変換。
          2. 対数の世界で (log_base_ret * multiplier + log_noise) の加算を行う。
             （※ここで noise_dist は対数リターン上の残差を直接返すことを前提とするため、
             　noise_dist には MonthlyDist など単利変換を行わないクラスを指定してください）
          3. 最後に exp() - 1 で全体の単利リターンに戻す。
        False の場合（デフォルト）：
          単純に単利リターン上で (base_ret * multiplier + noise) の加算を行います。
  """
  base: Optional[str] = None
  multiplier: float = 1.0
  noise_dist: Optional[Distribution] = None
  log_correlation: bool = False


@dataclasses.dataclass
class ForexAsset:
  """
  為替レートの推移を生成するクラス。
  """
  name: str
  dist: Distribution


@dataclasses.dataclass
class CpiAsset:
  """
  消費者物価指数（インフレ率）の推移を生成するクラス。
  """
  name: str
  dist: Distribution


@dataclasses.dataclass
class SlideAdjustedCpiAsset:
  """
  マクロ経済スライドによる調整をシミュレーションするためのCPI資産。
  ベースとなるCPIに対して、増加分を一定率(slide_rate)抑制しつつ、
  名目額が前月を下回らない（名目下限措置）ように調整される。

  計算式:
  月次調整リターン = max(0, ベースCPIの月次リターン - (年率調整率 / 12))
  
  これにより、インフレ時には年金の上昇が抑制され（実質価値の目減り）、
  デフレ時でも名目額が維持される挙動を再現します。
  """
  name: str
  base_cpi: str
  slide_rate: float  # 年率の調整率（例: 0.005）


AssetConfigType = Union[Asset, DerivedAsset, ForexAsset, CpiAsset,
                        SlideAdjustedCpiAsset]


def generate_monthly_asset_prices(configs: Sequence[AssetConfigType],
                                  n_paths: int, n_months: int,
                                  seed: int) -> Dict[str, np.ndarray]:
  """
  指定された設定のリストから、各資産の月次価格推移を一括で生成する。

  graphlibのTopologicalSorterを用いて依存関係（baseおよびforex）を解決し、
  生成可能な順序で月次リターンと価格の累積積を計算する。

  Args:
    configs: 資産および為替設定のリスト
    n_paths: シミュレーションのパス数
    n_months: シミュレーションの月数
    seed: 乱数シード

  Returns:
    資産名(または為替名)をキーとし、shape=(n_paths, n_months + 1)の価格推移配列を値とする辞書。
    初期の0ヶ月目 (倍率 1.0) を含む。
  """
  # ---------------------------------------------------------------------------
  # 依存関係のグラフ構築 (graphlib.TopologicalSorter を利用)
  # ---------------------------------------------------------------------------
  config_map = {config.name: config for config in configs}
  ts: graphlib.TopologicalSorter[str] = graphlib.TopologicalSorter()

  for config in configs:
    deps = []

    # DerivedAsset は base に依存
    if isinstance(config, DerivedAsset):
      if config.base is None:
        raise ValueError(
            f"DerivedAsset '{config.name}' must specify a base asset.")
      if config.base not in config_map:
        raise ValueError(
            f"Base asset '{config.base}' for derived asset '{config.name}' not found."
        )
      deps.append(config.base)

    # SlideAdjustedCpiAsset は base_cpi に依存
    if isinstance(config, SlideAdjustedCpiAsset):
      if config.base_cpi not in config_map:
        raise ValueError(
            f"Base CPI '{config.base_cpi}' for slide adjusted CPI '{config.name}' not found."
        )
      deps.append(config.base_cpi)

    # forex を指定している場合はそれに依存
    if isinstance(config, AssetConfig) and config.forex:
      if config.forex not in config_map:
        raise ValueError(
            f"Forex '{config.forex}' for asset '{config.name}' not found.")
      deps.append(config.forex)

    # グラフに追加（依存関係がある場合は deps をアンパック）
    ts.add(config.name, *deps)

  # トポロジカルソートを実行（循環依存があればここで graphlib.CycleError が発生）
  execution_order = list(ts.static_order())

  returns: Dict[str, np.ndarray] = {}
  final_prices: Dict[str, np.ndarray] = {}

  # ---------------------------------------------------------------------------
  # 順序に従って生成・計算処理
  # ---------------------------------------------------------------------------
  for name in execution_order:
    config = config_map[name]

    # 1. 月次リターンの生成
    if isinstance(config, (Asset, ForexAsset, CpiAsset)):
      dist = getattr(config, 'dist', None)
      if dist is None:
        raise ValueError(
            f"Asset, ForexAsset, or CpiAsset '{config.name}' requires a distribution (dist)."
        )
      # 資産ごとに一意で再現性のあるシードを計算 (ハッシュは環境依存の可能性があるためhashlibを使用)
      hash_str = config.name + str(seed)
      asset_seed = int(hashlib.md5(hash_str.encode('utf-8')).hexdigest(),
                       16) % (2**32)
      returns[config.name] = dist.generate((n_paths, n_months), asset_seed)

    elif isinstance(config, DerivedAsset):
      base_name = config.base
      assert base_name is not None  # 検証済み
      base_ret = returns[base_name]  # base はすでに計算済み（単利リターン）
      noise = np.zeros((n_paths, n_months))

      if config.noise_dist:
        noise_hash_str = config.name + "noise" + str(seed)
        noise_seed = int(
            hashlib.md5(noise_hash_str.encode('utf-8')).hexdigest(), 16) % (2**
                                                                            32)
        # noise_dist.generate は、log_correlation=True時は対数リターン上の残差を直接返す（MonthlyDist等を使う想定）
        # log_correlation=False時は単利上のノイズを返す想定
        noise = config.noise_dist.generate((n_paths, n_months), noise_seed)

      if config.log_correlation:
        # 相関関係が対数リターンの世界で算出されている場合の処理
        # 1. ベース資産の単利を対数に変換
        base_log_ret = np.log(1.0 + base_ret)

        # 2. 対数の世界で乗算とノイズ加算（このときnoiseは対数リターンの残差を想定）
        derived_log_ret = base_log_ret * config.multiplier + noise

        # 3. 最後に単利リターンに戻す
        returns[config.name] = np.exp(derived_log_ret) - 1.0
      else:
        # 単純な単利上での合成
        returns[config.name] = base_ret * config.multiplier + noise

    elif isinstance(config, SlideAdjustedCpiAsset):
      base_ret = returns[config.base_cpi]
      monthly_slide = config.slide_rate / 12.0
      # マクロ経済スライドの適用: max(0, base_ret - monthly_slide)
      returns[config.name] = np.maximum(0.0, base_ret - monthly_slide)

    # 2. 月次価格推移の計算 (コスト控除、レバレッジ、為替の適用)
    if isinstance(config, (ForexAsset, CpiAsset, SlideAdjustedCpiAsset)):
      # 為替やCPI自体はコスト等を持たない単純な累積積
      fx_multiplier = np.maximum(1.0 + returns[config.name], 0.0)
      prices = np.ones((n_paths, n_months + 1), dtype=np.float64)
      prices[:, 1:] = np.cumprod(fx_multiplier, axis=1)
      final_prices[config.name] = prices

    elif isinstance(config, AssetConfig):
      # ローカルリターンの月次倍率計算 (レバレッジと月次信託報酬の適用)
      monthly_trust_fee = config.trust_fee / 12.0
      local_multiplier = np.maximum(
          1.0 + config.leverage * returns[config.name] - monthly_trust_fee, 0.0)

      # 為替が指定されている場合は適用 (forex もすでに計算済みであることが保証される)
      if config.forex:
        fx_returns = returns[config.forex]
        fx_multiplier = np.maximum(1.0 + fx_returns, 0.0)
        final_multiplier = local_multiplier * fx_multiplier
      else:
        final_multiplier = local_multiplier

      # 累積積による価格の計算 (初期価格 = 1.0)
      prices = np.ones((n_paths, n_months + 1), dtype=np.float64)
      prices[:, 1:] = np.cumprod(final_multiplier, axis=1)

      final_prices[config.name] = prices

  return final_prices
