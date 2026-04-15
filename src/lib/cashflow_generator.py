"""
キャッシュフロー（現金収入・支出）のジェネレータ。
年金、一時的な収入/支出、死亡率に基づく確率的なイベントなどの
絶対的な名目金額を生成する。
"""

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import numpy as np


class CashflowType(Enum):
  """
  キャッシュフローの種類。
  """
  # --- REGULAR ---
  # 年間支出計算の「定常予算」に含めるキャッシュフロー。
  # 例：公的年金、定期的な生命保険料、安定した副収入など。
  # 
  # 特徴：
  # - DynamicSpending の「前年支出に基づく制限（天井・床）」の基準値に影響する。
  # - DynamicRebalance の「目標年間支出（ポートフォリオに要求する利回り）」を増減させる。
  # - 収入（正の値）の場合、実質的な「生活費」を押し下げる効果がある。
  # 
  # 注意点・落とし穴：
  # - 「資産が減った時だけ働く」ような変動の激しい労働収入を REGULAR に設定すると、
  #   働いている年だけ「生活費が極端に低い」と判定される。
  #   その結果、翌年の DynamicSpending が過剰に支出を抑制したり、
  #   労働を継続すべきかどうかの閾値判定が狂う（生活費が低いので安心だと誤認する）
  #   というフィードバックループ（副作用）が発生する可能性がある。
  REGULAR = auto()

  # --- EXTRAORDINARY ---
  # 独立したキャッシュフローとして扱う「臨時」の収支。
  # 例：相続、車の購入、一時的なお祝い金、緊急の医療費など。
  # 
  # 特徴：
  # - ポートフォリオの総資産残高（純資産）には即座に反映される。
  # - しかし、DynamicSpending や DynamicRebalance が参照する「定常的な生活費」
  #   の計算からは除外される（去年の実績にはカウントされない）。
  # 
  # 使い分けの指針：
  # - その収支が「将来の生活スタイルの基準」を左右するかどうかで判断する。
  # - 一時的なイベントであれば EXTRAORDINARY を、家計の基礎体力の一部であれば
  #   REGULAR を選択するのが基本。
  EXTRAORDINARY = auto()

# 追加キャッシュフローの倍率（条件付き労働など）を決めるコールバック関数
# 引数:
#   - m: 経過月数
#   - net_worth: 現在の純資産 (n_sim,)
#   - prev_net_ann_spend: 前年の正味年間支出（定常支出 - REGULARな収入）(n_sim,)
#   - prev_gross_ann_spend: 前年の総年間支出（収入を差し引く前の支出）(n_sim,)
# 戻り値:
#   - multiplier: キャッシュフローに乗じる倍率 (n_sim,)
ExtraCashflowMultiplierFn = Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class CashflowRule:
  """
  追加キャッシュフローのルール定義。
  """
  source_name: str
  cashflow_type: CashflowType
  multiplier_fn: Optional[ExtraCashflowMultiplierFn] = None


class CashflowConfig(ABC):
  """
  キャッシュフロー生成の基底クラス。
  """

  def __init__(self, name: str):
    self.name = name

  @abstractmethod
  def generate(self, n_sim: int, n_months: int,
               monthly_prices: Dict[str, np.ndarray]) -> np.ndarray:
    """
    キャッシュフローを生成する。
    戻り値は (n_months,) または (n_sim, n_months) の numpy array。
    正の値は収入、負の値は支出を表す。
    """
    pass


class PensionConfig(CashflowConfig):
  """
  年金や定期的な収入のキャッシュフロー設定。
  指定した月から毎月一定額のキャッシュフローを発生させる。
  cpi_name を指定すると、物価連動（インフレ調整）されたキャッシュフローを生成する。
  """

  def __init__(self,
               name: str,
               amount: float,
               start_month: int,
               end_month: Optional[int] = None,
               cpi_name: Optional[str] = None):
    """
    Args:
      name: このキャッシュフローの名前（後で Strategy で指定するキーとなる）
      amount: 毎月発生する一定の金額（万円など、全体の単位に合わせる）。正の値は収入、負の値は支出。
      start_month: 金額が発生し始める月（0始まり）。例えば 240 を指定すると、20年目から発生する。
      end_month: (オプション) 金額が発生しなくなる月（0始まり）。指定した月以降は発生しない。
      cpi_name: (オプション) 物価連動させるための CPI パスの名前。
    """
    super().__init__(name)
    self.amount = amount
    self.start_month = start_month
    self.end_month = end_month
    self.cpi_name = cpi_name

  def generate(self, n_sim: int, n_months: int,
               monthly_prices: Dict[str, np.ndarray]) -> np.ndarray:
    cf = np.zeros(n_months, dtype=np.float64)
    start = max(0, self.start_month)
    end = min(n_months, self.end_month) if self.end_month is not None else n_months
    
    if start < end:
      cf[start:end] = self.amount

    if self.cpi_name:
      if self.cpi_name not in monthly_prices:
        raise ValueError(f"CPI path '{self.cpi_name}' not found in monthly_prices.")
      # monthly_prices[cpi_name] is shape (n_sim, n_months + 1)
      # We take the first n_months elements for the month transitions
      cpi_array = monthly_prices[self.cpi_name][:, :n_months]
      return cf * cpi_array  # Broadcasts to (n_sim, n_months)

    return cf


class MortalityConfig(CashflowConfig):
  """
  死亡率に基づく確率的なイベントのキャッシュフロー設定。
  毎月サイコロを振り、死亡した月に payout の額をキャッシュフローとして発生させる。
  """

  def __init__(self,
               name: str,
               mortality_rates: List[float],
               initial_age: int,
               payout: float = 1_000_000.0):
    """
    Args:
      name: このキャッシュフローの名前
      mortality_rates: 各年齢での1年以内の死亡確率のリスト（0.0〜1.0）。
        インデックスが年齢に対応する（例: mortality_rates[60] は60歳の死亡確率）。
        通常は src.lib.life_table の MALE_MORTALITY_RATES などを指定する。
      initial_age: シミュレーション開始時の年齢。
      payout: 死亡時に発生させるキャッシュフローの金額（成功条件を満たすための巨大な収入など）。
    """
    super().__init__(name)
    self.mortality_rates = mortality_rates
    self.initial_age = initial_age
    self.payout = payout

  def generate(self, n_sim: int, n_months: int,
               monthly_prices: Dict[str, np.ndarray]) -> np.ndarray:
    cf = np.zeros((n_sim, n_months), dtype=np.float64)
    for m in range(n_months):
      age = self.initial_age + m // 12
      if age < len(self.mortality_rates):
        yearly_prob = self.mortality_rates[age]
        # 月次死亡率の計算: 1 - (1 - yearly_prob)^(1/12)
        monthly_prob = 1.0 - (1.0 - yearly_prob)**(1.0 / 12.0)
        dies = np.random.rand(n_sim) < monthly_prob
        cf[dies, m] = self.payout
      else:
        # 生命表の最大年齢を超えた場合は全員死亡したものとして扱う
        cf[:, m] = self.payout
    return cf


class SuddenSpendConfig(CashflowConfig):
  """
  一時的な支出（または収入）のキャッシュフロー設定。
  指定した月に一度だけ指定額のキャッシュフローを発生させる。
  負の値を指定すると支出になる。
  """

  def __init__(self, name: str, amount: float, month: int):
    """
    Args:
      name: このキャッシュフローの名前
      amount: 指定した月に発生する金額。正の値は収入、負の値は支出。
      month: 金額が発生する月（0始まり）。
    """
    super().__init__(name)
    self.amount = amount
    self.month = month

  def generate(self, n_sim: int, n_months: int,
               monthly_prices: Dict[str, np.ndarray]) -> np.ndarray:
    cf = np.zeros(n_months, dtype=np.float64)
    if 0 <= self.month < n_months:
      cf[self.month] = self.amount
    return cf


def generate_cashflows(configs: List[CashflowConfig],
                       monthly_prices: Dict[str, np.ndarray], n_sim: int,
                       n_months: int) -> Dict[str, np.ndarray]:
  """
  複数の CashflowConfig に基づいて、名前付きのキャッシュフロー配列を生成する。
  
  Returns:
    Dict[str, np.ndarray]: 各キャッシュフローの名前をキー、生成された配列を値とする辞書。
  """
  result = {}
  for config in configs:
    result[config.name] = config.generate(n_sim, n_months, monthly_prices)
  return result
