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
  INCLUDE_IN_ANNUAL_SPEND = auto()  # 年間支出計算に含める（例：年金）
  ISOLATED = auto()                # 独立したキャッシュフローとして扱う（例：一時的な支出）


# 追加キャッシュフローの倍率（条件付き労働など）を決めるコールバック関数
# (month, current_net_worth, previous_annual_spending) -> multiplier
ExtraCashflowMultiplierFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


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
