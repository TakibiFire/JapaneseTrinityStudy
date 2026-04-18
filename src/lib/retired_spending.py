"""
年齢階級別の支出データに基づき、各年齢の支出倍率を計算するライブラリ。
"""
from enum import Enum, auto
from typing import List

import numpy as np
from scipy.interpolate import CubicSpline

from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES

# 生命表に基づき推計された75歳以上の平均年齢 (calculate_average_age_75plus() の結果)
AVERAGE_AGE_75PLUS = 83.88316897574647


class SpendingType(Enum):
  """支出の種類を表す列挙型。"""
  CONSUMPTION = auto()  # 消費支出 (生活費)
  NON_CONSUMPTION = auto()  # 非消費支出 (税・保険料)
  NON_CONSUMPTION_EXCLUDE_PENSION = auto()  # 非消費支出 (年金保険料を除く)


# 家計調査報告のデータポイント
# https://www.stat.go.jp/data/kakei/sokuhou/tsuki/pdf/fies_gaikyo2024.pdf
BASE_AGES = np.array([34.4, 44.8, 54.1, 67.5, 72.5, AVERAGE_AGE_75PLUS])
NON_CONSUMPTION_DATA = np.array([90018, 129607, 141647, 41405, 34824, 30558])
CONSUMPTION_DATA = np.array([280544, 331526, 359951, 311281, 269015, 242840])


def calculate_average_age_75plus() -> float:
  """
  生命表データを用いて75歳以上の平均年齢を推計する。
  """
  m_surv = [1.0]
  f_surv = [1.0]
  for m in MALE_MORTALITY_RATES:
    m_surv.append(m_surv[-1] * (1 - m))
  for f in FEMALE_MORTALITY_RATES:
    f_surv.append(f_surv[-1] * (1 - f))

  pop_sum = 0.0
  age_sum = 0.0
  # 75歳から105歳まで
  for x in range(75, len(MALE_MORTALITY_RATES)):
    pop = (m_surv[x] + f_surv[x]) / 2.0
    pop_sum += pop
    age_sum += pop * (x + 0.5)

  return age_sum / pop_sum


def get_retired_spending_values(spending_types: List[SpendingType],
                               target_ages: np.ndarray) -> np.ndarray:
  """
  指定された年齢層の支出の絶対値（円）を返す。

  Args:
    spending_types: 支出の種類
    target_ages: 対象年齢の numpy 配列

  Returns:
    支出の絶対値の numpy 配列。
  """
  # 仮想データポイントの追加 (端部の安定化)
  last_age = BASE_AGES[-1]
  virtual_age = last_age + (last_age - BASE_AGES[-2])

  total_values = np.zeros(len(target_ages), dtype=float)

  for st in spending_types:
    if st == SpendingType.CONSUMPTION:
      virtual_con = CONSUMPTION_DATA[-1] * 0.9
      ages = np.append(BASE_AGES, virtual_age)
      con_full = np.append(CONSUMPTION_DATA, virtual_con)
      cs_con = CubicSpline(ages, con_full, bc_type='natural')
      total_values += np.maximum(cs_con(target_ages), virtual_con)
    elif st == SpendingType.NON_CONSUMPTION:
      virtual_non_con = NON_CONSUMPTION_DATA[-1] * 0.9
      ages = np.append(BASE_AGES, virtual_age)
      non_con_full = np.append(NON_CONSUMPTION_DATA, virtual_non_con)
      cs_non_con = CubicSpline(ages, non_con_full, bc_type='natural')
      total_values += np.maximum(cs_non_con(target_ages), virtual_non_con)
    elif st == SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION:
      # 元の家計調査データ（54.1歳: 141,647円など）は「世帯主が勤めている世帯（二人以上）」のデータです。
      # この中には、世帯主の厚生年金保険料が含まれています。配偶者は第3号被保険者と仮定するため、直接の支払いはありません。
      # 世帯主の年収を一定の500万円と仮定した場合、年金保険料の計算は以下の通りです。
      # * 厚生年金保険料率は固定で18.3%。労使折半により、従業員（世帯主）負担は 9.15%。
      # * 年間保険料 = 5,000,000円 × 9.15% = 457,500円
      # * 月額保険料 = 457,500円 ÷ 12ヶ月 = 38,125円
      #
      # スプライン補間を滑らかにするため、以下の手順で補間用データを作成します：
      # 1. 元の非消費支出データから 60 歳時点の値を推計し、そこから 38,125 円を引いた点を新たなデータポイントとして追加。
      # 2. 60 歳未満の既存データポイント（34.4歳、44.8歳、54.1歳）から 38,125 円を引く。
      # 3. 67.5 歳以降のデータポイントは元の値を使用。
      virtual_non_con = NON_CONSUMPTION_DATA[-1] * 0.9
      ages_orig = np.append(BASE_AGES, virtual_age)
      non_con_full = np.append(NON_CONSUMPTION_DATA, virtual_non_con)
      cs_orig = CubicSpline(ages_orig, non_con_full, bc_type='natural')

      val_at_60_adj = float(cs_orig(60.0)) - 38125.0

      ex_ages = np.insert(BASE_AGES, 3, 60.0)
      ex_values = np.insert(NON_CONSUMPTION_DATA.astype(float), 3, val_at_60_adj)
      ex_values[:3] -= 38125.0

      virtual_non_con_ex = ex_values[-1] * 0.9
      ages_ex = np.append(ex_ages, virtual_age)
      values_ex = np.append(ex_values, virtual_non_con_ex)
      cs_ex = CubicSpline(ages_ex, values_ex, bc_type='natural')
      total_values += np.maximum(cs_ex(target_ages), virtual_non_con_ex)

  return total_values


def get_retired_spending_multipliers(spending_types: List[SpendingType],
                                     start_age: int,
                                     num_years: int = 50,
                                     normalize: bool = True) -> List[float]:
  """
  指定された開始年齢からの支出の推移を返す。

  Args:
    spending_types: 支出の種類 (CONSUMPTION, NON_CONSUMPTION, NON_CONSUMPTION_EXCLUDE_PENSION) のリスト
    start_age: 開始年齢
    num_years: 取得する年数 (デフォルト 50)
    normalize: 開始年齢時を1.0とするかどうか (デフォルト True)

  Returns:
    支出（月額、単位：円）または倍率のリスト。
  """
  target_ages = np.arange(start_age, start_age + num_years)
  values = get_retired_spending_values(spending_types, target_ages)

  if normalize:
    # 開始年齢の値を 1.0 とした倍率を返す
    return (values / values[0]).tolist()
  return values.tolist()
