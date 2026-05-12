"""
動的リバランスに関わる型定義。
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np


@dataclass
class DPDebugOutput:
  """DP計算のデバッグ情報を保持するクラス。"""
  pY_N: np.ndarray  # 来年の予測純支出
  P_pred: np.ndarray  # 現在の状態からの予測生存確率
  W_N: np.ndarray  # 勝利しきい値


@dataclass
class DRResult:
  """リバランス関数の戻り値。"""
  target_ratios: Dict[str, Union[float, np.ndarray]]
  debug: Optional[DPDebugOutput]


# 引数:
# 1. total_net: 現在の資産合計 (n_sim,)
# 2. cur_ann_spend: 現在の年間正味支出 (n_sim,)
# 3. rem_years: 残り年数 (scalar)
# 4. post_tax_net: 税引き後資産見積もり (n_sim,)
# 5. prev_gross_ann_spend: 前年の年間総支出 (n_sim,)
# 6. current_prices: 現在のアセット価格辞書 (n_sim,)
# 7. prev_prices: 前年のアセット価格辞書 (n_sim,)
# 8. need_debug: デバッグ情報を出力すべきパスのマスク (n_sim,)
DynamicRebalanceFn = Callable[[
    np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, Optional[Dict[
        str, np.ndarray]], Optional[Dict[str, np.ndarray]], np.ndarray
], DRResult]
