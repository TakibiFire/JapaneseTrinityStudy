"""
日本版トリニティ・スタディのシミュレーションに必要なデータクラスと関数群（新エンジン版）。

アセット生成（asset_generator.py）で生成された月次価格推移に基づき、
ポート欧の運用、リバランス、取り崩し、税金計算などを実行する。
"""

import dataclasses
from typing import Callable, Dict, List, Optional, Union, cast

import numpy as np

# ---------------------------------------------------------------------------
# 1. データ構造 (Dataclasses)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ZeroRiskAsset:
  """
  無リスク資産（価格変動なし、固定利回り）を定義する。
  """
  name: str
  yield_rate: float  # 年利 (割合)


@dataclasses.dataclass
class DynamicSpending:
  """
  ダイナミック・スペンディング（動的支出）戦略の設定。
  """
  target_ratio: float
  upper_limit: float
  lower_limit: float


# ダイナミックリバランス用のコールバック関数
DynamicRebalanceFn = Callable[[np.ndarray, np.ndarray, float],
                              Dict[str, Union[float, np.ndarray]]]

# 追加キャッシュフローの倍率（条件付き労働など）を決めるコールバック関数
ExtraCashflowMultiplierFn = Callable[[int, np.ndarray], np.ndarray]


@dataclasses.dataclass
class Strategy:
  """
  運用戦略の定義。
  """
  name: str
  initial_money: float  # 万円
  initial_loan: float  # 万円
  yearly_loan_interest: float
  initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float]
  annual_cost: Union[float, List[float], DynamicSpending]
  inflation_rate: Union[float, str, None]
  selling_priority: List[str]
  tax_rate: float = 0.20315
  rebalance_interval: int = 0
  dynamic_rebalance_fn: Optional[DynamicRebalanceFn] = None
  record_annual_spend: bool = False
  extra_cashflow_sources: Dict[str, Optional[ExtraCashflowMultiplierFn]] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if isinstance(self.annual_cost, DynamicSpending) and self.inflation_rate and self.inflation_rate != 0.0:
      raise ValueError("inflation_rate must be 0.0 or None when using DynamicSpending, as it handles limits nominally.")

    valid_names = set()
    for key in self.initial_asset_ratio.keys():
      if isinstance(key, ZeroRiskAsset):
        valid_names.add(key.name)
      else:
        valid_names.add(key)

    if set(self.selling_priority) != valid_names:
      missing = valid_names - set(self.selling_priority)
      extra = set(self.selling_priority) - valid_names
      error_msg = "selling_priority must contain all assets in initial_asset_ratio."
      if missing:
        error_msg += f" Missing: {missing}."
      if extra:
        error_msg += f" Extra assets not in ratio: {extra}."
      raise ValueError(error_msg)


@dataclasses.dataclass
class SimulationResult:
  """
  シミュレーションの実行結果。
  """
  net_values: np.ndarray  # shape: (n_sim,)
  sustained_months: np.ndarray  # shape: (n_sim,)
  annual_spends: Optional[np.ndarray] = None  # shape: (n_sim, n_years)


# ---------------------------------------------------------------------------
# 2. シミュレーションコア
# ---------------------------------------------------------------------------


def simulate_strategy(
    strategy: Strategy,
    monthly_asset_prices: Dict[str, np.ndarray],
    monthly_cashflows: Optional[Dict[str, np.ndarray]] = None,
    fallback_n_sim: int = 1000,
    fallback_total_months: int = 600) -> SimulationResult:
  """
  指定された戦略に従い、資産推移をシミュレーションする。
  
  Args:
    strategy: 戦略設定
    monthly_asset_prices: アセット生成エンジンが作成した月次価格の辞書
    monthly_cashflows: cashflow_generatorが生成した追加の月次キャッシュフローの辞書
    fallback_n_sim: 価格推移辞書が空だった場合に使用するパス数
    fallback_total_months: 価格推移辞書が空だった場合に使用する月数
  """
  local_monthly_asset_prices = dict(monthly_asset_prices)

  if not local_monthly_asset_prices:
    # 資産データがない場合（テスト用などの限定的ケース）
    n_sim = fallback_n_sim
    total_months = fallback_total_months
  else:
    sample_prices = next(iter(local_monthly_asset_prices.values()))
    n_sim, n_months_plus_one = sample_prices.shape
    total_months = n_months_plus_one - 1

  # CPIパスの参照
  cpi_multiplier_path: Optional[np.ndarray] = None
  if isinstance(strategy.inflation_rate, str):
    if strategy.inflation_rate not in local_monthly_asset_prices:
      raise ValueError(
          f"CPI path '{strategy.inflation_rate}' not found in monthly_asset_prices."
      )
    cpi_multiplier_path = local_monthly_asset_prices[strategy.inflation_rate]

  # 初期資産の正規化と無リスク資産のセットアップ
  normalized_ratio: Dict[str, float] = {}
  zero_risk_assets: List[ZeroRiskAsset] = []
  for key, ratio in strategy.initial_asset_ratio.items():
    if isinstance(key, ZeroRiskAsset):
      asset_name = key.name
      normalized_ratio[asset_name] = float(ratio)
      zero_risk_assets.append(key)
      if asset_name not in local_monthly_asset_prices:
        local_monthly_asset_prices[asset_name] = np.ones(
            (n_sim, total_months + 1))
    else:
      normalized_ratio[cast(str, key)] = float(ratio)

  # 初期状態
  total_capital = strategy.initial_money + strategy.initial_loan
  total_ratio = sum(normalized_ratio.values())

  cash = np.full(n_sim, total_capital * (1.0 - total_ratio), dtype=np.float64)
  units: Dict[str, np.ndarray] = {
      name: np.full(n_sim, total_capital * ratio, dtype=np.float64)
      for name, ratio in normalized_ratio.items()
  }
  average_cost: Dict[str, np.ndarray] = {
      name: np.ones(n_sim, dtype=np.float64) for name in normalized_ratio
  }

  yearly_capital_gains = np.zeros(n_sim, dtype=np.float64)
  tax_to_pay = np.zeros(n_sim, dtype=np.float64)
  bankrupt = np.zeros(n_sim, dtype=bool)
  sustained_months = np.full(n_sim, total_months, dtype=np.int32)
  net_values = np.zeros(n_sim, dtype=np.float64)

  # DynamicSpending
  annual_spending = np.zeros(n_sim, dtype=np.float64)
  if isinstance(strategy.annual_cost, DynamicSpending):
    annual_spending.fill(total_capital * strategy.annual_cost.target_ratio)
  prev_annual_spending = np.zeros(n_sim, dtype=np.float64)

  # 追加キャッシュフローの倍率（ソース名ごとに保持）
  extra_cf_multipliers: Dict[str, np.ndarray] = {
      name: np.ones(n_sim, dtype=np.float64)
      for name, fn in strategy.extra_cashflow_sources.items()
      if fn is not None
  }
  has_dynamic_cf = len(extra_cf_multipliers) > 0

  # 年次支出の記録用
  annual_spends_record: Optional[np.ndarray] = None
  if strategy.record_annual_spend:
    n_years = (total_months + 11) // 12
    annual_spends_record = np.zeros((n_sim, n_years), dtype=np.float64)

  # === 追加キャッシュフローの事前計算 ===
  # 静的なキャッシュフロー（倍率関数が None のもの）は事前に合計しておく
  total_static_cf: Optional[np.ndarray] = None
  # 動的なキャッシュフロー（倍率関数があるもの）は個別で保持
  dynamic_cfs: Dict[str, np.ndarray] = {}

  if strategy.extra_cashflow_sources and monthly_cashflows:
    for source, fn in strategy.extra_cashflow_sources.items():
      if source not in monthly_cashflows:
        raise ValueError(f"Cashflow source '{source}' not found in monthly_cashflows.")
      cf = monthly_cashflows[source]
      if cf.shape != (total_months,) and cf.shape != (n_sim, total_months):
        raise ValueError(f"Cashflow source '{source}' has invalid shape {cf.shape}. Expected ({total_months},) or ({n_sim}, {total_months}).")

      if fn is None:
        if total_static_cf is None:
          if cf.ndim == 1:
            total_static_cf = np.broadcast_to(cf, (n_sim, total_months)).copy()
          else:
            total_static_cf = cf.copy()
        else:
          total_static_cf += cf
      else:
        if cf.ndim == 1:
          dynamic_cfs[source] = np.broadcast_to(cf, (n_sim, total_months))
        else:
          dynamic_cfs[source] = cf

  # 月次ループ
  for m in range(total_months):
    active_paths = ~bankrupt
    if not np.any(active_paths):
      break

    # 1. 無リスク資産の利回り
    for zr in zero_risk_assets:
      yield_payment = units[zr.name] * (zr.yield_rate / 12.0) * (
          1.0 - strategy.tax_rate)
      cash[active_paths] += yield_payment[active_paths]

    # 2. 支出額の決定
    # 純資産の計算（動的支出または条件付きキャッシュフローがある場合のみ、年1回実行）
    if m % 12 == 0:
      if isinstance(strategy.annual_cost, DynamicSpending) or has_dynamic_cf:
        current_net_worth = cash.copy()
        for name, u in units.items():
          current_net_worth[active_paths] += u[active_paths] * local_monthly_asset_prices[name][active_paths, m]
        current_net_worth[active_paths] -= strategy.initial_loan

        # 動的支出の更新 (m > 0 の時のみ前年比を考慮)
        if isinstance(strategy.annual_cost, DynamicSpending):
          if m > 0:
            prev_annual_spending[active_paths] = annual_spending[active_paths]
            target_spending = np.maximum(0.0, current_net_worth * strategy.annual_cost.target_ratio)
            ceiling = prev_annual_spending * (1.0 + strategy.annual_cost.upper_limit)
            floor = prev_annual_spending * (1.0 + strategy.annual_cost.lower_limit)
            annual_spending[active_paths] = np.clip(target_spending[active_paths], floor[active_paths], ceiling[active_paths])
          else:
            # m=0 の時は初期値
            annual_spending.fill(total_capital * strategy.annual_cost.target_ratio)

        # 追加キャッシュフロー倍率の更新
        for source, fn in strategy.extra_cashflow_sources.items():
          if fn is not None:
            extra_cf_multipliers[source][active_paths] = fn(m, current_net_worth[active_paths])

    cpi_multiplier: Union[float, np.ndarray] = 1.0
    if strategy.inflation_rate is None:
      cpi_multiplier = 1.0
    elif isinstance(strategy.inflation_rate, float):
      cpi_multiplier = (1.0 + strategy.inflation_rate)**(m / 12.0)
    else:
      cpi_multiplier = cpi_multiplier_path[:, m]  # type: ignore

    if isinstance(strategy.annual_cost, DynamicSpending):
      cost_m = annual_spending / 12.0
      if m % 12 == 0 and annual_spends_record is not None:
        annual_spends_record[:, m // 12] = annual_spending
        annual_spends_record[bankrupt, m // 12] = 0.0
    elif isinstance(strategy.annual_cost, list):
      full_cost = (strategy.annual_cost[m // 12]) * cpi_multiplier
      cost_m = np.full(n_sim, full_cost / 12.0, dtype=np.float64)
      if m % 12 == 0 and annual_spends_record is not None:
        annual_spends_record[:, m // 12] = full_cost
        annual_spends_record[bankrupt, m // 12] = 0.0
    else:
      full_cost = strategy.annual_cost * cpi_multiplier
      cost_m = np.full(n_sim, full_cost / 12.0, dtype=np.float64)
      if m % 12 == 0 and annual_spends_record is not None:
        annual_spends_record[:, m // 12] = full_cost
        annual_spends_record[bankrupt, m // 12] = 0.0

    # 3. 現金需要 (支出 + 利息 + 前年分の税金)
    interest = strategy.initial_loan * (strategy.yearly_loan_interest / 12.0)
    required_cash = np.zeros(n_sim, dtype=np.float64)
    required_cash[active_paths] = cost_m[active_paths]
    required_cash[active_paths] += interest

    # 前年分の税金の支払い
    if m > 0 and m % 12 == 0:
      required_cash[active_paths] += tax_to_pay[active_paths]
      tax_to_pay.fill(0.0)

    # 追加のキャッシュフロー（年金、死亡時収入など）を反映
    # 正の値は収入（現金需要を減らす）、負の値は支出（現金需要を増やす）
    if total_static_cf is not None:
      required_cash[active_paths] -= total_static_cf[active_paths, m]
    for source, cf in dynamic_cfs.items():
      required_cash[active_paths] -= cf[active_paths, m] * extra_cf_multipliers[source][active_paths]

    cash[active_paths] -= required_cash[active_paths]

    # 4. 資産売却 (現金不足時)
    shortage_paths = active_paths & (cash < 0)
    if np.any(shortage_paths):
      for asset_name in strategy.selling_priority:
        still_short = shortage_paths & (cash < 0)
        if not np.any(still_short):
          break

        price_m_plus_1 = local_monthly_asset_prices[asset_name][still_short,
                                                                m + 1]
        asset_val = units[asset_name][still_short] * price_m_plus_1
        sell_amount = np.minimum(asset_val, -cash[still_short])

        # 売却口数
        valid_price = price_m_plus_1 > 0
        units_to_sell = np.zeros_like(sell_amount)
        units_to_sell[valid_price] = sell_amount[valid_price] / price_m_plus_1[
            valid_price]

        # 利益確定
        gain = sell_amount - units_to_sell * average_cost[asset_name][
            still_short]
        yearly_capital_gains[still_short] += gain

        units[asset_name][still_short] -= units_to_sell
        cash[still_short] += sell_amount

    # 5. リバランス
    if strategy.rebalance_interval > 0 and (
        m + 1) % strategy.rebalance_interval == 0:
      reb_paths = active_paths
      if np.any(reb_paths):
        total_net = cash[reb_paths].copy()
        current_vals = {}
        for name in units:
          cur_p = local_monthly_asset_prices[name][reb_paths, m + 1]
          val = units[name][reb_paths] * cur_p
          current_vals[name] = val
          total_net += val

        # 目標割合
        if strategy.dynamic_rebalance_fn:
          rem_years = (total_months - (m + 1)) / 12.0
          cur_ann_spend = cost_m[reb_paths] * 12.0
          target_ratios = strategy.dynamic_rebalance_fn(total_net,
                                                        cur_ann_spend,
                                                        rem_years)
        else:
          target_ratios = {
              k: np.full(np.sum(reb_paths), v)
              for k, v in normalized_ratio.items()
          }

        # 売却
        for name in normalized_ratio:
          diff = current_vals[name] - total_net * target_ratios[name]
          sell_mask = diff > 1e-8
          if np.any(sell_mask):
            idx = np.where(reb_paths)[0][sell_mask]
            amt = diff[sell_mask]
            p = local_monthly_asset_prices[name][idx, m + 1]
            u_sell = np.zeros_like(amt)
            u_sell[p > 0] = amt[p > 0] / p[p > 0]
            yearly_capital_gains[idx] += amt - u_sell * average_cost[name][idx]
            units[name][idx] -= u_sell
            cash[idx] += amt

        # 購入
        for name in normalized_ratio:
          # 最新価格での評価額再計算は省略し、目標値との差分で購入
          val_after_sell = units[name][reb_paths] * local_monthly_asset_prices[
              name][reb_paths, m + 1]
          diff = total_net * target_ratios[name] - val_after_sell
          buy_mask = diff > 1e-8
          if np.any(buy_mask):
            idx = np.where(reb_paths)[0][buy_mask]
            amt = np.minimum(diff[buy_mask], cash[idx])
            p = local_monthly_asset_prices[name][idx, m + 1]
            u_buy = np.zeros_like(amt)
            u_buy[p > 0] = amt[p > 0] / p[p > 0]

            # 平均単価更新
            new_u = units[name][idx] + u_buy
            upd = new_u > 0
            if np.any(upd):
              i_upd = idx[upd]
              average_cost[name][i_upd] = (
                  units[name][i_upd] * average_cost[name][i_upd] +
                  u_buy[upd] * p[upd]) / new_u[upd]

            units[name][idx] += u_buy
            cash[idx] -= amt

    # 6. 破産判定と年末処理
    total_val = cash.copy()
    for name, u in units.items():
      total_val[active_paths] += u[active_paths] * local_monthly_asset_prices[
          name][active_paths, m + 1]

    new_bankrupt = active_paths & (total_val < strategy.initial_loan)
    bankrupt[new_bankrupt] = True
    sustained_months[new_bankrupt] = m

    if m % 12 == 11:
      tax_to_pay = np.maximum(yearly_capital_gains, 0.0) * strategy.tax_rate
      yearly_capital_gains.fill(0.0)

    if m == total_months - 1:
      survivors = ~bankrupt
      net_values[survivors] = total_val[survivors] - strategy.initial_loan

  return SimulationResult(net_values=net_values,
                          sustained_months=sustained_months,
                          annual_spends=annual_spends_record)
