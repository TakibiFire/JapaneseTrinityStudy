"""
日本版トリニティ・スタディのシミュレーションに必要なデータクラスと関数群（新エンジン版）。

アセット生成（asset_generator.py）で生成された月次価格推移に基づき、
ポート欧の運用、リバランス、取り崩し、税金計算などを実行する。
"""

import dataclasses
from typing import Callable, Dict, List, Optional, Union, cast

import numpy as np

from src.lib.cashflow_generator import (CashflowRule, CashflowType,
                                        ExtraCashflowMultiplierFn)

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
  cashflow_rules: List[CashflowRule] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    # キャッシュフロールールのソース名が重複していないか確認
    rule_names = [rule.source_name for rule in self.cashflow_rules]
    if len(rule_names) != len(set(rule_names)):
      raise ValueError("Duplicate source_name found in cashflow_rules.")

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
  debug_results: Optional[Dict[int, List[str]]] = None


# ---------------------------------------------------------------------------
# 2. シミュレーションコア
# ---------------------------------------------------------------------------


def simulate_strategy(
    strategy: Strategy,
    monthly_asset_prices: Dict[str, np.ndarray],
    monthly_cashflows: Optional[Dict[str, np.ndarray]] = None,
    fallback_n_sim: int = 1000,
    fallback_total_months: int = 600,
    debug_indices: Optional[List[int]] = None,
    exp_regard_interest_tax_as_regular: bool = False) -> SimulationResult:
  """
  指定された戦略に従い、資産推移をシミュレーションする。
  
  Args:
    strategy: 戦略設定
    monthly_asset_prices: アセット生成エンジンが作成した月次価格の辞書
    monthly_cashflows: cashflow_generatorが生成した追加の月次キャッシュフローの辞書
    fallback_n_sim: 価格推移辞書が空だった場合に使用するパス数
    fallback_total_months: 価格推移辞書が空だった場合に使用する月数
    debug_indices: デバッグ対象のパスのインデックスリスト
    exp_regard_interest_tax_as_regular: 利息や税金を定常的な支出として扱う実験的フラグ
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

  # 定常支出の管理
  target_annual_spend = np.zeros(n_sim, dtype=np.float64)
  prev_net_reg_spend_y = np.zeros(n_sim, dtype=np.float64)
  annual_net_reg_spend_tracker = np.zeros(n_sim, dtype=np.float64)

  if isinstance(strategy.annual_cost, DynamicSpending):
    init_val = total_capital * strategy.annual_cost.target_ratio
    target_annual_spend.fill(init_val)
    prev_net_reg_spend_y.fill(init_val)
  elif isinstance(strategy.annual_cost, list):
    init_val = strategy.annual_cost[0]
    target_annual_spend.fill(init_val)
    prev_net_reg_spend_y.fill(init_val)
  else:
    init_val = cast(float, strategy.annual_cost)
    target_annual_spend.fill(init_val)
    prev_net_reg_spend_y.fill(init_val)
  
  # 追加キャッシュフローの倍率（ソース名ごとに保持）
  extra_cf_multipliers: Dict[str, np.ndarray] = {
      rule.source_name: np.ones(n_sim, dtype=np.float64)
      for rule in strategy.cashflow_rules
      if rule.multiplier_fn is not None
  }
  has_dynamic_cf = len(extra_cf_multipliers) > 0

  # 年次支出の記録用
  annual_spends_record: Optional[np.ndarray] = None
  if strategy.record_annual_spend:
    n_years = (total_months + 11) // 12
    annual_spends_record = np.zeros((n_sim, n_years), dtype=np.float64)

  # === 追加キャッシュフローの準備 ===
  # (rule, processed_cf) のリストを保持
  prepared_cashflows: List[tuple[CashflowRule, np.ndarray]] = []

  debug_results: Optional[Dict[int, List[str]]] = None
  if debug_indices is not None:
    debug_results = {idx: [] for idx in debug_indices}

  if strategy.cashflow_rules and monthly_cashflows:
    for rule in strategy.cashflow_rules:
      source = rule.source_name
      if source not in monthly_cashflows:
        raise ValueError(f"Cashflow source '{source}' not found in monthly_cashflows.")
      cf = monthly_cashflows[source]
      if cf.shape != (total_months,) and cf.shape != (n_sim, total_months):
        raise ValueError(
            f"Cashflow source '{source}' has invalid shape {cf.shape}. Expected ({total_months},) or ({n_sim}, {total_months})."
        )

      if cf.ndim == 1:
        processed_cf = np.broadcast_to(cf, (n_sim, total_months))
      else:
        processed_cf = cf
      prepared_cashflows.append((rule, processed_cf))

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

        # 年間支出と基準支出 (prev_net_reg_spend_y) の更新
        if m > 0:
          prev_net_reg_spend_y[active_paths] = annual_net_reg_spend_tracker[active_paths]
          annual_net_reg_spend_tracker.fill(0.0)

        if isinstance(strategy.annual_cost, DynamicSpending):
          if m > 0:
            # ダイナミックスペンディングの目標額（名目）
            target_spending_nominal = np.maximum(0.0, current_net_worth * strategy.annual_cost.target_ratio)
            # 前年の正味支出に基づく上下限
            ceiling = prev_net_reg_spend_y * (1.0 + strategy.annual_cost.upper_limit)
            floor = prev_net_reg_spend_y * (1.0 + strategy.annual_cost.lower_limit)
            target_annual_spend[active_paths] = np.clip(target_spending_nominal[active_paths], floor[active_paths], ceiling[active_paths])
          else:
            # m=0 の時は初期値
            target_annual_spend.fill(total_capital * strategy.annual_cost.target_ratio)
            prev_net_reg_spend_y.fill(total_capital * strategy.annual_cost.target_ratio)

        # 追加キャッシュフロー倍率の更新
        for rule in strategy.cashflow_rules:
          if rule.multiplier_fn is not None:
            extra_cf_multipliers[rule.source_name][active_paths] = rule.multiplier_fn(
                m, current_net_worth[active_paths],
                prev_net_reg_spend_y[active_paths])

    # インフレ調整
    cpi_multiplier: Union[float, np.ndarray] = 1.0
    if strategy.inflation_rate is None:
      cpi_multiplier = 1.0
    elif isinstance(strategy.inflation_rate, float):
      cpi_multiplier = (1.0 + strategy.inflation_rate)**(m / 12.0)
    else:
      cpi_multiplier = cpi_multiplier_path[:, m]  # type: ignore

    # 基本支出の決定 (annual_base_spend_nominal: 万円/年)
    if isinstance(strategy.annual_cost, DynamicSpending):
      annual_base_spend_nominal = target_annual_spend
      base_spend_m = annual_base_spend_nominal / 12.0
    elif isinstance(strategy.annual_cost, list):
      annual_base_spend_nominal = np.full(n_sim, (strategy.annual_cost[m // 12]) * cpi_multiplier)
      base_spend_m = annual_base_spend_nominal / 12.0
    else:
      annual_base_spend_nominal = np.full(n_sim, strategy.annual_cost * cpi_multiplier)
      base_spend_m = annual_base_spend_nominal / 12.0

    # 3. 収支の計算
    # 定常的な支出（reg_spend_m）と収入（reg_income_m）を別々に集計
    reg_spend_m = base_spend_m.copy()
    reg_income_m = np.zeros(n_sim, dtype=np.float64)
    iso_spend_m = np.zeros(n_sim, dtype=np.float64)
    iso_income_m = np.zeros(n_sim, dtype=np.float64)

    # 追加キャッシュフローの反映
    # 収入は正、支出は負の値として与えられる
    for rule, cf_path in prepared_cashflows:
      source = rule.source_name
      multiplier = extra_cf_multipliers.get(source, 1.0)
      impact = cf_path[:, m] * multiplier
      
      if rule.cashflow_type == CashflowType.INCLUDE_IN_ANNUAL_SPEND:
        # 定常収支
        reg_income_m[impact >= 0] += impact[impact >= 0]
        reg_spend_m[impact < 0] += np.abs(impact[impact < 0])
      else:
        # 非定常収支 (Isolated)
        iso_income_m[impact >= 0] += impact[impact >= 0]
        iso_spend_m[impact < 0] += np.abs(impact[impact < 0])

    # 金融コスト（利息・税金）
    interest_cost_m = strategy.initial_loan * (strategy.yearly_loan_interest / 12.0)
    tax_cost_m = np.zeros(n_sim, dtype=np.float64)
    if m > 0 and m % 12 == 0:
      tax_cost_m = tax_to_pay.copy()
      tax_to_pay.fill(0.0)

    # 定常的な正味の支出 (DRやDynamicSpendingの基準となる)
    # 収入は支出を減らす方向に働く
    net_reg_spend_m = reg_spend_m - reg_income_m
    if exp_regard_interest_tax_as_regular:
      net_reg_spend_m += interest_cost_m + tax_cost_m
      
    annual_net_reg_spend_tracker[active_paths] += net_reg_spend_m[active_paths]

    # ポートフォリオからの月間の総引き出し額 (Withdrawal)
    total_withdrawal_m = (reg_spend_m - reg_income_m) + (iso_spend_m - iso_income_m) + interest_cost_m + tax_cost_m
    cash[active_paths] -= total_withdrawal_m[active_paths]

    # 4. 資産売却 (現金不足時)
    shortage_paths = active_paths & (cash < 0)
    if np.any(shortage_paths):
      # --- DEBUG ---
      if debug_indices is not None and debug_results is not None:
        for idx in debug_indices:
          if idx < len(cash) and active_paths[idx] and shortage_paths[idx]:
            debug_results[idx].append(f"[Debug Path {idx}] Month {m}: Cash shortage before sell-off: {cash[idx]:.4f}")
      # -------------
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
        # リバランス時の純資産合計
        total_net = cash[reb_paths].copy()
        for name in units:
          # 各アセットの評価額を加算 (m+1月、つまり来月頭の価格を使用)
          asset_price_at_rebalance = local_monthly_asset_prices[name][reb_paths, m + 1]
          total_net += units[name][reb_paths] * asset_price_at_rebalance

        # 目標割合
        if strategy.dynamic_rebalance_fn:
          # ---
          # 寿命ギリギリで100%無リスク資産に移行したパスが、正確に最終月（Month 599）で
          # 枯渇する（Bankruptクラスターが発生する）現象を防ぐためのバッファ。
          # rebalance_interval=12 の場合、rem_years は常に整数（49.0, ..., 1.0, 0.0）となる。
          # これが「無リスク資産のみでちょうど N 年生きられる」という境界条件（n_ruin == N）
          # と完全に一致してしまい、多数のパスが同時に100%無リスクにロックされてしまう。
          # そこで、シミュレーション期間よりも少し長生きする必要がある（+0.25年 = 3ヶ月）
          # と見せかけることで、この人為的な境界条件の完全一致を回避する。
          # ---
          rem_years = (total_months - (m + 1)) / 12.0 + 0.25
          # DRには正味の年間支出を渡す（負の値にならないよう0でクリップ）
          # ここで '-' を使うのは、収入（正の値）がポートフォリオからの引き出しを減らすため。
          cur_ann_spend = np.maximum(0.0, net_reg_spend_m[reb_paths]) * 12.0
          target_ratios = strategy.dynamic_rebalance_fn(total_net,
                                                        cur_ann_spend,
                                                        rem_years)
          # --- DEBUG ---
          if debug_indices is not None and debug_results is not None:
            for idx in debug_indices:
              if idx < len(reb_paths) and reb_paths[idx]:
                mask_idx = np.sum(reb_paths[:idx])
                debug_results[idx].append(f"[Debug Path {idx}] Month {m} Rebalance: rem_years={rem_years:.4f}, total_net={total_net[mask_idx]:.2f}, target_ratios={ {k: (v if isinstance(v, float) else v[mask_idx]) for k, v in target_ratios.items()} }")
          # -------------
        else:
          target_ratios = {
              k: np.full(np.sum(reb_paths), v)
              for k, v in normalized_ratio.items()
          }

        # 売却
        for name in normalized_ratio:
          # 目標割合に合わせるための売却額の計算
          price_for_sell = local_monthly_asset_prices[name][reb_paths, m + 1]
          current_asset_val = units[name][reb_paths] * price_for_sell
          diff = current_asset_val - total_net * target_ratios[name]
          sell_mask = diff > 1e-8
          if np.any(sell_mask):
            idx = np.where(reb_paths)[0][sell_mask]
            amt = diff[sell_mask]
            p_subset = local_monthly_asset_prices[name][idx, m + 1]
            u_sell = np.zeros_like(amt)
            u_sell[p_subset > 0] = amt[p_subset > 0] / p_subset[p_subset > 0]
            yearly_capital_gains[idx] += amt - u_sell * average_cost[name][idx]
            units[name][idx] -= u_sell
            cash[idx] += amt

        # 購入
        for name in normalized_ratio:
          # 最新価格での評価額再計算
          price_for_buy = local_monthly_asset_prices[name][reb_paths, m + 1]
          val_after_sell = units[name][reb_paths] * price_for_buy
          diff = total_net * target_ratios[name] - val_after_sell
          buy_mask = diff > 1e-8
          if np.any(buy_mask):
            idx = np.where(reb_paths)[0][buy_mask]
            amt = np.minimum(diff[buy_mask], cash[idx])
            p_subset = local_monthly_asset_prices[name][idx, m + 1]
            u_buy = np.zeros_like(amt)
            u_buy[p_subset > 0] = amt[p_subset > 0] / p_subset[p_subset > 0]

            # 平均単価更新
            new_u = units[name][idx] + u_buy
            upd = new_u > 0
            if np.any(upd):
              i_upd = idx[upd]
              average_cost[name][i_upd] = (
                  units[name][i_upd] * average_cost[name][i_upd] +
                  u_buy[upd] * p_subset[upd]) / new_u[upd]

            units[name][idx] += u_buy
            cash[idx] -= amt

    # 6. 破産判定と年末記録
    total_val = cash.copy()
    for name, u in units.items():
      total_val[active_paths] += u[active_paths] * local_monthly_asset_prices[
          name][active_paths, m + 1]

    new_bankrupt = active_paths & (total_val < strategy.initial_loan)
    # --- DEBUG ---
    if debug_indices is not None and debug_results is not None:
      for idx in debug_indices:
        if idx < len(active_paths) and active_paths[idx]:
          debug_results[idx].append(f"[Debug Path {idx}] Month {m} Step 6: cash={cash[idx]:.2f}, total_val={total_val[idx]:.2f}, new_bankrupt={new_bankrupt[idx]}")
    # -------------
    bankrupt[new_bankrupt] = True
    sustained_months[new_bankrupt] = m

    if m % 12 == 11:
      tax_to_pay = np.maximum(yearly_capital_gains, 0.0) * strategy.tax_rate
      yearly_capital_gains.fill(0.0)
      
      # 年次支出の記録
      if annual_spends_record is not None:
        # 去年の正味定常支出を記録 (trackerには12ヶ月分溜まっているはず)
        annual_spends_record[:, m // 12] = annual_net_reg_spend_tracker
        annual_spends_record[bankrupt, m // 12] = 0.0

    if m == total_months - 1:
      survivors = ~bankrupt
      net_values[survivors] = total_val[survivors] - strategy.initial_loan

  return SimulationResult(net_values=net_values,
                          sustained_months=sustained_months,
                          annual_spends=annual_spends_record,
                          debug_results=debug_results)
