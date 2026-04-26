"""
日本版トリニティ・スタディのシミュレーションに必要なデータクラスと関数群（新エンジン版）。

アセット生成（asset_generator.py）で生成された月次価格推移に基づき、
ポートフォリオの運用、リバランス、取り崩し、税金計算などを実行する。
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

  支出額を資産残高の一定割合（target_ratio）に保とうとするが、
  前年の支出実績（prev_actual_amount）に対して一定の範囲（lower_limit, upper_limit）
  に収まるように調整する。
  """
  initial_annual_spend: float  # 初年度の目標基本支出額 (万円/年)
  target_ratio: float          # 目標支出率 (純資産に対する割合)
  upper_limit: float           # 前年比の最大上昇率 (0.03 = +3%)
  lower_limit: float           # 前年比の最大下降率 (-0.005 = -0.5%)

  def evaluate(self, m: int, active_paths: np.ndarray,
               current_net_worth: np.ndarray, tax_cost_m: np.ndarray,
               prev_actual_amount: np.ndarray, other_net_m: np.ndarray,
               precomputed_cf_m: np.ndarray,
               precomputed_cf_prev_m: np.ndarray) -> np.ndarray:
    """
    Vanguard のダイナミックスペンディング規則に基づき、新しい名目基本支出額を算出する。

    Args:
      m: シミュレーション開始からの経過月数
      active_paths: 現在生存しているパスのマスク
      current_net_worth: 現在の純資産
      tax_cost_m: 今月の税金
      prev_actual_amount: 前年の実績支出額
      other_net_m: 他のキャッシュフローによる正味収支
      precomputed_cf_m: 事前計算された今月のキャッシュフロー額
      precomputed_cf_prev_m: 事前計算された12ヶ月前のキャッシュフロー額
    """
    n_sim = len(active_paths)
    new_base_nom = np.zeros(n_sim, dtype=np.float64)

    if m == 0:
      new_base_nom.fill(self.initial_annual_spend)
      return new_base_nom

    # 有効なパスのみ計算対象とする
    # Effective NW = current_net_worth - tax_cost_m
    nw_active = current_net_worth[active_paths] - tax_cost_m[active_paths]
    prev_actual_active = prev_actual_amount[active_paths]
    other_net_active = other_net_m[active_paths]

    # precomputed_cf_m / precomputed_cf_prev_m が CPI比率に相当する
    cpi_ratio = np.ones(len(nw_active))
    mask_nonzero = precomputed_cf_prev_m[active_paths] > 0
    cpi_ratio[mask_nonzero] = precomputed_cf_m[active_paths][mask_nonzero] / precomputed_cf_prev_m[active_paths][mask_nonzero]

    # 1. 目標支出額 (Target Withdrawal)
    # ポートフォリオ残高に対する一定割合を目指す。
    target_withdrawal = np.maximum(0.0, nw_active * self.target_ratio)
    # 目標支出額から他のキャッシュフロー（年金等）を引いたものが、基本支出の目標となる。
    # other_net_m は 支出 - 収入 なので、目標引き出し額から other_net_m*12 を足せば
    # 基本支出で賄うべき額が出る。
    y_target = np.maximum(0.0, target_withdrawal + (other_net_active * 12.0))

    # 2. 前年の支出をインフレ調整 (Inflation-adjusted previous spend)
    y_prev_adjusted = prev_actual_active * cpi_ratio

    # 3. 上限と下限の算出 (Calculate Limits)
    # 上限・下限はインフレ調整後の前年支出額に対して適用される。
    y_max = y_prev_adjusted * (1.0 + self.upper_limit)
    y_min = y_prev_adjusted * (1.0 + self.lower_limit)

    # 4. 範囲内に制限 (Apply Limits)
    res_active = np.clip(y_target, y_min, y_max)

    new_base_nom[active_paths] = res_active
    return new_base_nom

  def calculate_nominal_spend(self, m: int, net_worth: np.ndarray,
                              prev_base_spend_y: np.ndarray, cpi_m: np.ndarray,
                              cpi_m_minus_12: np.ndarray,
                              other_net_m: np.ndarray,
                              active_paths: np.ndarray) -> np.ndarray:
    """
    (Deprecated) 旧エンジン用の計算メソッド。
    """
    return self.evaluate(
        m=m, active_paths=active_paths, current_net_worth=net_worth,
        tax_cost_m=np.zeros_like(net_worth), prev_actual_amount=prev_base_spend_y,
        other_net_m=other_net_m, precomputed_cf_m=cpi_m, precomputed_cf_prev_m=cpi_m_minus_12
    )


# ダイナミックリバランス用のコールバック関数
DynamicRebalanceFn = Callable[[np.ndarray, np.ndarray, float, np.ndarray],
                               Dict[str, Union[float, np.ndarray]]]


@dataclasses.dataclass
class Strategy:
  """
  運用戦略の定義。
  """
  name: str
  initial_money: Union[float, np.ndarray]  # 万円
  initial_loan: float  # 万円
  yearly_loan_interest: float
  initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float]
  selling_priority: List[str]
  tax_rate: float = 0.20315
  rebalance_interval: int = 0
  dynamic_rebalance_fn: Optional[DynamicRebalanceFn] = None
  record_annual_spend: bool = False
  # シミュレーション開始時点（1年目の年初）において、「前年の支出実績」として扱う金額。
  # multiplier_fn や DynamicSpending ハンドラが 1年目の支出や行動を決定する際の基準となる。
  initial_prev_net_reg_spend: float = 0.0  # 前年の正味定常支出（支出 - 収入）
  initial_prev_gross_reg_spend: float = 0.0 # 前年の総定常支出（収入控除前、ライフスタイルコスト）
  cashflow_rules: List[CashflowRule] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    # 資産配分の合計が 1.0 であるか確認
    if not np.isclose(sum(self.initial_asset_ratio.values()), 1.0):
      raise ValueError("Asset allocation must sum to 1.0")

    # キャッシュフロールールのソース名が重複していないか確認
    rule_names = [rule.source_name for rule in self.cashflow_rules]
    if len(rule_names) != len(set(rule_names)):
      raise ValueError("Duplicate source_name found in cashflow_rules.")

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
  post_tax_net_values: Optional[np.ndarray] = None  # shape: (n_sim,)
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
    exp_regard_interest_tax_as_regular: bool = False,
    calculate_post_tax: bool = False) -> SimulationResult:
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
    calculate_post_tax: 全ての資産が課税対象であると仮定した保守的な純資産額（Post-tax Net Value）を計算するフラグ
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
  if isinstance(total_capital, np.ndarray):
    if total_capital.shape != (n_sim,):
      raise ValueError(
          f"initial_money array shape {total_capital.shape} does not match n_sim {n_sim}"
      )

  total_ratio = sum(normalized_ratio.values())

  # 資産ごとの保有口数 (units) と平均取得単価 (average_cost) の初期化。
  # 価格は 1.0 からスタートするとは限らないので、開始時の価格で除算して正しい保有口数を
  # 算出する。
  units: Dict[str, np.ndarray] = {}
  if isinstance(total_capital, np.ndarray):
    cash = (total_capital * (1.0 - total_ratio)).astype(np.float64)
    for name, ratio in normalized_ratio.items():
      units[name] = ((total_capital * ratio).astype(np.float64) /
                     local_monthly_asset_prices[name][:, 0])
  else:
    cash = np.full(n_sim, total_capital * (1.0 - total_ratio), dtype=np.float64)
    for name, ratio in normalized_ratio.items():
      units[name] = (np.full(n_sim, total_capital * ratio, dtype=np.float64) /
                     local_monthly_asset_prices[name][:, 0])

  # 平均取得単価も開始時の価格で初期化する。
  average_cost: Dict[str, np.ndarray] = {
      name: local_monthly_asset_prices[name][:, 0].copy()
      for name in normalized_ratio
  }

  yearly_capital_gains = np.zeros(n_sim, dtype=np.float64)
  tax_to_pay = np.zeros(n_sim, dtype=np.float64)
  bankrupt = np.zeros(n_sim, dtype=bool)
  sustained_months = np.full(n_sim, total_months, dtype=np.int32)
  net_values = np.zeros(n_sim, dtype=np.float64)

  # 定常支出の管理
  # prev_net_reg_spend_y: 前年の正味の年間支出額（実績）
  # prev_gross_reg_spend_y: 前年の総年間支出額（実績、収入控除前）
  # annual_net_reg_spend_tracker: 今年の正味年間支出額の累計
  # annual_gross_reg_spend_tracker: 今年の総年間支出額の累計
  prev_net_reg_spend_y = np.full(n_sim, strategy.initial_prev_net_reg_spend, dtype=np.float64)
  prev_gross_reg_spend_y = np.full(n_sim, strategy.initial_prev_gross_reg_spend, dtype=np.float64)
  annual_net_reg_spend_tracker = np.zeros(n_sim, dtype=np.float64)
  annual_gross_reg_spend_tracker = np.zeros(n_sim, dtype=np.float64)

  # ルールごとの支出管理
  # actual_annual_spend_tracker: 今年の各ルールの支出実績累計
  # prev_annual_spend_y: 前年の各ルールの支出実績
  # dynamic_annual_amount: 今年の各ルールの動的決定支出額 (年初に決定)
  actual_annual_spend_tracker: Dict[str, np.ndarray] = {
      rule.source_name: np.zeros(n_sim, dtype=np.float64)
      for rule in strategy.cashflow_rules
  }
  prev_annual_spend_y: Dict[str, np.ndarray] = {
      rule.source_name: np.zeros(n_sim, dtype=np.float64)
      for rule in strategy.cashflow_rules
  }
  dynamic_annual_amount: Dict[str, np.ndarray] = {
      rule.source_name: np.zeros(n_sim, dtype=np.float64)
      for rule in strategy.cashflow_rules
  }

  # 追加キャッシュフローの倍率（ソース名ごとに保持）
  extra_cf_multipliers: Dict[str, np.ndarray] = {
      rule.source_name: np.ones(n_sim, dtype=np.float64)
      for rule in strategy.cashflow_rules
      if rule.multiplier_fn is not None
  }

  # 年次支出の記録用
  annual_spends_record: Optional[np.ndarray] = None
  if strategy.record_annual_spend:
    n_years = (total_months + 11) // 12
    annual_spends_record = np.zeros((n_sim, n_years), dtype=np.float64)

  # 年次税金の支払額
  tax_cost_m = np.zeros(n_sim, dtype=np.float64)

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
        raise ValueError(
            f"Cashflow source '{source}' not found in monthly_cashflows.")
      cf = monthly_cashflows[source]
      if cf.shape != (total_months,) and cf.shape != (n_sim, total_months):
        raise ValueError(
            f"Cashflow source '{source}' has invalid shape {cf.shape}."
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
      yield_payment = units[zr.name] * (zr.yield_rate /
                                        12.0) * (1.0 - strategy.tax_rate)
      cash[active_paths] += yield_payment[active_paths]

    # 2. 税金の支払額の決定 (年初のみ). SpendAwareDynamicSpending は Net Worth
    # を求めるため、tax_cost_m は先に決定する必要がある。
    if m > 0 and m % 12 == 0:
      tax_cost_m = tax_to_pay.copy()
      tax_to_pay.fill(0.0)
    else:
      tax_cost_m.fill(0.0)

    # 3. キャッシュフローの評価 (年初のみ)
    if m % 12 == 0:
      # 年間支出トラッカーのリセット
      if m > 0:
        prev_net_reg_spend_y[active_paths] = annual_net_reg_spend_tracker[
            active_paths]
        prev_gross_reg_spend_y[active_paths] = annual_gross_reg_spend_tracker[
            active_paths]
        annual_net_reg_spend_tracker.fill(0.0)
        annual_gross_reg_spend_tracker.fill(0.0)
        for source in actual_annual_spend_tracker:
          prev_annual_spend_y[source][active_paths] = actual_annual_spend_tracker[
              source][active_paths]
          actual_annual_spend_tracker[source].fill(0.0)

      # 純資産の計算
      current_net_worth = cash.copy()
      for name, u in units.items():
        current_net_worth[active_paths] += u[
            active_paths] * local_monthly_asset_prices[name][active_paths, m]
      current_net_worth[active_paths] -= strategy.initial_loan

      # 倍率の更新 (支出決定の前に実行)
      for rule in strategy.cashflow_rules:
        if rule.multiplier_fn is not None:
          extra_cf_multipliers[
              rule.source_name][active_paths] = rule.multiplier_fn(
                  m, current_net_worth[active_paths],
                  prev_net_reg_spend_y[active_paths],
                  prev_gross_reg_spend_y[active_paths])

      # 動的ハンドラの評価
      # 各キャッシュフロールールに対して動的な金額（支出額）を決定する
      for rule, cf_path in prepared_cashflows:
        if rule.dynamic_handler:
          # 他の REGULAR キャッシュフローの合計を算出し、ハンドラに渡す。
          # これにより、ハンドラは「年金などで賄いきれない不足分」を把握できる。
          other_net_m_val = np.zeros(n_sim, dtype=np.float64)
          for other_rule, other_cf_path in prepared_cashflows:
            if other_rule.source_name != rule.source_name and other_rule.cashflow_type == CashflowType.REGULAR:
              other_multiplier = extra_cf_multipliers.get(other_rule.source_name, 1.0)
              other_impact = other_cf_path[:, m] * other_multiplier
              other_net_m_val -= other_impact # 収入(正)なら net_m を減らすのでマイナス

          cf_m = np.abs(cf_path[:, m])
          cf_prev = np.abs(cf_path[:, m-12]) if m >= 12 else cf_m

          # 注意: ここで prev_gross_reg_spend_y を渡すと、DynamicSpending や SpendAware が
          # 自らの出力実績以外の現金流出（年金保険料など）も前年実績として取り込んでしまい、
          # さらに other_net_m でも二重カウントされることで、支出が指数関数的に膨張する致命的なバグに繋がります。
          # そのため、各ハンドラには必ず「そのルール自身による前年の出力実績」のみを渡します。
          prev_amt = prev_annual_spend_y[rule.source_name]

          dynamic_annual_amount[rule.source_name][active_paths] = rule.dynamic_handler.evaluate(
              m=m, active_paths=active_paths, current_net_worth=current_net_worth,
              tax_cost_m=tax_cost_m, prev_actual_amount=prev_amt,
              other_net_m=other_net_m_val, precomputed_cf_m=cf_m, precomputed_cf_prev_m=cf_prev
          )[active_paths]

          # --- DEBUG ---
          if debug_indices is not None and debug_results is not None:
            for d_idx in debug_indices:
              if d_idx < len(active_paths) and active_paths[d_idx]:
                debug_results[d_idx].append(
                    f"[Debug {rule.source_name}] Year {m//12}: NW={current_net_worth[d_idx]-tax_cost_m[d_idx]:.2f}, target={dynamic_annual_amount[rule.source_name][d_idx]:.2f}, prev={prev_annual_spend_y[rule.source_name][d_idx]:.2f}"
                )

    # 収支の計算
    reg_spend_m = np.zeros(n_sim, dtype=np.float64)
    reg_income_m = np.zeros(n_sim, dtype=np.float64)
    iso_spend_m = np.zeros(n_sim, dtype=np.float64)
    iso_income_m = np.zeros(n_sim, dtype=np.float64)

    # キャッシュフローの反映
    # impact の正負に注意:
    # - 正の値: 収入 (income)
    # - 負の値: 支出 (spend)
    for rule, cf_path in prepared_cashflows:
      source = rule.source_name
      multiplier = extra_cf_multipliers.get(source, 1.0)
      
      if rule.dynamic_handler:
        # 動的ハンドラの結果を月割りにする
        impact = -(dynamic_annual_amount[source] / 12.0)
      else:
        impact = cf_path[:, m] * multiplier

      if rule.cashflow_type == CashflowType.REGULAR:
        # 定常収支
        reg_income_m[impact >= 0] += impact[impact >= 0]
        reg_spend_m[impact < 0] += np.abs(impact[impact < 0])
      else:
        # 非定常収支 (Isolated)
        iso_income_m[impact >= 0] += impact[impact >= 0]
        iso_spend_m[impact < 0] += np.abs(impact[impact < 0])

      actual_annual_spend_tracker[source][active_paths] += np.abs(impact[active_paths])

    # 金融コスト（利息・税金）
    interest_cost_m = strategy.initial_loan * (strategy.yearly_loan_interest /
                                               12.0)

    # 定常的な正味の支出
    # ポートフォリオが賄うべき金額 (支出 - 収入)。
    # 収入 (reg_income_m) は支出を減らす方向に働くため差し引く。
    net_reg_spend_m = reg_spend_m - reg_income_m
    if exp_regard_interest_tax_as_regular:
      net_reg_spend_m += interest_cost_m + tax_cost_m

    annual_net_reg_spend_tracker[active_paths] += net_reg_spend_m[active_paths]
    annual_gross_reg_spend_tracker[active_paths] += reg_spend_m[active_paths]

    # ポートフォリオからの月間の総引き出し額 (Withdrawal)
    # net_reg_spend_m は (reg_spend_m - reg_income_m) と等価
    total_withdrawal_m = net_reg_spend_m + (
        iso_spend_m - iso_income_m) + interest_cost_m + tax_cost_m
    cash[active_paths] -= total_withdrawal_m[active_paths]

    # 5. 資産売却 (現金不足時)
    shortage_paths = active_paths & (cash < 0)
    if np.any(shortage_paths):
      # --- DEBUG ---
      if debug_indices is not None and debug_results is not None:
        for idx in debug_indices:
          if idx < len(cash) and active_paths[idx] and shortage_paths[idx]:
            debug_results[idx].append(
                f"[Debug Path {idx}] Month {m}: Cash shortage before sell-off: {cash[idx]:.4f}"
            )
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

    # 6. リバランス
    if strategy.rebalance_interval > 0 and (
        m + 1) % strategy.rebalance_interval == 0:
      reb_paths = active_paths
      if np.any(reb_paths):
        # リバランス時の純資産合計
        total_net = cash[reb_paths].copy()
        for name in units:
          # 各アセットの評価額を加算 (m+1月、つまり来月頭の価格を使用)
          asset_price_at_rebalance = local_monthly_asset_prices[name][reb_paths,
                                                                      m + 1]
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
          cur_ann_spend = np.maximum(0.0, net_reg_spend_m[reb_paths]) * 12.0
          
          # リバランス時の未払い税金を計算する（正確な Post-tax Net Value を求めるため）
          unrealized_gains_at_rebalance = np.zeros_like(total_net)
          for name in units:
            asset_price_at_rebalance = local_monthly_asset_prices[name][reb_paths, m + 1]
            gain = units[name][reb_paths] * asset_price_at_rebalance - units[name][reb_paths] * average_cost[name][reb_paths]
            unrealized_gains_at_rebalance += np.maximum(0.0, gain)
          post_tax_net = total_net - unrealized_gains_at_rebalance * strategy.tax_rate
          
          target_ratios = strategy.dynamic_rebalance_fn(total_net,
                                                        cur_ann_spend,
                                                        rem_years,
                                                        post_tax_net)
          # --- DEBUG ---
          if debug_indices is not None and debug_results is not None:
            for idx in debug_indices:
              if idx < len(reb_paths) and reb_paths[idx]:
                mask_idx = np.sum(reb_paths[:idx])
                debug_results[idx].append(
                    f"[Debug Path {idx}] Month {m} Rebalance: cur_ann_spend={cur_ann_spend[mask_idx]:.2f}, rem_years={rem_years:.4f}, total_net={total_net[mask_idx]:.2f}, post_tax_net={post_tax_net[mask_idx]:.2f}, target_ratios={ {k: (v if isinstance(v, float) else v[mask_idx]) for k, v in target_ratios.items()} }"
                )
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
          current_units = units[name]
          current_asset_val = current_units[reb_paths] * price_for_sell
          diff = current_asset_val - total_net * target_ratios[name]
          sell_mask = diff > 1e-8
          if np.any(sell_mask):
            sell_idx = np.where(reb_paths)[0][sell_mask]
            amt = diff[sell_mask]
            p_subset = local_monthly_asset_prices[name][sell_idx, m + 1]
            u_sell = np.zeros_like(amt)
            u_sell[p_subset > 0] = amt[p_subset > 0] / p_subset[p_subset > 0]
            yearly_capital_gains[sell_idx] += amt - u_sell * average_cost[name][
                sell_idx]
            current_units[sell_idx] -= u_sell
            cash[sell_idx] += amt

        # 購入
        for name in normalized_ratio:
          # 最新価格での評価額再計算
          price_for_buy = local_monthly_asset_prices[name][reb_paths, m + 1]
          current_units = units[name]
          val_after_sell = current_units[reb_paths] * price_for_buy
          diff = total_net * target_ratios[name] - val_after_sell
          buy_mask = diff > 1e-8
          if np.any(buy_mask):
            buy_idx = np.where(reb_paths)[0][buy_mask]
            amt = np.minimum(diff[buy_mask], cash[buy_idx])
            p_subset = local_monthly_asset_prices[name][buy_idx, m + 1]
            u_buy = np.zeros_like(amt)
            u_buy[p_subset > 0] = amt[p_subset > 0] / p_subset[p_subset > 0]

            # 平均単価更新
            new_u = current_units[buy_idx] + u_buy
            upd = new_u > 0
            if np.any(upd):
              i_upd = buy_idx[upd]
              avg_cost_arr = average_cost[name]
              avg_cost_arr[i_upd] = (
                  current_units[i_upd] * avg_cost_arr[i_upd] +
                  u_buy[upd] * p_subset[upd]) / new_u[upd]

            current_units[buy_idx] += u_buy
            cash[buy_idx] -= amt

    # 7. 破産判定と年末記録
    total_val = cash.copy()
    for name, u in units.items():
      total_val[active_paths] += u[active_paths] * local_monthly_asset_prices[
          name][active_paths, m + 1]

    new_bankrupt = active_paths & (total_val < strategy.initial_loan)
    # --- DEBUG ---
    if debug_indices is not None and debug_results is not None:
      for idx in debug_indices:
        if idx < len(active_paths) and active_paths[idx]:
          debug_results[idx].append(
              f"[Debug Path {idx}] Month {m} Step 7: cash={cash[idx]:.2f}, total_val={total_val[idx]:.2f}, new_bankrupt={new_bankrupt[idx]}"
          )
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

  post_tax_net_values: Optional[np.ndarray] = None
  if calculate_post_tax:
    # 最終的な未払い税金を計算する
    # 1. 最後の年に確定したキャピタルゲインに対する税金（m=11等で処理済みの場合は tax_to_pay に入っている）
    # 2. まだ tax_to_pay に移っていない、確定済みのキャピタルゲインに対する税金
    unpaid_tax = tax_to_pay + np.maximum(yearly_capital_gains, 0.0) * strategy.tax_rate

    # 3. 現在保有している資産の含み益に対する税金
    unrealized_gains = np.zeros(n_sim, dtype=np.float64)
    for name, u in units.items():
      current_price = local_monthly_asset_prices[name][:, total_months]
      gain = u * current_price - u * average_cost[name]
      unrealized_gains += np.maximum(0.0, gain)
    
    tax_on_unrealized = unrealized_gains * strategy.tax_rate
    
    # 全ての税金を引いた保守的な見積もり（Post-tax Net Value）
    post_tax_net_values = net_values - unpaid_tax - tax_on_unrealized

  return SimulationResult(net_values=net_values,
                          sustained_months=sustained_months,
                          post_tax_net_values=post_tax_net_values,
                          annual_spends=annual_spends_record,
                          debug_results=debug_results)
