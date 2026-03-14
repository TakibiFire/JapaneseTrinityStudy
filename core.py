"""
日本版トリニティ・スタディのシミュレーションに必要なデータクラスと関数群。

資産クラスや戦略の定義、およびモンテカルロ法を用いた資産推移の計算など、
シミュレーション全体で使われる共通のユーティリティ関数を提供する。
"""

import dataclasses
from typing import Dict, List, Union, cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. グローバルパラメータ
# ---------------------------------------------------------------------------
SEED = 42
MU = 0.07  # オルカン期待リターン 7%
SIGMA = 0.15  # オルカンボラティリティ 15%
YEARS = 50  # シミュレーション期間（年）
TRADING_DAYS = 252  # 1年あたりの営業日数
N_SIM = 1000  # モンテカルロ・シミュレーションのパス数

# ---------------------------------------------------------------------------
# 2. データ構造 (Dataclasses)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Asset:
  """
  シミュレーションにおける単一の資産クラスを定義する。
  
  Attributes:
    name: 資産の名前 (例: "オルカン", "レバカン")
    yearly_cost: 信託報酬など、資産を保有するための年率コスト (割合)
    leverage: 資産のレバレッジ倍率 (1: 通常, 2: 2倍レバレッジ)
  """
  name: str
  yearly_cost: float
  leverage: int


@dataclasses.dataclass(frozen=True)
class ZeroRiskAsset:
  """
  無リスク資産（価格変動なし、固定利回り）を定義する。
  
  Attributes:
    name: 資産の名前 (例: "現金", "米国短期国債")
    yield_rate: 年利 (割合)
  """
  name: str
  yield_rate: float


@dataclasses.dataclass
class Strategy:
  """
  シミュレーションの初期状態と、運用期間中の取り崩し・利払いルールを定義する。
  
  Attributes:
    name: 戦略の名前
    initial_money: 自己資金の初期額 (万円)
    initial_loan: 証券担保ローンの初期借入額 (万円)
    yearly_loan_interest: ローンの年利 (割合)
    initial_asset_ratio: 各資産への初期投資割合。合計が 1.0 以下の時、残りは現金とする。
                         キーは Asset の name (str) または ZeroRiskAsset インスタンス。
    annual_cost: 初年の生活費 (万円)。月割で取り崩す。
    annual_cost_inflation: 生活費の年率インフレ率 (割合)
    selling_priority: 現金不足時に売却する資産の優先順位 (資産の name のリスト)
    tax_rate: 譲渡益に対する税率。デフォルトは 20.315% (0.20315)
    rebalance_interval: リバランスを実行する間隔 (月数)。0 の場合は実行しない。
  """
  name: str
  initial_money: float
  initial_loan: float
  yearly_loan_interest: float
  initial_asset_ratio: Dict[Union[str, ZeroRiskAsset], float]
  annual_cost: float
  annual_cost_inflation: float
  selling_priority: List[str]
  tax_rate: float = 0.20315
  rebalance_interval: int = 0

  def __post_init__(self):
    """
    selling_priority に含まれる資産名が initial_asset_ratio に存在するか検証する。
    """
    valid_names = set()
    for key in self.initial_asset_ratio.keys():
      if isinstance(key, ZeroRiskAsset):
        valid_names.add(key.name)
      else:
        valid_names.add(key)

    for name in self.selling_priority:
      if name not in valid_names:
        raise ValueError(
            f"Selling priority asset '{name}' not found in initial_asset_ratio."
        )


@dataclasses.dataclass
class SimulationResult:
  """
  シミュレーションの実行結果を保持するクラス。
  """
  # 各パスの最終純資産額 (万円)。shape: (n_sim,)
  net_values: np.ndarray

  # 各パスが破産せずに継続できた月数。shape: (n_sim,)
  # 破産しなかった場合は、シミュレーションの総月数 (YEARS * 12) が入る。
  sustained_months: np.ndarray


# ---------------------------------------------------------------------------
# 3. コア機能のシグネチャと数学的仕様
# ---------------------------------------------------------------------------


def generate_monthly_asset_prices(assets: List[Asset],
                                  mu: float = MU,
                                  sigma: float = SIGMA,
                                  years: int = YEARS,
                                  trading_days: int = TRADING_DAYS,
                                  n_sim: int = N_SIM,
                                  seed: int = SEED) -> Dict[str, np.ndarray]:
  """
  幾何ブラウン運動に基づいて、各資産の月次価格推移をシミュレーションする。
  
  モンテカルロ法により指定されたパス数分の株価推移を日次で計算し、
  そこから資産固有のコストとレバレッジを考慮したうえで月次単位の価格に
  集約する。初期値を 1.0 とする。
  
  Args:
    assets: シミュレーション対象となる Asset インスタンスのリスト。
    mu: ベースとなる期待リターン (年率)。
    sigma: ベースとなるボラティリティ (年率)。
    years: シミュレーション期間 (年)。
    trading_days: 1年あたりの営業日数 (日次シミュレーション用)。
    n_sim: モンテカルロ・シミュレーションのパス数。
    seed: 乱数シード。
    
  Returns:
    資産名をキー、shape が (n_sim, 12 * years + 1) の numpy 配列を値とする Dict。
    初期の0ヶ月目 (価格 1.0) を含むため、要素数は月数 + 1 になる。
  """
  np.random.seed(seed)

  # 時間ステップ
  dt = 1.0 / trading_days

  # 総日数
  total_days = years * trading_days

  # 乱数の生成 Z ~ N(0, 1)
  Z = np.random.normal(0, 1, (n_sim, total_days))

  # ベースリターン: r_base = exp((mu - 0.5 * sigma^2)dt + sigma * sqrt(dt) * Z) - 1
  # log_returns_base = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
  # r_base = np.exp(log_returns_base) - 1

  # シンプルな幾何ブラウン運動のリターン (design.md に従う)
  r_base = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) - 1

  monthly_prices: Dict[str, np.ndarray] = {}

  days_per_month = trading_days // 12
  total_months = years * 12

  for asset in assets:
    # 日次コスト
    c_daily = asset.yearly_cost / trading_days

    # 日次倍率: M_daily = max(1 + L * r_base - c_daily, 0)
    m_daily = np.maximum(1.0 + asset.leverage * r_base - c_daily, 0.0)

    # 月次へ集約するために、shape を (n_sim, total_months, days_per_month) に変更
    m_daily_reshaped = m_daily.reshape(n_sim, total_months, days_per_month)

    # 各月の倍率を計算 (日次倍率の積)
    m_monthly = np.prod(m_daily_reshaped, axis=2)

    # 月次価格推移 (累積積)
    # 初期値 1.0 を追加する
    prices = np.ones((n_sim, total_months + 1), dtype=np.float64)
    prices[:, 1:] = np.cumprod(m_monthly, axis=1)

    monthly_prices[asset.name] = prices

  return monthly_prices


def simulate_strategy(
    strategy: Strategy,
    monthly_asset_prices: Dict[str, np.ndarray]) -> SimulationResult:
  """
  指定された戦略パラメータに従い、各シミュレーションパスにおける最終的な純資産総額を計算する。
  
  月ごとの生活費取り崩し、ローン利払い、必要に応じた資産売却 (税金計算を含む) 
  およびリバランス処理を実行する。総資産がローン借入額を下回った場合は破産とみなす。
  
  Args:
    strategy: 実行する投資戦略を定義した Strategy インスタンス。
    monthly_asset_prices: generate_monthly_asset_prices() で計算された各資産の月次価格推移。
    
  Returns:
    SimulationResult インスタンス。
  """
  # 引数の辞書を変更しないようにコピーを作成
  local_monthly_asset_prices = dict(monthly_asset_prices)

  # 任意の資産から n_sim と total_months を取得 (既存の価格データがあればそれを使用)
  if local_monthly_asset_prices:
    sample_asset_name = list(local_monthly_asset_prices.keys())[0]
    prices_shape = local_monthly_asset_prices[sample_asset_name].shape
    n_sim = prices_shape[0]
    total_months = prices_shape[1] - 1  # 初期月が含まれるため -1
  else:
    # monthly_asset_pricesが空で、ZeroRiskAssetのみの場合のフォールバック
    # 現実的には n_sim と total_months が分からないため、これは想定外だが
    # 今回の要件の範囲では通常資産が少なくとも1つはあるか、外部から渡されると想定する。
    # とはいえ安全のため N_SIM, YEARS を使うことも考えられる。
    n_sim = N_SIM
    total_months = YEARS * 12

  # ZeroRiskAsset の処理と initial_asset_ratio の正規化
  # keysは資産名(str)、valuesは投資割合(float)のDict。
  # 正規化が必要な理由は、strategy.initial_asset_ratio が ZeroRiskAsset オブジェクトと
  # 文字列の両方をキーとして受け入れる仕様である一方、これ以降のシミュレーションの
  # 主要なデータ構造(units, average_cost, monthly_asset_prices)は、
  # 全て文字列の「資産名」をキーとして状態を管理するように設計されているため。
  normalized_ratio: Dict[str, float] = {}
  zero_risk_assets: List[ZeroRiskAsset] = []

  for key, ratio in strategy.initial_asset_ratio.items():
    if isinstance(key, ZeroRiskAsset):
      asset_name = key.name
      normalized_ratio[asset_name] = float(ratio)
      zero_risk_assets.append(key)
      # ZeroRiskAsset用の価格推移(常に1.0)を生成して追加
      if asset_name not in local_monthly_asset_prices:
        local_monthly_asset_prices[asset_name] = np.ones(
            (n_sim, total_months + 1), dtype=np.float64)
    else:
      str_key = cast(str, key)
      normalized_ratio[str_key] = float(ratio)

  # 初期資金
  total_capital = strategy.initial_money + strategy.initial_loan

  # 各資産の初期投資割合の合計
  total_ratio = sum(normalized_ratio.values())

  # 初期状態の確保
  cash = np.full(n_sim, total_capital * (1.0 - total_ratio), dtype=np.float64)

  # 各資産の保有口数 (初期価格は1.0のため、金額＝口数)
  # keysは資産名(str)、valuesは各シミュレーションパスにおける保有口数の配列(shape: (n_sim,))。
  # シミュレーション上、全資産(通常資産も無リスク資産も)の初期価格を1.0として開始するため、
  # 初期の保有口数は、各資産への初期投資金額(万円)と完全に一致する仕様となっている。
  units: Dict[str, np.ndarray] = {}
  for asset_name, ratio in normalized_ratio.items():
    units[asset_name] = np.full(n_sim, total_capital * ratio, dtype=np.float64)

  # 税金計算用の配列
  yearly_capital_gains = np.zeros(n_sim, dtype=np.float64)
  tax_to_pay = np.zeros(n_sim, dtype=np.float64)

  # 各資産の平均取得単価
  average_cost: Dict[str, np.ndarray] = {}
  for asset_name in normalized_ratio.keys():
    average_cost[asset_name] = np.ones(n_sim, dtype=np.float64)

  # 破産フラグ (True なら破産)
  bankrupt = np.zeros(n_sim, dtype=bool)

  # 各パスが破産せずに継続できた月数
  sustained_months = np.full(n_sim, total_months, dtype=np.int32)

  # 最終的な純資産総額を保存する配列
  net_values = np.zeros(n_sim, dtype=np.float64)

  # 月次ループ
  for m in range(total_months):
    # すべてのパスが破産していればループ終了
    if np.all(bankrupt):
      break

    # ZeroRiskAsset の利回り支払い (税引き後を直接 cash に加算)
    active_paths = ~bankrupt
    if np.any(active_paths):
      for zr_asset in zero_risk_assets:
        asset_name = zr_asset.name
        if asset_name in units:
          # units == 金額 (価格が常に1.0のため)
          # 利回り = (保有口数) * (年利 / 12) * (1 - 税率)
          yield_payment = units[asset_name] * (zr_asset.yield_rate /
                                               12.0) * (1.0 - strategy.tax_rate)
          cash[active_paths] += yield_payment[active_paths]

    # mヶ月目の生活費 (インフレ考慮)
    cost_m = (strategy.annual_cost /
              12.0) * (1.0 + strategy.annual_cost_inflation)**(m / 12.0)

    # 月次利息支払額
    interest = strategy.initial_loan * (strategy.yearly_loan_interest / 12.0)

    # 必要な現金
    required_cash = np.full(n_sim, cost_m + interest, dtype=np.float64)

    # 1月 (m % 12 == 0) の場合、前年に確定した税金を支払う (初年除く)
    if m % 12 == 0 and m > 0:
      required_cash += tax_to_pay
      tax_to_pay.fill(0.0)

    # 現金の取り崩し
    # 破産していないパスのみ現金を減少させる
    cash[active_paths] -= required_cash[active_paths]

    # 現金不足のパスを特定
    shortage_paths = active_paths & (cash < 0)

    if np.any(shortage_paths):
      # 優先順位に従って資産を売却し、現金を補填
      for asset_name in strategy.selling_priority:
        if asset_name not in units:
          continue

        # まだ現金が不足しているパス
        still_shortage = shortage_paths & (cash < 0)
        if not np.any(still_shortage):
          break

        # 現在の月次価格 (m+1 が現在の月末価格、m=0の時 prices[:, 1])
        # 取り崩しは月末に行うと仮定する
        current_price = local_monthly_asset_prices[asset_name][still_shortage,
                                                               m + 1]

        # 保有資産の評価額
        asset_value = units[asset_name][still_shortage] * current_price

        # 不足分 (正の値)
        shortage_amount = -cash[still_shortage]

        # 売却額: min(保有資産評価額, 不足分)
        sell_amount = np.minimum(asset_value, shortage_amount)

        # 売却した分だけ口数を減らす
        # current_price が 0 の場合は売却できないため 0 で割るのを防ぐ
        valid_price_mask = current_price > 0
        units_to_sell = np.zeros_like(sell_amount)
        units_to_sell[valid_price_mask] = sell_amount[
            valid_price_mask] / current_price[valid_price_mask]

        # 譲渡益の計算
        gain = sell_amount - units_to_sell * average_cost[asset_name][
            still_shortage]
        yearly_capital_gains[still_shortage] += gain

        units[asset_name][still_shortage] -= units_to_sell

        # 現金に加算
        cash[still_shortage] += sell_amount

    # リバランス処理
    if strategy.rebalance_interval > 0 and (
        m + 1) % strategy.rebalance_interval == 0:
      rebalance_paths = ~bankrupt
      if np.any(rebalance_paths):
        # 現在の総純資産額(total_net_value)の計算
        current_total_net_value = cash[rebalance_paths].copy()
        current_values = {}
        for asset_name in units:
          current_price = local_monthly_asset_prices[asset_name][
              rebalance_paths, m + 1]
          current_val = units[asset_name][rebalance_paths] * current_price
          current_values[asset_name] = current_val
          current_total_net_value += current_val

        # 売却処理 (目標よりも多い資産を売る)
        for asset_name, ratio in normalized_ratio.items():
          target_val = current_total_net_value * ratio
          current_val = current_values[asset_name]
          diff = current_val - target_val

          # 超過分が微小な誤差以上の場合のみ売却
          sell_mask = diff > 1e-8
          if np.any(sell_mask):
            sell_paths_idx = np.where(rebalance_paths)[0][sell_mask]
            sell_amount = diff[sell_mask]
            current_price = local_monthly_asset_prices[asset_name][
                sell_paths_idx, m + 1]

            valid_price_mask = current_price > 0
            units_to_sell = np.zeros_like(sell_amount)
            units_to_sell[valid_price_mask] = sell_amount[
                valid_price_mask] / current_price[valid_price_mask]

            # 譲渡益の計算
            avg_cost = average_cost[asset_name][sell_paths_idx]
            gain = sell_amount - units_to_sell * avg_cost
            yearly_capital_gains[sell_paths_idx] += gain

            units[asset_name][sell_paths_idx] -= units_to_sell
            cash[sell_paths_idx] += sell_amount

        # 購入処理 (目標よりも少ない資産を買う)
        for asset_name, ratio in normalized_ratio.items():
          # 現時点での評価額を再計算（売却後の現金を反映するため、というより目標値と比較するため）
          target_val = current_total_net_value * ratio
          current_price_full = local_monthly_asset_prices[asset_name][
              rebalance_paths, m + 1]
          current_val = units[asset_name][rebalance_paths] * current_price_full
          diff = target_val - current_val

          buy_mask = diff > 1e-8
          if np.any(buy_mask):
            buy_paths_idx = np.where(rebalance_paths)[0][buy_mask]
            # 現金残高を超えないように購入額をクリッピング
            buy_amount = np.minimum(diff[buy_mask], cash[buy_paths_idx])

            # 購入可能な現金が微小な場合はスキップ
            actual_buy_mask = buy_amount > 1e-8
            if np.any(actual_buy_mask):
              buy_paths_idx = buy_paths_idx[actual_buy_mask]
              buy_amount = buy_amount[actual_buy_mask]
              current_price = local_monthly_asset_prices[asset_name][
                  buy_paths_idx, m + 1]

              valid_price_mask = current_price > 0
              units_to_buy = np.zeros_like(buy_amount)
              units_to_buy[valid_price_mask] = buy_amount[
                  valid_price_mask] / current_price[valid_price_mask]

              # 平均取得単価の更新
              current_units = units[asset_name][buy_paths_idx]
              current_avg_cost = average_cost[asset_name][buy_paths_idx]
              new_total_units = current_units + units_to_buy

              # 口数が0より大きい場合のみ更新
              update_mask = new_total_units > 0
              if np.any(update_mask):
                update_idx = buy_paths_idx[update_mask]
                average_cost[asset_name][update_idx] = (
                    current_units[update_mask] * current_avg_cost[update_mask] +
                    units_to_buy[update_mask] *
                    current_price[update_mask]) / new_total_units[update_mask]

              units[asset_name][buy_paths_idx] += units_to_buy
              cash[buy_paths_idx] -= buy_amount

    # 月末の総資産の計算 (現金 + 各資産の評価額)
    total_assets = cash.copy()
    for asset_name, unit_array in units.items():
      total_assets[active_paths] += unit_array[
          active_paths] * local_monthly_asset_prices[asset_name][active_paths,
                                                                 m + 1]

    # 破産判定: 総資産 < 初期借入額
    new_bankrupts = active_paths & (total_assets < strategy.initial_loan)
    bankrupt[new_bankrupts] = True
    sustained_months[new_bankrupts] = m

    # 年末 (12月) に税額を確定する
    if m % 12 == 11:
      tax_to_pay = np.maximum(yearly_capital_gains, 0.0) * strategy.tax_rate
      yearly_capital_gains.fill(0.0)

    # 最終月の場合は純資産を記録
    if m == total_months - 1:
      survivors = ~bankrupt
      net_values[survivors] = total_assets[survivors] - strategy.initial_loan

  return SimulationResult(net_values=net_values,
                          sustained_months=sustained_months)


def create_styled_summary(
    results: Dict[str, SimulationResult]) -> "pd.io.formats.style.Styler":
  """
  シミュレーション結果の辞書からサマリー統計を計算し、
  フォーマットされた Styler オブジェクトを返す。
  
  分位点や破産確率など、複数の指標を算出して視覚的に整えたテーブルを作成する。
  
  Args:
    results: 戦略名をキー、SimulationResult インスタンスを値とする辞書。
  
  Returns:
    表示用にフォーマット・スタイリングされた pandas Styler オブジェクト。
  """
  summary_data = {}
  for name, res in results.items():
    net_values = res.net_values
    sustained_months = res.sustained_months

    summary_data[name] = {
        "下位1% (だいぶ運が悪い)": np.quantile(net_values, 0.01),
        "下位10% (運が悪い)": np.quantile(net_values, 0.10),
        "下位25% (やや不運)": np.quantile(net_values, 0.25),
        "中央値 (普通)": np.median(net_values),
        "上位25% (やや幸運)": np.quantile(net_values, 0.75),
        "上位10% (運が良い)": np.quantile(net_values, 0.90),
        "20年破産確率 (%)": np.mean(sustained_months < 20 * 12) * 100.0,
        "30年破産確率 (%)": np.mean(sustained_months < 30 * 12) * 100.0,
        "40年破産確率 (%)": np.mean(sustained_months < 40 * 12) * 100.0,
        "50年破産確率 (%)": np.mean(sustained_months < 50 * 12) * 100.0,
    }

  summary_df = pd.DataFrame(summary_data).T

  def format_oku(x: float) -> str:
    return f"約 {x / 10000:.1f}億円"

  def format_pct(x: float) -> str:
    return f"{x:.1f}%"

  styled_summary = summary_df.style.format({
      "下位1% (だいぶ運が悪い)": format_oku,
      "下位10% (運が悪い)": format_oku,
      "下位25% (やや不運)": format_oku,
      "中央値 (普通)": format_oku,
      "上位25% (やや幸運)": format_oku,
      "上位10% (運が良い)": format_oku,
      "20年破産確率 (%)": format_pct,
      "30年破産確率 (%)": format_pct,
      "40年破産確率 (%)": format_pct,
      "50年破産確率 (%)": format_pct,
  })
  styled_summary.index.name = "戦略"

  return styled_summary
