"""
シミュレーションシナリオを構築するための高レベルな宣言型API。
ビジネスロジックを低レベルなエンジン設定（Strategy, AssetConfigs, CashflowConfigs）
に変換するコンパイラのようなインターフェースを提供する。
"""

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import scipy.stats as stats

from src.core import DynamicSpending, Strategy, ZeroRiskAsset
from src.lib.asset_generator import (Asset, AssetConfigType, CpiAsset,
                                     DerivedAsset, ForexAsset,
                                     MonthlyARLogNormal, MonthlyLogDist,
                                     MonthlyLogNormal, SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (BaseSpendConfig, CashflowConfig,
                                        CashflowDynamicHandler, CashflowRule,
                                        CashflowType, MortalityConfig,
                                        PensionConfig, generate_cashflows)
from src.lib.dp_predictor import DPOptimalStrategyPredictor
from src.lib.dynamic_rebalance import calculate_optimal_strategy
from src.lib.dynamic_rebalance_dp import calculate_optimal_strategy_dp
from src.lib.life_table import FEMALE_MORTALITY_RATES, MALE_MORTALITY_RATES
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.simulation_defaults import (AcwiModelKey,
                                         get_acwi_fat_tail_config,
                                         get_cpi_ar12_1981_config,
                                         get_cpi_ar12_config)
from src.lib.spend_aware_dynamic_spending import SpendAwareDynamicSpending

# --- 厳密な型付けのための列挙型 (Enums) ---


class PredefinedStock(Enum):
  """
  シナリオビルダーによって解決されるサポート対象の株式（リスク）資産。
  各資産には、標準的な信託報酬 (trust_fee) があらかじめ設定されているとみなされる。
  """
  # S&P500の155年分の将来時系列との相関を基に算出された全世界株式 (オール・カントリー) モデル。
  # 信託報酬 0.05775% 以内。
  ORUKAN_155 = auto()
  # 155年分のデータに基づいた米国株式 (S&P500) モデル。
  # 信託報酬 0.0814% 以内。
  SP500_155 = auto()
  # 30年分のデータに基づいた米国株式 (S&P500) モデル。
  SP500_30Y = auto()
  # 18年分のデータに基づいた全世界株式 (ACWI) モデル。
  ACWI_18Y = auto()
  # ACWI の対数正規分布モデル。
  ACWI_LOGNORMAL = auto()
  # ACWI の Johnson SU 分布モデル。
  ACWI_JSU = auto()
  # 算術平均 7%, 標準偏差 15% のシンプルな対数正規分布モデル（為替なし、信託報酬なし）。
  SIMPLE_7_15_ORUKAN = auto()


class PredefinedZeroRisk(Enum):
  """
  シナリオビルダーによって解決されるサポート対象のゼロリスク資産。
  """
  # 現金 (コンテキストに応じて日本円または米ドル)。
  CASH = auto()
  # 年率4%の利回りを持つゼロリスク資産 (米国債など)。
  ZERO_RISK_4PCT = auto()


# 株式とゼロリスク資産のいずれかを表す型。
PredefinedAsset = Union[PredefinedStock, PredefinedZeroRisk]


class CpiType(Enum):
  """消費者物価指数 (CPI) モデルの種類。"""
  # 過去の日本のCPI (AR12モデル)。物価の粘着性を考慮。
  JAPAN_AR12 = auto()
  # 固定の年率1.77%インフレ。
  FIXED_1_77 = auto()
  # インフレなし (0%)。
  FIXED_0 = auto()
  # 実験用: インフレ率 1.0% (ボラティリティ 0)
  FIXED_1_0 = auto()
  # 実験用: インフレ率 1.5% (ボラティリティ 0)
  FIXED_1_5 = auto()
  # 実験用: インフレ率 2.0% (ボラティリティ 0)
  FIXED_2_0 = auto()
  # 実験用: 歴史的平均 2.44% (ボラティリティ 0)
  FIXED_2_44 = auto()
  # 実験用: 平均 2.0%, ボラティリティ 2.0%
  FIXED_2_0_VOL_2_0 = auto()
  # 実験用: 平均 2.0%, 歴史的標準偏差 4.13%
  FIXED_2_0_VOL_4_13 = auto()
  # 実験用: 歴史的平均 2.44%, 歴史的標準偏差 4.13%
  FIXED_2_44_VOL_4_13 = auto()
  # 実験用: AR(12) 粘着性モデル (1981年〜)
  JAPAN_AR12_1981 = auto()


class FxType(Enum):
  """
  非日本円資産のための為替(Forex)モデル。
  
  実装詳細:
  USDJPY モデルは、年次対数正規分布 (YearlyLogNormal) を使用し、
  期待値 mu とボラティリティ sigma を用いて為替変動を生成する。
  """
  # 為替なし (すべて日本円として扱うか、変換済みと仮定)。
  NONE = auto()
  # 標準的なUSD/JPYモデル (mu=0.0, sigma=0.1053)。
  USDJPY = auto()
  # 低リスクパラメータのUSD/JPYモデル (mu=0.0, sigma=0.05)。
  USDJPY_LOW_RISK = auto()
  # 修正パラメータのUSD/JPYモデル (mu=0.01, sigma=0.10)。
  USDJPY_MODIFIED = auto()


class PensionStatus(Enum):
  """
  年金保険料および受給額に影響する雇用・保険ステータス。
  
  注記:
  NONE 以外は、日本の公的年金制度（国民年金・厚生年金）への加入を前提とする。
  """
  # 標準的な会社員 (厚生年金 + 基礎年金)。保険料を全額支払い、受給も満額。
  FULL = auto()
  # 保険料免除。保険料の支払いは免除されるが、基礎年金の受給額が 1/2（国庫負担分のみ）に減額される。
  EXEMPT = auto()
  # 保険料未納。保険料を支払わず、その期間に対応する基礎年金を受給できない。
  # (受給額の計算において、未納期間分が反映されない状態)
  UNPAID = auto()
  # 年金制度に一切関与しない。保険料も支払わず、将来の受給も完全にゼロ。
  NONE = auto()


class Gender(Enum):
  """生命表選択のための性別。"""
  MALE = auto()
  FEMALE = auto()


# --- 型付けされた戦略パラメータ ---


@dataclass(frozen=True)
class DynamicV1Adjustment:
  """
  バンガードスタイルのダイナミックスペンディング (バージョン1) 設定。
  
  資産残高の一定割合を取り崩そうとするが、前年比の増減幅を一定範囲に制限する。
  """
  # 目標となる年間引き出し率 (例: 4%の場合は0.04)。
  target_ratio: float
  # 前年のインフレ調整後支出に対する最大増加率。
  upper_limit: float = 0.01
  # 前年のインフレ調整後支出に対する最大減少率。
  lower_limit: float = -0.015
  # 初期年額引き出し額。指定がない場合は初期支出額が使用されます。
  initial_annual_spend: Optional[float] = None


@dataclass(frozen=True)
class SpendAwareAdjustment:
  """
  DP（動的計画法）ベースの支出認識型ダイナミックスペンディング (バージョン2) 設定。
  
  生存確率と効率性を考慮した最適な支出額を決定する。
  """
  # JSONモデルファイルのパス (例: data/optimal_strategy_v2_models.json)。
  model_name: str
  # 目標生存確率の下限。
  p_low: float = 0.85
  # 目標生存確率の上限。
  p_high: float = 0.97
  # 前年比の支出倍率の下限。
  lower_mult: float = 0.99
  # 前年比の支出倍率の上限。
  upper_mult: float = 1.02


@dataclass(frozen=True)
class DynamicV1Rebalance:
  """
  標準的な固定間隔のリバランス。
  指定された月数ごとに、目標とする資産配分比率に戻す。
  """
  # リバランス対象の株式資産。例: ORUKAN_155
  risky_asset: PredefinedStock
  # リバランスの残りの振り分け先（ゼロリスク資産）。例: ZERO_RISK_4PCT
  zero_risk_asset: PredefinedZeroRisk
  # リバランスの間隔
  interval_months: int = 12


@dataclass(frozen=True)
class SpendAwareDPRebalance:
  """
  生存確率・効率性モデルを使用したDPベースのリバランス。

  注意:
  現在、このリバランス方式は 12ヶ月（1年）間隔でのみ実行されるよう設計されており、
  他の間隔での実行はサポートされていない。
  """
  # リバランス対象の株式資産。例: ORUKAN_155
  risky_asset: PredefinedStock
  # リバランスの残りの振り分け先（ゼロリスク資産）。例: ZERO_RISK_4PCT
  zero_risk_asset: PredefinedZeroRisk
  # 使用するDPモデルファイルのパス。
  model_name: str


# --- 高レベルな宣言 ---


@dataclass(frozen=True)
class WorldConfig:
  """
  シミュレーションのためのマクロ経済環境を定義する。
  
  Attributes:
    n_sim: シミュレーションするパスの数。
    n_years: シミュレーションの期間（年数）。
    start_age: 個人のシミュレーション開始時の年齢。
    tax_rate: キャピタルゲイン税率（デフォルトは日本の分離課税 20.315%）。
    seed: 乱数生成の再現性のためのシード値。
    cpi_type: 物価変動モデルの選択。
    fx_type: 外貨建て資産（米国株等）に適用する為替モデル。
    current_year: シミュレーション開始時の西暦。
    macro_economic_slide_end_year: マクロ経済スライドによる年金抑制が終了する年。
  """
  n_sim: int
  n_years: int
  start_age: int
  tax_rate: float = 0.20315
  seed: int = 42
  cpi_type: CpiType = CpiType.JAPAN_AR12
  fx_type: FxType = FxType.USDJPY
  current_year: int = 2026
  macro_economic_slide_end_year: int = 2057


@dataclass(frozen=True)
class ConstantSpend:
  """
  年間の名目支出額を一定にする設定。
  
  動作: 
  指定された年間支出額を12で割った額を毎月の支出とする。
  インフレ調整が適用される場合、名目額は CPI に連動して増加する。
  """
  # 年間の基本支出額 (万円/年)。
  annual_amount: float


@dataclass(frozen=True)
class CurveSpend:
  """
  統計データ（家計調査等）に基づく、年齢とともに変動する支出カーブ。
  
  Attributes:
    first_year_annual_amount: 開始年齢 (start_age) 時点での年間の支出額 (万円/年)。
      None の場合、統計データの生の金額がそのまま使用される。
      数値を指定した場合、開始年齢時点の金額がこの値になるように、カーブ全体がスケーリングされる。
    spending_types: 適用する統計データの種類（消費支出、非消費支出等）。
  """
  first_year_annual_amount: Optional[float] = None
  spending_types: Tuple[SpendingType,
                        ...] = (SpendingType.CONSUMPTION,
                                SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION)


@dataclass(frozen=True)
class Lifeplan:
  """
  個人のライフイベントとキャッシュフローの高レベルな宣言。
  
  Attributes:
    base_spend: 基本支出モデル (Constant または Curve)。年間の支出額を指定する。
    retirement_start_age: 定期的な給与収入が停止する年齢。
      将来の年金受給額の計算において、22歳からこの年齢（この年齢自体は含まない）までの期間、厚生年金に加入していたものとして扱われる。
      例えば 35歳でリタイアした場合、34.999...歳までの加入実績に基づいた年金額が計算される。
    pension_status: 公的年金への加入・支払い状況。
      年金が存在しない世界をシミュレーションする場合は PensionStatus.NONE を指定する。
    pension_start_age: 公的年金の受給を開始する年齢。60歳から75歳の間で指定可能。
    household_size: 世帯の人数。1の場合は単身世帯、2以上の場合は二人以上の世帯として支出統計等が適用される。
      (COMMENT: 具体的にどの統計データが適用されるかは、今後の実装で詳細化する。)
    side_fire_income_monthly: 副業等による月額の追加収入。
    side_fire_duration_months: 追加収入が得られる期間（月数）。
    mortality_gender: 性別を指定した場合、シミュレーション中に生存確率に基づいた死亡イベントを発生させる。
  """
  base_spend: Union[ConstantSpend, CurveSpend]
  retirement_start_age: int

  pension_status: PensionStatus = PensionStatus.NONE
  pension_start_age: int = 65
  household_size: int = 1

  side_fire_income_monthly: float = 0.0
  side_fire_duration_months: int = 0
  mortality_gender: Optional[Gender] = None


@dataclass(frozen=True)
class StrategySpec:
  """
  投資資産の運用および取り崩し戦略の高レベルな宣言。
  
  Attributes:
    initial_money: シミュレーション開始時の運用資産総額 (万円)。
    initial_asset_ratio: 初期のポートフォリオ配分。((資産名, 比率), ...) の形式で指定。
    selling_priority: 資金不足時にどの資産から順に売却するかを指定。
    rebalance: リバランス戦略の選択。
    spend_adjustment: 資産残高に応じた支出額の動的な調整ロジック。
  """
  initial_money: float
  initial_asset_ratio: Tuple[Tuple[PredefinedAsset, float], ...]
  selling_priority: Tuple[PredefinedAsset, ...]

  rebalance: Union[DynamicV1Rebalance, SpendAwareDPRebalance, None] = None
  spend_adjustment: Union[DynamicV1Adjustment, SpendAwareAdjustment,
                          None] = None


# --- セットアップビルダ API ---


@dataclass(frozen=True)
class _ExperimentVariant:
  """
  ベースラインから変更された個別の実験条件（内部管理用）。
  """
  # 実験名。
  name: str
  # ライフプラン設定。
  lifeplan: Lifeplan
  # 戦略設定。
  strategy: StrategySpec
  # 世界設定。
  world: WorldConfig


@dataclass
class Setup:
  """
  ベースライン設定と、それに対する複数の実験条件（バリアント）を管理するクラス。
  
  Attributes:
    name: ベースライン（Setup全体）の名前。
    world: ベースラインの世界設定。
    lifeplan: ベースラインのライフプラン設定。
    strategy: ベースラインの戦略設定。
    experiments: 追加された実験バリアントのリスト。
  """
  name: str
  world: WorldConfig
  lifeplan: Lifeplan
  strategy: StrategySpec

  # 各実験のバリアントを保持するリスト。
  experiments: List[_ExperimentVariant] = field(default_factory=list)

  def add_experiment(self,
                     name: str,
                     overwrite_lifeplan: Optional[Lifeplan] = None,
                     overwrite_strategy: Optional[StrategySpec] = None,
                     overwrite_world: Optional[WorldConfig] = None):
    """
    ベースラインを元に、一部の設定を変更した新しい実験を追加する。
    指定されなかった項目はベースラインの設定が引き継がれる。

    Args:
      name: 追加する実験の名前。
      overwrite_lifeplan: (オプション) 上書きするライフプラン設定。
      overwrite_strategy: (オプション) 上書きする戦略設定。
      overwrite_world: (オプション) 上書きする世界設定。
    """
    self.experiments.append(
        _ExperimentVariant(name=name,
                           lifeplan=overwrite_lifeplan or self.lifeplan,
                           strategy=overwrite_strategy or self.strategy,
                           world=overwrite_world or self.world))


@dataclass
class CompiledExperiment:
  """
  コンパイラ（create_experiment_setup）によって生成された、シミュレーション実行可能な最終的なオブジェクト。
  """
  # 実験名。
  name: str
  # src.core.Strategy オブジェクト。エンジンの入力として使用される。
  strategy: Strategy
  # 各資産の月次価格推移。Dict キーは資産名、値は shape (n_sim, n_months + 1) の配列。
  monthly_prices: Dict[str, np.ndarray]
  # 生成されたキャッシュフロー（年金等）。Dict キーはソース名、値は shape (n_sim, n_months) の配列。
  monthly_cashflows: Dict[str, np.ndarray]


def create_experiment_setup(
    setup: Setup,
    record_annual_spend: bool = False) -> List[CompiledExperiment]:
  """
  Setup 宣言を受け取り、重複する計算を排除しながら、各実験に対応する実行可能な CompiledExperiment のリストを生成する。
  """
  # 全シナリオ（ベースライン + 追加実験）をフラットなリストにする
  scenarios = [
      _ExperimentVariant(setup.name, setup.lifeplan, setup.strategy,
                         setup.world)
  ] + setup.experiments

  # 1. unique な WorldConfig を収集
  unique_worlds = list({s.world for s in scenarios})

  # 2. World ごとにアセットとキャッシュフローを生成
  # world_data[WorldConfig] = (prices_dict, cashflows_dict, lp_cf_names_dict, lp_real_cost_dict)
  world_data: Dict[WorldConfig, Tuple[Dict[str, np.ndarray], Dict[str,
                                                                  np.ndarray],
                                      Dict[Lifeplan, Dict[str, str]],
                                      Dict[Lifeplan, np.ndarray]]] = {}

  for world in unique_worlds:
    # この World に属するシナリオを抽出
    world_scenarios = [s for s in scenarios if s.world == world]

    # この World で必要な unique な Lifeplan を収集
    unique_lifeplans = list({s.lifeplan for s in world_scenarios})

    # 資産の推移を生成
    # StrategySpec から必要な PredefinedAsset を収集
    required_assets: Set[PredefinedAsset] = set()
    needs_pension = False
    for s in world_scenarios:
      if s.lifeplan.pension_status != PensionStatus.NONE:
        needs_pension = True
      for asset_enum, _ in s.strategy.initial_asset_ratio:
        required_assets.add(asset_enum)
      for asset_enum in s.strategy.selling_priority:
        required_assets.add(asset_enum)

    asset_configs = _compile_assets(required_assets, world, needs_pension)
    monthly_prices = generate_monthly_asset_prices(asset_configs,
                                                   n_paths=world.n_sim,
                                                   n_months=world.n_years * 12,
                                                   seed=world.seed)

    # 全ての Lifeplan から CashflowConfig を生成
    all_cf_configs: List[CashflowConfig] = []
    lp_cf_names_for_world: Dict[Lifeplan, Dict[str, str]] = {}
    lp_real_cost_for_world: Dict[Lifeplan, np.ndarray] = {}
    for lp in unique_lifeplans:
      compiled_lp = _compile_lifeplan(lp, world)
      lp_configs = compiled_lp.configs
      lp_real_cost = compiled_lp.real_cost_curve
      lp_cf_map = {}
      # Lifeplan を文字列化してハッシュ化し、キャッシュフロー名が衝突しないようにする
      lp_hash = hashlib.md5(str(lp).encode()).hexdigest()[:8]
      for cfg in lp_configs:
        original_name = cfg.name
        hashed_name = f"{original_name}_{lp_hash}"
        cfg.name = hashed_name
        lp_cf_map[original_name] = hashed_name
        all_cf_configs.append(cfg)
      lp_cf_names_for_world[lp] = lp_cf_map
      lp_real_cost_for_world[lp] = lp_real_cost

    monthly_cashflows = generate_cashflows(all_cf_configs,
                                           monthly_prices,
                                           n_sim=world.n_sim,
                                           n_months=world.n_years * 12)

    world_data[world] = (monthly_prices, monthly_cashflows,
                         lp_cf_names_for_world, lp_real_cost_for_world)

  # 3. 各シナリオに対して Strategy を構築し CompiledExperiment を作成
  compiled_experiments = []
  for s in scenarios:
    prices, cashflows, lp_cf_names_dict, lp_real_cost_dict = world_data[s.world]
    cf_map = lp_cf_names_dict[s.lifeplan]
    real_cost = lp_real_cost_dict[s.lifeplan]

    strategy = _build_strategy(s, cf_map, prices, cashflows, real_cost,
                               record_annual_spend)
    compiled_experiments.append(
        CompiledExperiment(name=s.name,
                           strategy=strategy,
                           monthly_prices=prices,
                           monthly_cashflows=cashflows))

  return compiled_experiments


def _compile_assets(assets: Set[PredefinedAsset],
                    world: WorldConfig,
                    needs_pension: bool = False) -> List[AssetConfigType]:
  """PredefinedAsset を AssetConfig 系のオブジェクトに変換する。"""
  configs: List[AssetConfigType] = []

  # 1. 為替 (シード値の一貫性のために最初に追加)
  fx_name: Optional[str] = None
  if world.fx_type != FxType.NONE:
    if world.fx_type == FxType.USDJPY:
      mu, sigma = 0.0, 0.1053
    elif world.fx_type == FxType.USDJPY_LOW_RISK:
      mu, sigma = 0.0, 0.05
    elif world.fx_type == FxType.USDJPY_MODIFIED:
      mu, sigma = 0.01, 0.10
    else:
      raise ValueError(f"未知の為替タイプです: {world.fx_type}")

    fx_name = f"USDJPY_{world.fx_type.name}"
    configs.append(ForexAsset(fx_name, YearlyLogNormalArithmetic(mu, sigma)))

  # 2. 株式 (ベース資産の後に派生資産を追加)
  added_base_assets = set()

  # S&P500 155y は共通の依存先
  def ensure_base_sp500_155y():
    if "Base_SP500_155y" not in added_base_assets:
      configs.append(get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y))
      added_base_assets.add("Base_SP500_155y")

  # ORUKAN_155, ACWI_JSU, ACWI_18Y は Base_ACWI_Approx に依存
  if any(a in assets for a in [
      PredefinedStock.ORUKAN_155, PredefinedStock.ACWI_JSU,
      PredefinedStock.ACWI_18Y
  ]):
    ensure_base_sp500_155y()
    base = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)
    if base.name not in added_base_assets:
      configs.append(base)
      added_base_assets.add(base.name)

  if PredefinedStock.SP500_155 in assets:
    ensure_base_sp500_155y()

  # その他の特定モデル
  if PredefinedStock.SP500_30Y in assets:
    if "Base_SP500_30y" not in added_base_assets:
      # 年率 mu=11.64%, sigma=17.14%
      params = (0.4879653982267047, 0.033214317138593324, 0.017280587830235443)
      configs.append(
          Asset(name="Base_SP500_30y",
                dist=MonthlyLogDist(stats.genlogistic, params=params),
                trust_fee=0.0,
                leverage=1))
      added_base_assets.add("Base_SP500_30y")

  if PredefinedStock.ACWI_LOGNORMAL in assets:
    if "Base_ACWI_LogNormal" not in added_base_assets:
      # 年率 mu=8.32%, sigma=16.72%
      configs.append(
          Asset(name="Base_ACWI_LogNormal",
                dist=MonthlyLogDist(stats.norm, params=(0.006393, 0.048285)),
                trust_fee=0.0,
                leverage=1))
      added_base_assets.add("Base_ACWI_LogNormal")

  # 派生資産の追加
  for a in sorted(list(assets), key=lambda x: x.name):
    if a == PredefinedStock.ORUKAN_155:
      configs.append(
          DerivedAsset(name="ORUKAN_155",
                       base="Base_ACWI_Approx",
                       trust_fee=0.0005775,
                       forex=fx_name))
    elif a == PredefinedStock.SP500_155:
      configs.append(
          DerivedAsset(name="SP500_155",
                       base="Base_SP500_155y",
                       trust_fee=0.000814,
                       forex=fx_name))
    elif a == PredefinedStock.SP500_30Y:
      configs.append(
          DerivedAsset(name="SP500_30Y",
                       base="Base_SP500_30y",
                       trust_fee=0.000814,
                       forex=fx_name))
    elif a == PredefinedStock.ACWI_18Y or a == PredefinedStock.ACWI_JSU:
      configs.append(
          DerivedAsset(name=a.name,
                       base="Base_ACWI_Approx",
                       trust_fee=0.0005775,
                       forex=fx_name))
    elif a == PredefinedStock.ACWI_LOGNORMAL:
      configs.append(
          DerivedAsset(name="ACWI_LOGNORMAL",
                       base="Base_ACWI_LogNormal",
                       trust_fee=0.0005775,
                       forex=fx_name))
    elif a == PredefinedStock.SIMPLE_7_15_ORUKAN:
      configs.append(
          Asset(name="SIMPLE_7_15_ORUKAN",
                dist=YearlyLogNormalArithmetic(mu=0.07, sigma=0.15),
                trust_fee=0.0,
                forex=None))
    elif isinstance(a, PredefinedZeroRisk):
      # PredefinedZeroRisk はここでは処理しない
      pass
    else:
      raise ValueError(f"未知の株式タイプです: {a}")

  # 3. 消費者物価指数 (CPI)
  if world.cpi_type == CpiType.JAPAN_AR12:
    configs.append(get_cpi_ar12_config("Japan_CPI"))
  elif world.cpi_type == CpiType.FIXED_1_77:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.0177,
                                                                   0.0)))
  elif world.cpi_type == CpiType.FIXED_0:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.0, 0.0)))
  elif world.cpi_type == CpiType.FIXED_1_0:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.01, 0.0)))
  elif world.cpi_type == CpiType.FIXED_1_5:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.015, 0.0)))
  elif world.cpi_type == CpiType.FIXED_2_0:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.02, 0.0)))
  elif world.cpi_type == CpiType.FIXED_2_44:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.0244, 0.0)))
  elif world.cpi_type == CpiType.FIXED_2_0_VOL_2_0:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.02, 0.02)))
  elif world.cpi_type == CpiType.FIXED_2_0_VOL_4_13:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.02, 0.0413)))
  elif world.cpi_type == CpiType.FIXED_2_44_VOL_4_13:
    configs.append(CpiAsset("Japan_CPI", YearlyLogNormalArithmetic(0.0244, 0.0413)))
  elif world.cpi_type == CpiType.JAPAN_AR12_1981:
    configs.append(get_cpi_ar12_1981_config("Japan_CPI"))
  else:
    raise ValueError(f"未知の CPI タイプです: {world.cpi_type}")

  # 4. 年金用CPI (マクロ経済スライド)
  if needs_pension:
    slide_end_month = (world.macro_economic_slide_end_year -
                       world.current_year) * 12
    configs.append(
        SlideAdjustedCpiAsset(name="Pension_CPI",
                              base_cpi="Japan_CPI",
                              slide_rate=0.005,
                              slide_end_month=max(0, slide_end_month)))

  return configs


@dataclass(frozen=True)
class _CompiledLifeplan:
  """
  Lifeplan のコンパイル結果を保持する内部用データクラス。
  """
  configs: List[CashflowConfig]
  # 統計データに基づく、インフレ調整前の実質支出額の推移。
  real_cost_curve: np.ndarray


def _compile_lifeplan(lp: Lifeplan, world: WorldConfig) -> _CompiledLifeplan:
  """
  Lifeplan を CashflowConfig 系のオブジェクトに変換する。

  Args:
    lp: ライフプラン設定。
    world: 世界設定。

  Returns:
    コンパイルされたキャッシュフロー設定と、実質支出額のカーブ。
  """
  configs: List[CashflowConfig] = []
  cpi_name = "Japan_CPI"

  # Base Spend
  if isinstance(lp.base_spend, ConstantSpend):
    configs.append(
        BaseSpendConfig(name="BaseSpend",
                        amount=lp.base_spend.annual_amount,
                        cpi_name=cpi_name))
    # ConstantSpend の場合、実質コストのカーブは全期間 annual_amount
    real_cost_curve = np.full(world.n_years, lp.base_spend.annual_amount)
  elif isinstance(lp.base_spend, CurveSpend):
    multipliers = get_retired_spending_multipliers(
        spending_types=list(lp.base_spend.spending_types),
        start_age=world.start_age,
        num_years=world.n_years,
        normalize=(lp.base_spend.first_year_annual_amount is not None))
    base_amount = lp.base_spend.first_year_annual_amount or 1.0
    configs.append(
        BaseSpendConfig(name="BaseSpend",
                        amount=(multipliers * base_amount).tolist(),
                        cpi_name=cpi_name))
    real_cost_curve = multipliers * base_amount
  else:
    raise ValueError(f"未知の支出タイプです: {lp.base_spend}")

  # Pension
  if lp.pension_status != PensionStatus.NONE:
    # 22歳から retirement_start_age まで厚生年金に加入、年収500万を想定
    kousei_unit_annual = 2.736
    kiso_full_annual = 81.6
    # 国民年金保険料 (FY2026 value is 17,920/mo = ~21.5万円/yr)
    premium_annual = 21.5

    # 1. 保険料支払い (60歳まで)
    months_to_60 = max(0, (60 - world.start_age) * 12)
    if months_to_60 > 0 and lp.pension_status == PensionStatus.FULL:
      # FULL の場合のみ保険料を支払う (Exempt/Unpaid は 0)
      # 世帯人数が2以上の場合は2人分とみなす
      premium_multiplier = 2.0 if lp.household_size >= 2 else 1.0
      configs.append(
          PensionConfig(name="PensionPremium",
                        amount=-(premium_annual * premium_multiplier / 12.0),
                        start_month=0,
                        end_month=months_to_60,
                        cpi_name=cpi_name))

    # 2. 年金受給 (pension_start_age から)
    p_start_age = lp.pension_start_age
    start_month = max(0, (p_start_age - world.start_age) * 12)

    # 受給額倍率の計算 (65歳基準)
    if p_start_age < 65:
      # 繰り上げ (0.4% / 月 減額)
      reduction_rate = 1.0 - 0.004 * (65 - p_start_age) * 12
    else:
      # 繰り下げ (0.7% / 月 増額)
      reduction_rate = 1.0 + 0.007 * (p_start_age - 65) * 12

    # 厚生年金
    kousei_annual = kousei_unit_annual * (lp.retirement_start_age -
                                          22) * reduction_rate
    if kousei_annual > 0:
      configs.append(
          PensionConfig(name="PensionKousei",
                        amount=kousei_annual / 12.0,
                        start_month=start_month,
                        cpi_name=cpi_name))

    # 基礎年金
    if lp.pension_status == PensionStatus.FULL:
      # 満額受給 (22歳から60歳まで40年納付とみなす)
      kiso_annual = kiso_full_annual * reduction_rate
    elif lp.pension_status == PensionStatus.EXEMPT:
      # 免除期間あり (リタイアから60歳まで免除、受給額 1/2)
      kiso_annual = (
          kiso_full_annual *
          (lp.retirement_start_age - 22) / 40.0 + kiso_full_annual *
          (60 - lp.retirement_start_age) / 40.0 * 0.5) * reduction_rate
    elif lp.pension_status == PensionStatus.UNPAID:
      # 未納 (リタイアから60歳まで未納、受給額 0)
      kiso_annual = (kiso_full_annual *
                     (lp.retirement_start_age - 22) / 40.0) * reduction_rate
    else:
      raise ValueError(f"未知の年金ステータスです: {lp.pension_status}")

    if kiso_annual > 0:
      configs.append(
          PensionConfig(name="PensionKiso",
                        amount=kiso_annual / 12.0,
                        start_month=start_month,
                        cpi_name="Pension_CPI"))

    # 3. 配偶者年金 (二人以上世帯の場合)
    if lp.household_size >= 2:
      # 配偶者も本人と同じ受給開始時期・条件と仮定（no-op 互換性のため）
      kiso_annual_spouse = kiso_full_annual * reduction_rate
      configs.append(
          PensionConfig(name="PensionReceiptSpouseKiso",
                        amount=kiso_annual_spouse / 12.0,
                        start_month=start_month,
                        cpi_name="Pension_CPI"))

  # Side FIRE
  if lp.side_fire_income_monthly > 0:
    configs.append(
        PensionConfig(name="SideFire",
                      amount=lp.side_fire_income_monthly,
                      start_month=0,
                      end_month=lp.side_fire_duration_months,
                      cpi_name=cpi_name))

  # Mortality
  if lp.mortality_gender:
    if lp.mortality_gender == Gender.MALE:
      rates = MALE_MORTALITY_RATES
    elif lp.mortality_gender == Gender.FEMALE:
      rates = FEMALE_MORTALITY_RATES
    else:
      raise ValueError(f"未知の性別です: {lp.mortality_gender}")

    configs.append(
        MortalityConfig(name="Mortality",
                        mortality_rates=rates,
                        initial_age=world.start_age,
                        payout=1000000.0))

  return _CompiledLifeplan(configs=configs, real_cost_curve=real_cost_curve)


def _build_strategy(variant: _ExperimentVariant, cf_map: Dict[str, str],
                    prices: Dict[str, np.ndarray], cashflows: Dict[str,
                                                                   np.ndarray],
                    annual_cost_real: np.ndarray,
                    record_annual_spend: bool) -> Strategy:
  """_ExperimentVariant から Strategy オブジェクトを構築する。"""
  spec = variant.strategy
  world = variant.world

  # 資産配分比率の変換
  ratio_dict: Dict[Union[str, ZeroRiskAsset], float] = {}
  for asset_enum, ratio in spec.initial_asset_ratio:
    if asset_enum == PredefinedZeroRisk.CASH:
      ratio_dict[ZeroRiskAsset("CASH", 0.0)] = ratio
    elif asset_enum == PredefinedZeroRisk.ZERO_RISK_4PCT:
      ratio_dict[ZeroRiskAsset("ZERO_RISK_4PCT", 0.04)] = ratio
    elif isinstance(asset_enum, PredefinedStock):
      ratio_dict[asset_enum.name] = ratio
    else:
      raise ValueError(f"未知の資産タイプです: {asset_enum}")

  # 売却優先順位の変換
  priority = []
  for a in spec.selling_priority:
    if a == PredefinedZeroRisk.CASH:
      priority.append("CASH")
    elif a == PredefinedZeroRisk.ZERO_RISK_4PCT:
      priority.append("ZERO_RISK_4PCT")
    elif isinstance(a, PredefinedStock):
      priority.append(a.name)
    else:
      raise ValueError(f"未知の資産タイプです: {a}")

  # キャッシュフロールールの構築
  rules = []
  # BaseSpend
  rules.append(
      CashflowRule(source_name=cf_map["BaseSpend"],
                   cashflow_type=CashflowType.REGULAR))
  # SideFire
  if "SideFire" in cf_map:
    rules.append(
        CashflowRule(source_name=cf_map["SideFire"],
                     cashflow_type=CashflowType.REGULAR))
  # Mortality
  if "Mortality" in cf_map:
    rules.append(
        CashflowRule(source_name=cf_map["Mortality"],
                     cashflow_type=CashflowType.EXTRAORDINARY))

  # Pension
  for original_name, hashed_name in cf_map.items():
    if original_name.startswith("Pension"):
      rules.append(
          CashflowRule(source_name=hashed_name,
                       cashflow_type=CashflowType.REGULAR))

  # リバランス設定
  rebalance_interval = 0
  dynamic_rebalance_fn = None
  if spec.rebalance:
    if isinstance(spec.rebalance, DynamicV1Rebalance):
      rebalance_interval = spec.rebalance.interval_months

      # 必要に応じて dynamic_rebalance_fn を設定
      # ここでは calculate_optimal_strategy をラップして使用する
      def dr_fn(
          total_net: np.ndarray, cur_ann_spend: np.ndarray, rem_years: float,
          post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        reb = cast(DynamicV1Rebalance, spec.rebalance)
        s_rate = cur_ann_spend / np.maximum(post_tax_net, 1e-10)

        # ゼロリスク資産の利回り決定
        if reb.zero_risk_asset == PredefinedZeroRisk.ZERO_RISK_4PCT:
          zr_yield = 0.04
        elif reb.zero_risk_asset == PredefinedZeroRisk.CASH:
          zr_yield = 0.0
        else:
          raise ValueError(f"リバランスの振り分け先に指定できない資産です: {reb.zero_risk_asset}")

        # calculate_optimal_strategy はインフレ率 0.0177 を前提にチューニングされている
        ratio = calculate_optimal_strategy(s_rate,
                                           rem_years,
                                           base_yield=zr_yield,
                                           tax_rate=world.tax_rate,
                                           inflation_rate=0.0177)

        # 資産名へのマッピング
        return {
            reb.risky_asset.name: ratio,
            reb.zero_risk_asset.name: 1.0 - ratio
        }

      dynamic_rebalance_fn = dr_fn

    elif isinstance(spec.rebalance, SpendAwareDPRebalance):
      rebalance_interval = 12
      reb_dp = cast(SpendAwareDPRebalance, spec.rebalance)
      predictor = DPOptimalStrategyPredictor(reb_dp.model_name)

      def dr_dp_fn(
          total_net: np.ndarray, cur_ann_spend: np.ndarray, rem_years: float,
          post_tax_net: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        reb = cast(SpendAwareDPRebalance, spec.rebalance)
        ratio = calculate_optimal_strategy_dp(total_net=total_net,
                                              cur_ann_spend=cur_ann_spend,
                                              rem_years=rem_years,
                                              post_tax_net=post_tax_net,
                                              dp_predictor=predictor,
                                              initial_age=world.start_age)
        return {
            reb.risky_asset.name: ratio,
            reb.zero_risk_asset.name: 1.0 - ratio
        }

      dynamic_rebalance_fn = dr_dp_fn

  # 動的支出設定
  if spec.spend_adjustment:
    base_spend_rule_name = cf_map["BaseSpend"]
    # ルールのインデックスを見つけて上書き
    for i, rule in enumerate(rules):
      if rule.source_name == base_spend_rule_name:
        handler: CashflowDynamicHandler
        if isinstance(spec.spend_adjustment, DynamicV1Adjustment):
          init_spend = spec.spend_adjustment.initial_annual_spend
          if init_spend is None:
            init_spend = float(annual_cost_real[0])
          handler = DynamicSpending(
              initial_annual_spend=init_spend,
              target_ratio=spec.spend_adjustment.target_ratio,
              upper_limit=spec.spend_adjustment.upper_limit,
              lower_limit=spec.spend_adjustment.lower_limit)
        elif isinstance(spec.spend_adjustment, SpendAwareAdjustment):
          adj = spec.spend_adjustment
          handler = SpendAwareDynamicSpending(
              initial_age=world.start_age,
              p_low=adj.p_low,
              p_high=adj.p_high,
              lower_mult=adj.lower_mult,
              upper_mult=adj.upper_mult,
              annual_cost_real=annual_cost_real.tolist(),
              dp_predictor=DPOptimalStrategyPredictor(adj.model_name))
        else:
          raise ValueError(f"未知の支出調整タイプです: {spec.spend_adjustment}")
        rules[i] = replace(rule, dynamic_handler=handler)
        break

  return Strategy(name=variant.name,
                  initial_money=spec.initial_money,
                  initial_loan=0.0,
                  yearly_loan_interest=0.0,
                  initial_asset_ratio=ratio_dict,
                  selling_priority=priority,
                  tax_rate=world.tax_rate,
                  rebalance_interval=rebalance_interval,
                  dynamic_rebalance_fn=dynamic_rebalance_fn,
                  record_annual_spend=record_annual_spend,
                  initial_prev_net_reg_spend=0.0,
                  initial_prev_gross_reg_spend=0.0,
                  cashflow_rules=rules)
