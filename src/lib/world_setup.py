"""
資産運用・取り崩しシミュレーションのための標準的な「世界」設定を構築するモジュール。
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core import CashflowRule, CashflowType, ZeroRiskAsset
from src.lib.asset_generator import (AssetConfigType, DerivedAsset, ForexAsset,
                                     SlideAdjustedCpiAsset,
                                     YearlyLogNormalArithmetic,
                                     generate_monthly_asset_prices)
from src.lib.cashflow_generator import (CashflowConfig, PensionConfig,
                                        generate_cashflows)
from src.lib.retired_spending import (SpendingType,
                                      get_retired_spending_multipliers)
from src.lib.simulation_defaults import (AcwiModelKey, get_acwi_fat_tail_config,
                                         get_cpi_ar12_config)

# 共有アセット名の定数
ORUKAN_NAME = "オルカン"
ZERO_RISK_NAME = "ゼロリスク資産"
FX_NAME = "USDJPY_0_10.53"
CPI_NAME = "Japan_CPI"
PENSION_CPI_NAME = "Pension_CPI"

# 標準的な定数
DEFAULT_TRUST_FEE = 0.0005775
DEFAULT_ZERO_RISK_YIELD = 0.04
DEFAULT_TAX_RATE = 0.20315
CURRENT_YEAR = 2026
MACRO_ECONOMIC_SLIDE_END_YEAR = 2057


@dataclasses.dataclass(frozen=True)
class WorldSetup:
  """
  シミュレーションの「世界」設定を保持するデータクラス。
  """
  # アセット価格パス (n_sim, n_months + 1)
  monthly_prices: Dict[str, np.ndarray]
  # キャッシュフロー設定のリスト
  cf_configs: List[CashflowConfig]
  # キャッシュフロールールのリスト
  cf_rules: List[CashflowRule]
  # 年齢別の月間支出（名目、円） (years,)
  spending_monthly_values: np.ndarray
  # 無リスク資産オブジェクト
  zr_asset_obj: ZeroRiskAsset
  # シミュレーション年数
  years: int
  # 税率
  tax_rate: float
  # オルカンのアセット名
  ORUKAN_NAME: str
  # 無リスク資産のアセット名
  ZERO_RISK_NAME: str
  # CPIのアセット名
  CPI_NAME: str
  # 年金用CPIのアセット名
  PENSION_CPI_NAME: str
  # 為替のアセット名
  FX_NAME: str


def create_standard_world(
    n_sim: int,
    start_age: int,
    end_age: int,
    retirement_age: int,
    pension_start_age: int,
    seed: int = 42,
    tax_rate: float = DEFAULT_TAX_RATE,
    zero_risk_yield: float = DEFAULT_ZERO_RISK_YIELD,
    trust_fee: float = DEFAULT_TRUST_FEE,
    kousei_annual_base: float = 49.2,
    kiso_annual_base: float = 81.6,
    pension_premium_annual: float = 20.4,
) -> WorldSetup:
  """
  標準的なアセットモデル、支出設定、年金設定を構築し、アセット価格を生成します。

  Args:
    n_sim: シミュレーション試行回数
    start_age: シミュレーション開始年齢
    end_age: シミュレーション終了年齢（この年齢の終わりまで）
    retirement_age: 退職年齢（年金保険料の支払いが終わる年齢）
    pension_start_age: 年金受給開始年齢（60〜75歳）
    seed: 乱数シード
    tax_rate: 税率
    zero_risk_yield: 無リスク資産の利回り（税引前）
    trust_fee: オルカンの信託報酬
    kousei_annual_base: 65歳時点での厚生年金年額（万円）
    kiso_annual_base: 65歳時点での基礎年金年額（万円）
    pension_premium_annual: 国民年金保険料の年額（万円）

  Returns:
    WorldSetup: 世界設定オブジェクト
  """
  years = end_age + 1 - start_age

  # 1. アセット設定 (STRICT ORDER FOR SEED CONSISTENCY)
  # 為替 (USDJPY 0%, 10.53%)
  fx_asset = ForexAsset(name=FX_NAME,
                        dist=YearlyLogNormalArithmetic(mu=0.0, sigma=0.1053))
  # オルカン (共通モデルから取得)
  base_sp500 = get_acwi_fat_tail_config(AcwiModelKey.BASE_SP500_155Y)
  base_acwi = get_acwi_fat_tail_config(AcwiModelKey.BASE_ACWI_APPROX)
  # 投資対象としてのオルカン (為替と信託報酬を適用)
  orukan = DerivedAsset(name=ORUKAN_NAME,
                        base=base_acwi.name,
                        trust_fee=trust_fee,
                        forex=FX_NAME)
  # CPI (共通モデル)
  base_cpi = get_cpi_ar12_config(name=CPI_NAME)
  # 年金用CPI (マクロ経済スライド 0.5% 抑制)
  pension_cpi = SlideAdjustedCpiAsset(
      name=PENSION_CPI_NAME,
      base_cpi=CPI_NAME,
      slide_rate=0.005,
      slide_end_month=(MACRO_ECONOMIC_SLIDE_END_YEAR - CURRENT_YEAR) * 12)

  # アセット生成順序を固定
  asset_configs: List[AssetConfigType] = [
      fx_asset, base_sp500, base_acwi, orukan, base_cpi, pension_cpi
  ]

  monthly_prices = generate_monthly_asset_prices(asset_configs,
                                                 n_paths=n_sim,
                                                 n_months=years * 12,
                                                 seed=seed)

  zr_asset_obj = ZeroRiskAsset(name=ZERO_RISK_NAME, yield_rate=zero_risk_yield)

  # 2. キャッシュフロー設定
  cf_configs: List[CashflowConfig] = []
  cf_rules: List[CashflowRule] = []

  # 国民年金保険料 (開始年齢から retirement_age まで)
  months_to_retirement = max(0, (retirement_age - start_age) * 12)
  if months_to_retirement > 0:
    cf_configs.append(
        PensionConfig(name="Pension_Premium_Kiso",
                      amount=-pension_premium_annual / 12.0,
                      start_month=0,
                      end_month=months_to_retirement,
                      cpi_name=CPI_NAME))
    cf_rules.append(
        CashflowRule(source_name="Pension_Premium_Kiso",
                     cashflow_type=CashflowType.REGULAR))

  # 年金受給額の計算
  if pension_start_age < 65:
    # 繰り上げ (0.4% / 月 減額)
    reduction_rate = 1.0 - 0.004 * (65 - pension_start_age) * 12
  else:
    # 繰り下げ (0.7% / 月 増額)
    reduction_rate = 1.0 + 0.007 * (pension_start_age - 65) * 12

  kousei_annual = kousei_annual_base * reduction_rate
  kiso_annual = kiso_annual_base * reduction_rate

  receipt_start_month = max(0, (pension_start_age - start_age) * 12)

  # 厚生年金 (CPI連動)
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kousei",
                    amount=kousei_annual / 12.0,
                    start_month=receipt_start_month,
                    cpi_name=CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kousei",
                   cashflow_type=CashflowType.REGULAR))

  # 基礎年金 (年金用CPI=マクロスライド連動)
  cf_configs.append(
      PensionConfig(name="Pension_Receipt_Kiso",
                    amount=kiso_annual / 12.0,
                    start_month=receipt_start_month,
                    cpi_name=PENSION_CPI_NAME))
  cf_rules.append(
      CashflowRule(source_name="Pension_Receipt_Kiso",
                   cashflow_type=CashflowType.REGULAR))

  # 3. 支出設定
  # 年齢による月額支出（名目ベースライン、円）の取得
  spending_monthly_values = get_retired_spending_multipliers(
      [SpendingType.CONSUMPTION, SpendingType.NON_CONSUMPTION_EXCLUDE_PENSION],
      start_age=start_age,
      num_years=years,
      normalize=False)

  return WorldSetup(
      monthly_prices=monthly_prices,
      cf_configs=cf_configs,
      cf_rules=cf_rules,
      spending_monthly_values=spending_monthly_values,
      zr_asset_obj=zr_asset_obj,
      years=years,
      tax_rate=tax_rate,
      ORUKAN_NAME=ORUKAN_NAME,
      ZERO_RISK_NAME=ZERO_RISK_NAME,
      CPI_NAME=CPI_NAME,
      PENSION_CPI_NAME=PENSION_CPI_NAME,
      FX_NAME=FX_NAME,
  )
