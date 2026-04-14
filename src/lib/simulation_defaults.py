"""
シミュレーションで使用する共通の資産モデル設定を定義するライブラリ。

このライブラリは、複数のシミュレーションスクリプトで共通して使用される
標準的な資産（オルカン、CPIなど）のモデル定義を集約し、設定の一貫性を保つことを目的とする。
"""

from enum import Enum, auto
from typing import Union

from scipy import stats

from src.lib.asset_generator import (Asset, CpiAsset, DerivedAsset,
                                     MonthlyARLogNormal, MonthlyDist,
                                     MonthlyLogDist)


class AcwiModelKey(Enum):
  """
  ACWI(オール・カントリー・ワールド・インデックス)関連のモデルを識別するためのキー。
  
  株式リターンのシミュレーションにおいて、単純な正規分布ではなく、
  歴史的な暴落（ファットテール）を考慮したモデルを選択するために使用する。
  """
  # S&P500の155年間の月次データ(1871年〜)に基づいたベースモデル。
  # genlogistic分布を用いてフィッティングされており、正規分布よりも裾が重い（暴落確率が高い）。
  # 信託報酬や為替リスクを含まない、純粋な指数の対数リターンモデル。
  BASE_SP500_155Y = auto()

  # BASE_SP500_155Y から派生させた、ACWI(オルカン)の近似モデル。
  # ACWI自体のデータ期間は短いため、より長期のS&P500データと相関させつつ、
  # dweibull分布によるノイズを加えることでACWI特有の特性を再現している。
  # S&P500に比べて期待リターンがわずかに低く、リスク特性が異なる。
  # 信託報酬や為替リスクを含まない。
  BASE_ACWI_APPROX = auto()


def get_acwi_fat_tail_config(key: AcwiModelKey) -> Union[Asset, DerivedAsset]:
  """
  指定されたキーに対応する、ファットテールを考慮したACWI関連のベース資産設定を返す。
  
  この関数が返す資産は「ベース指数」のみであり、為替リスクや信託報酬は含まれていない。
  呼び出し側で DerivedAsset を用いて、用途に応じた為替(USDJPY等)や
  特定の商品の信託報酬を適用して使用することを想定している。
  
  Args:
    key: 取得したいベース資産の識別キー (AcwiModelKey)
    
  Returns:
    Asset or DerivedAsset: 資産設定オブジェクト。
      - BASE_SP500_155Y: genlogistic分布によるS&P500
      - BASE_ACWI_APPROX: S&P500から派生したACWI近似
  """
  # S&P500 155y Model C (genlogistic) のフィッティングパラメータ
  # (shape, loc, scale)
  sp500_155y_params = (0.5983257553837089, 0.024055922548623175,
                       0.017141333060447166)

  # ACWIをS&P500から近似するための回帰パラメータ
  # ACWI = 1.0269 * SP500 - 0.002907 + noise
  acwi_approx_mult = 1.0269
  acwi_approx_intercept = -0.002907
  
  # 近似に使用するノイズ成分の分布 (dweibull)
  # (c, loc, scale)
  acwi_approx_noise_params = (1.2199932203810953, acwi_approx_intercept,
                              0.010652296731100462)

  if key == AcwiModelKey.BASE_SP500_155Y:
    return Asset(
        name="Base_SP500_155y",
        dist=MonthlyLogDist(stats.genlogistic, params=sp500_155y_params),
        trust_fee=0.0)

  if key == AcwiModelKey.BASE_ACWI_APPROX:
    return DerivedAsset(
        name="Base_ACWI_Approx",
        base="Base_SP500_155y",
        multiplier=acwi_approx_mult,
        noise_dist=MonthlyDist(stats.dweibull, params=acwi_approx_noise_params),
        log_correlation=True,
        trust_fee=0.0)

  raise ValueError(f"Unknown key: {key}")


def get_cpi_ar12_config(name: str = "Japan_CPI") -> CpiAsset:
  """
  1970年からの日本の月次CPIデータに基づいた、AR(12)粘着性モデルの設定を返す。
  
  資産取り崩しシミュレーションにおいて、インフレは将来の購買力に直結する。
  このモデルは単なる独立な変動ではなく、過去12ヶ月の変動が現在に影響する
  自己相関（粘着性）をAR(12)モデルで再現しており、物価上昇が続く局面や、
  逆にデフレが続く局面などの「トレンド」を伴う変動をシミュレーションできる。
  
  Args:
    name: 資産名（デフォルト: "Japan_CPI"）

  Returns:
    CpiAsset: CPI（消費者物価指数）モデル。
  """
  # モデルの初期状態として使用する、直近12ヶ月の対数リターン (2025/03〜2026/02)
  # シミュレーションの開始時点を現実の最新データに合わせるために必要。
  initial_y = [
      0.0027039223324009146, 0.0035938942545892623, 0.0026869698208253877,
      -0.0008948546458437107, 0.0017889092427246362, 0.0017857147602345312,
      -0.0008924587830196112, 0.007117467768863955, 0.003539826705123987,
      -0.0017683470567420034, -0.0008853475567242145, -0.00621947806702042
  ]
  # AR(12) モデルの係数 (1970年1月〜直近のデータを基に推計)
  # 物価の強固な自己相関と季節性が反映されている。
  phis_1970 = [
      0.15268125115684014, -0.10485953085717699, 0.04007371599591021,
      0.01877889962124559, 0.12481104840559767, 0.07426982556030279,
      0.10457421059971438, 0.028405474126351145, 0.08655547241690399,
      -0.11318585572419704, 0.09698211923123926, 0.36329524916212186
  ]

  # AR(12)モデルの定義
  # c: 定数項, phis: AR係数, sigma_e: 残差の標準偏差
  dist = MonthlyARLogNormal(c=0.00018532,
                            phis=phis_1970,
                            sigma_e=0.00446792,
                            initial_y=initial_y)

  return CpiAsset(name=name, dist=dist)
