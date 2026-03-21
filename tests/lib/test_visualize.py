"""
visualize.py のテストコード。
"""

import numpy as np
import pandas as pd

from src.core import SimulationResult
from src.lib.visualize import (create_styled_summary,
                               create_survival_probability_chart)


def test_create_styled_summary():
  """
  create_styled_summary の基本的な動作を確認するテスト。
  特定のパーセンタイルや破産確率が正しく計算されているか。
  """
  # ダミーデータの生成 (2戦略、各3パス、120ヶ月=10年)
  n_sim = 3
  months = 120

  # Strategy 1: 全員破産しない。最終的な net_values を [1000, 2000, 3000] とする
  res1 = SimulationResult(net_values=np.array([1000.0, 2000.0, 3000.0]),
                          sustained_months=np.array([months, months, months]))

  # Strategy 2: 1パスが5年(60ヶ月)で破産する。net_values を [-500, 500, 1500] とする
  res2 = SimulationResult(net_values=np.array([-500.0, 500.0, 1500.0]),
                          sustained_months=np.array([60, months, months]))

  results = {"Safe Strategy": res1, "Risky Strategy": res2}

  df, styler = create_styled_summary(results,
                                     quantiles=[0.5],
                                     bankruptcy_years=[10])

  # DataFrame の内容確認
  assert "Safe Strategy" in df.index
  assert "Risky Strategy" in df.index
  assert "中央値 (普通)" in df.columns
  assert "10年破産確率 (%)" in df.columns

  # 中央値の確認
  # Safe: median(1000, 2000, 3000) = 2000
  # Risky: median(-500, 500, 1500) = 500
  assert df.loc["Safe Strategy", "中央値 (普通)"] == "0.2億円"
  assert df.loc["Risky Strategy", "中央値 (普通)"] == "0.1億円"

  # 10年(120ヶ月)破産確率の確認
  # Safe: [120, 120, 120] -> 10年未満で破産したものは0
  # Risky: [60, 120, 120] -> 1/3 が10年未満で破産 -> 33.333...%
  assert df.loc["Safe Strategy", "10年破産確率 (%)"] == "0.0%"
  assert df.loc["Risky Strategy", "10年破産確率 (%)"] == "33.3%"

  # Markdown 出力がエラーなく可能か
  md_text = df.to_markdown()
  assert "Safe Strategy" in md_text
  assert "Risky Strategy" in md_text
  assert "0.2億円" in md_text
  assert "0.1億円" in md_text
  assert "33.3%" in md_text


def test_create_styled_summary_legacy():
  """
  シミュレーション結果の辞書から、各戦略のパーセンタイルや
  破産確率を計算し、Styler オブジェクトとしてフォーマットして返す
  処理が正しく行われるかを検証する（古いテストケース）。
  """
  # モックデータ
  results = {
      "Strategy1":
          SimulationResult(
              net_values=np.array([0.0, 10000.0, 20000.0, 50000.0, 100000.0]),
              sustained_months=np.array([12, 600, 600, 600, 600])  # 1つだけ1年で破産
          ),
      "Strategy2":
          SimulationResult(net_values=np.array(
              [5000.0, 15000.0, 25000.0, 60000.0, 150000.0]),
                           sustained_months=np.array([600, 600, 600, 600, 600]))
  }

  df, styled = create_styled_summary(results, bankruptcy_years=[20])


  # 型チェック
  assert isinstance(styled, pd.io.formats.style.Styler)

  # 破産確率のチェック (Strategy1 は 20年(240ヶ月)時点で破産しているのが1つあるため 20%)
  assert df.loc["Strategy1", "20年破産確率 (%)"] == "20.0%"
  assert df.loc["Strategy2", "20年破産確率 (%)"] == "0.0%"

  # 表示形式の確認
  html = styled.to_html()
  assert "20.0%" in html
  assert "0.0%" in html
  # 10000万円 -> 約 1.0億円
  assert "1.0億円" in html


def test_create_survival_probability_chart():
  """
  create_survival_probability_chart の計算が正しいか、
  特に中間 DataFrame の値を確認するテスト。
  """
  months = 120

  # Safe: 10年(120ヶ月)全て生存
  res1 = SimulationResult(net_values=np.array([1000.0, 2000.0, 3000.0]),
                          sustained_months=np.array([months, months, months]))

  # Risky: 5年(60ヶ月)で1つ破産。つまり5年時点では生存確率 2/3 = 66.6...%
  res2 = SimulationResult(net_values=np.array([-500.0, 500.0, 1500.0]),
                          sustained_months=np.array([60, months, months]))

  results = {"Safe Strategy": res1, "Risky Strategy": res2}

  df, chart = create_survival_probability_chart(results, max_years=10)

  # Safe は 10年時点でも 100% 生存
  safe_y10 = df[(df['Strategy'] == 'Safe Strategy') &
                (df['Year'] == 10)]['Survival Probability (%)'].values[0]
  assert safe_y10 == 100.0

  # Risky は 5年時点ではまだ100% (60ヶ月目で破産とすると、60ヶ月以上は満たす)
  # 実際には 60 >= 60 は True。
  risky_y5 = df[(df['Strategy'] == 'Risky Strategy') &
                (df['Year'] == 5)]['Survival Probability (%)'].values[0]
  assert risky_y5 == 100.0

  # Risky は 6年(72ヶ月)時点では、1つが60なので False になり 2/3 の生存率となる
  risky_y6 = df[(df['Strategy'] == 'Risky Strategy') &
                (df['Year'] == 6)]['Survival Probability (%)'].values[0]
  np.testing.assert_almost_equal(risky_y6, 66.66666666666667)
