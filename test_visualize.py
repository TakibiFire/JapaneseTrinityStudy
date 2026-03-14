"""
visualize.py のテストコード。
"""

import numpy as np

from core import SimulationResult
from visualize import create_styled_summary


def test_create_styled_summary():
  """
  create_styled_summary の基本的な動作を確認するテスト。
  特定のパーセンタイルや破産確率が正しく計算されているか。
  """
  # ダミーデータの生成 (2戦略、各3パス、120ヶ月=10年)
  n_sim = 3
  months = 120

  # Strategy 1: 全員破産しない。最終的な net_values を [1000, 2000, 3000] とする
  res1 = SimulationResult(
      net_values=np.array([1000.0, 2000.0, 3000.0]),
      sustained_months=np.array([months, months, months])
  )

  # Strategy 2: 1パスが5年(60ヶ月)で破産する。net_values を [-500, 500, 1500] とする
  res2 = SimulationResult(
      net_values=np.array([-500.0, 500.0, 1500.0]),
      sustained_months=np.array([60, months, months])
  )

  results = {
      "Safe Strategy": res1,
      "Risky Strategy": res2
  }

  df, styler = create_styled_summary(
      results,
      quantiles=[0.5],
      bankruptcy_years=[10]
  )

  # DataFrame の内容確認
  assert "Safe Strategy" in df.index
  assert "Risky Strategy" in df.index
  assert "中央値 (普通)" in df.columns
  assert "10年破産確率 (%)" in df.columns

  # 中央値の確認
  # Safe: median(1000, 2000, 3000) = 2000
  # Risky: median(-500, 500, 1500) = 500
  assert df.loc["Safe Strategy", "中央値 (普通)"] == 2000.0
  assert df.loc["Risky Strategy", "中央値 (普通)"] == 500.0

  # 10年(120ヶ月)破産確率の確認
  # Safe: [120, 120, 120] -> 10年未満で破産したものは0
  # Risky: [60, 120, 120] -> 1/3 が10年未満で破産 -> 33.333...%
  assert df.loc["Safe Strategy", "10年破産確率 (%)"] == 0.0
  assert np.isclose(df.loc["Risky Strategy", "10年破産確率 (%)"], 33.3333333)

  # Markdown 出力がエラーなく可能か
  md_text = df.to_markdown()
  assert "Safe Strategy" in md_text
  assert "Risky Strategy" in md_text
