import numpy as np
import pandas as pd

from src.lib.survival_contours import (generate_smooth_contour_data,
                                       get_contour_anchor_points)


def test_get_contour_anchor_points():
  """
    生存確率の補間から正確なアンカーポイントが抽出されるかをテストする。
    """
  # モックデータの作成
  # Spend = 500 で、Rule が 3, 4, 5% のときの生存確率
  data = [
      {
          "initial_annual_cost": 500.0,
          "spending_rule": 3.0,
          "35": 0.99
      },
      {
          "initial_annual_cost": 500.0,
          "spending_rule": 4.0,
          "35": 0.95
      },
      {
          "initial_annual_cost": 500.0,
          "spending_rule": 5.0,
          "35": 0.80
      },
  ]
  df = pd.DataFrame(data)

  target_prob = 0.95
  anchors = get_contour_anchor_points(df, target_prob, "35")

  assert len(anchors) == 1
  rule, spend, m_money = anchors[0]

  # 0.95 に該当する Rule は 4.0 なので、抽出される Rule は 4.0 に近いはず
  assert np.isclose(rule, 4.0)
  assert spend == 500.0
  # M = 500 / (4.0 / 100) = 12500
  assert np.isclose(m_money, 12500.0)


def test_generate_smooth_contour_data():
  """
    アンカーポイントから高密度なデータポイントが生成されるかをテストする。
    """
  # 2つのアンカーポイントを用意
  anchors = [
      (3.0, 500.0, 500.0 / 0.03),  # (Rule, Spend, M)
      (4.0, 1000.0, 1000.0 / 0.04)
  ]

  plot_data = generate_smooth_contour_data(anchors, "95%", num_points=10)

  assert len(plot_data) == 10

  # 最初と最後のポイントがアンカーポイントと一致するか確認
  first_pt = plot_data[0]
  assert first_pt["target_prob"] == "95%"
  assert np.isclose(first_pt["annual_spend_man"], 500.0)
  assert np.isclose(first_pt["spending_rule"], 3.0)

  last_pt = plot_data[-1]
  assert np.isclose(last_pt["annual_spend_man"], 1000.0)
  assert np.isclose(last_pt["spending_rule"], 4.0)

  # 生成されたMが M = Spend / Rule で計算されているか確認
  for pt in plot_data:
    expected_m = pt["annual_spend_man"] / (pt["spending_rule"] / 100.0)
    assert np.isclose(pt["initial_money"], expected_m)
