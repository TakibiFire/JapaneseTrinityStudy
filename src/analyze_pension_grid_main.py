"""
data/pension_grid_comp.csv の結果を可視化するスクリプト。

入力:
- data/pension_grid_comp.csv: 以下のカラムを含むグリッドサーチ結果
  - initial_money, initial_annual_cost: 初期資産と年間支出
  - initial_age: シミュレーション開始時の年齢
  - pension_start_age: 年金受給開始年齢 (60 or 65)
  - initial_pension_nominal: 年金月額 (0, 5.6, 14.4)
  - "1"〜"60": 各経過年数における生存確率

出力:
- docs/imgs/pension/grid_heatmap_age{initial_age}_target{target_age}.svg
  - 開始年齢(30,40,50,60)とターゲット年齢(80,90)の組み合わせ（計8枚）

可視化の構成:
- スライス: initial_age ごとにグラフを作成
- X軸: 年金設定（なし、5.6万繰り上げ、5.6万、14.4万繰り上げ、14.4万）
- Y軸: 資産・支出設定（(5000, 200), (10000, 400), (20000, 800)）
- 値: 指定したターゲット年齢（= ターゲット年齢 - 開始年齢 年後）の生存確率 (%)
"""

import os

import altair as alt
import pandas as pd


def create_heatmap(df: pd.DataFrame, initial_age: int, target_age: int,
                   output_path: str):
  """
  特定の開始年齢とターゲット年齢に対するヒートマップを作成して保存する。
  """
  target_year = target_age - initial_age
  if str(target_year) not in df.columns:
    print(
        f"Warning: Year {target_year} not found for initial_age {initial_age}. Skipping."
    )
    return

  # データの抽出とラベル付け
  plot_df = df[df["initial_age"] == initial_age].copy()

  def get_x_label(row):
    scenario = row['scenario']
    amount_monthly = row['initial_pension_nominal_annual'] / 12.0
    amount_str = f"({amount_monthly:.0f}万)"

    mapping = {
        "NoPensionWorld": "年金制度なし",
        "PayNoReceive": "受給なし",
        "Pay_60": "継続-60歳受給",
        "Pay_65": "継続-65歳受給",
        "Pay_70": "継続-70歳受給",
        "Pay_75": "継続-75歳受給",
        "Exempt_60": "免除-60歳受給",
        "Exempt_65": "免除-65歳受給",
        "Unpaid_65": "未納-65歳受給"
    }
    label = mapping.get(scenario, scenario)
    if scenario not in ["NoPensionWorld", "PayNoReceive"]:
      label += f"\n{amount_str}"
    return label

  def get_y_label(row):
    return f"({int(row['initial_money'])}, {int(row['initial_annual_cost'])})"

  plot_df["scenario_label"] = plot_df.apply(get_x_label, axis=1)
  plot_df["asset_cost"] = plot_df.apply(get_y_label, axis=1)
  plot_df["survival_rate"] = plot_df[str(target_year)]
  plot_df["survival_rate_pct"] = plot_df["survival_rate"] * 100

  # X軸の順序定義 (ラベルが動的なので、scenario でソートした後のラベルのリストを取得する)
  scenario_order = [
      "NoPensionWorld", "PayNoReceive", "Pay_60", "Pay_65", "Pay_70", "Pay_75",
      "Exempt_60", "Exempt_65", "Unpaid_65"
  ]
  x_order = []
  for s in scenario_order:
    matched = plot_df[plot_df["scenario"] == s]["scenario_label"].unique()
    if len(matched) > 0:
      x_order.append(matched[0])
  # Y軸の順序定義 (降順)
  y_order = ["(20000, 800)", "(10000, 400)", "(5000, 200)"]

  base = alt.Chart(plot_df).encode(
      x=alt.X('scenario_label:O',
              title='シナリオ',
              sort=x_order,
              axis=alt.Axis(labelAngle=-45)),
      y=alt.Y('asset_cost:O', title='(初期資産, 年間支出)', sort=y_order),
  )

  # ヒートマップ部分
  heatmap = base.mark_rect(lineBreak=r'\n').encode(
      color=alt.Color('survival_rate:Q',
                      title='生存確率',
                      scale=alt.Scale(scheme='redyellowgreen', domain=[0, 1])))

  # テキスト部分
  text = base.mark_text(baseline='middle').encode(
      text=alt.Text('survival_rate_pct:Q', format='.1f'),
      color=alt.condition(alt.datum.survival_rate > 0.3, alt.value('black'),
                          alt.value('white')))

  chart = (heatmap + text).properties(
      title=f'{initial_age}歳開始 - {target_age}歳時点の生存確率 (%)',
      width=450,
      height=200)

  # STDOUTにヒートマップの値を出力
  print(f"\n--- {initial_age}歳開始 - {target_age}歳時点の生存確率 (%) ---")
  pivot_df = plot_df.pivot(index="asset_cost",
                           columns="scenario_label",
                           values="survival_rate_pct")
  pivot_df = pivot_df.reindex(index=y_order, columns=x_order)
  print(pivot_df.to_string())

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  chart.save(output_path)
  print(f"✅ {output_path} に保存しました。")


def main():
  csv_path = "data/pension/exp1.csv"
  if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    return

  df = pd.read_csv(csv_path)

  initial_ages = [30, 40, 50, 60]
  target_ages = [80, 90]

  for init_age in initial_ages:
    for target_age in target_ages:
      output_path = f"docs/imgs/pension/grid_heatmap_age{init_age}_target{target_age}.svg"
      create_heatmap(df, init_age, target_age, output_path)


if __name__ == "__main__":
  main()
