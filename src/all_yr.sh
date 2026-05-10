#!/bin/bash
#
# このスクリプトは、40年、50年、60年の各シナリオに対して、最適戦略の計算、
# グリッドサーチの実行、および結果の分析を一括で行います。
# 各年齢層（40歳、50歳、60歳リタイア）ごとに、指定された支出倍率のモデルを生成し、
# それらを使用して各種実験（exp_type）を実行します。
#
# 使用方法:
#   ./src/all_yr.sh [function_name]
#   function_name: run_all, execute_all_60yr, execute_all_50yr, execute_all_40yr
#

set -e

# グローバル変数：各年齢層のシナリオと実験タイプ
SCENARIOS_60YR=("re60_pen70_95_m0_75" "re60_pen70_95_m1" "re60_pen70_95_m1_5" "re60_pen70_95_m2" "re60_pen70_95_m3")
EXP_TYPES_60YR=("optimal-pension" "pen70-lifeplan" "pen70-formula" "pen70-ds")

SCENARIOS_50YR=("re50_pen70_95_m0_75" "re50_pen70_95_m1" "re50_pen70_95_m1_2" "re50_pen70_95_m1_5" "re50_pen70_95_m2" "re50_pen70_95_m3")
EXP_TYPES_50YR=("optimal-pension" "pen70-lifeplan" "pen70-formula" "pen70-ds")

SCENARIOS_40YR=("re40_pen65_95_m0_5" "re40_pen65_95_m0_75" "re40_pen65_95_m1" "re40_pen65_95_m1_5" "re40_pen65_95_m2" "re40_pen65_95_m3")
EXP_TYPES_40YR=("optimal-pension" "pen65-lifeplan" "pen65-formula" "pen65-ds")

# 最適戦略データの削除
delete_old_data() {
  local age_group=$1
  local scenarios=()

  case "$age_group" in
    "60yr")
      scenarios=("${SCENARIOS_60YR[@]}")
      ;;
    "50yr")
      scenarios=("${SCENARIOS_50YR[@]}")
      ;;
    "40yr")
      scenarios=("${SCENARIOS_40YR[@]}")
      ;;
    *)
      echo "Unknown age group for deletion: $age_group"
      return 1
      ;;
  esac

  echo "Are you sure you want to delete data files for $age_group under data/optimal_strategy_dp/? (y/n)"
  read -r answer
  if [ "$answer" != "${answer#[Yy]}" ]; then
    for scenario in "${scenarios[@]}"; do
      local file="data/optimal_strategy_dp/${scenario}.json"
      if [ -f "$file" ]; then
        echo "Deleting $file"
        rm "$file"
      fi
    done
  else
    echo "Deletion for $age_group cancelled."
  fi
}

# 60年シナリオの実行
execute_all_60yr() {
  echo "--- Executing all 60yr scenarios ---"
  local skip_grid=false

  for scenario in "${SCENARIOS_60YR[@]}"; do
    if ! python3 src/optimal_strategy_dp_main.py --scenario "$scenario" --n_sim 1000; then
      echo "Warning: optimal_strategy_dp_main.py failed for $scenario. Skipping grid and analysis for 60yr."
      skip_grid=true
    fi
  done

  if [ "$skip_grid" = false ]; then
    for exp_type in "${EXP_TYPES_60YR[@]}"; do
      if python3 src/all_60yr_grid_main.py --exp_type "$exp_type"; then
        python3 src/analyze_all_60yr_grid_main.py --exp_type "$exp_type"
      else
        echo "Warning: all_60yr_grid_main.py failed for $exp_type. Skipping analysis."
      fi
    done
  fi
}

# 50年シナリオの実行
execute_all_50yr() {
  echo "--- Executing all 50yr scenarios ---"
  local skip_grid=false

  for scenario in "${SCENARIOS_50YR[@]}"; do
    if ! python3 src/optimal_strategy_dp_main.py --scenario "$scenario" --n_sim 1000; then
      echo "Warning: optimal_strategy_dp_main.py failed for $scenario. Skipping grid and analysis for 50yr."
      skip_grid=true
    fi
  done

  if [ "$skip_grid" = false ]; then
    for exp_type in "${EXP_TYPES_50YR[@]}"; do
      if python3 src/all_50yr_grid_main.py --exp_type "$exp_type"; then
        python3 src/analyze_all_50yr_grid_main.py --exp_type "$exp_type"
      else
        echo "Warning: all_50yr_grid_main.py failed for $exp_type. Skipping analysis."
      fi
    done
  fi
}

# 40年シナリオの実行
execute_all_40yr() {
  echo "--- Executing all 40yr scenarios ---"
  local skip_grid=false

  for scenario in "${SCENARIOS_40YR[@]}"; do
    if ! python3 src/optimal_strategy_dp_main.py --scenario "$scenario" --n_sim 1000; then
      echo "Warning: optimal_strategy_dp_main.py failed for $scenario. Skipping grid and analysis for 40yr."
      skip_grid=true
    fi
  done

  if [ "$skip_grid" = false ]; then
    for exp_type in "${EXP_TYPES_40YR[@]}"; do
      if python3 src/all_40yr_grid_main.py --exp_type "$exp_type"; then
        python3 src/analyze_all_40yr_grid_main.py --exp_type "$exp_type"
      else
        echo "Warning: all_40yr_grid_main.py failed for $exp_type. Skipping analysis."
      fi
    done
  fi
}

# 全実行
run_all() {
  delete_old_data "60yr"
  delete_old_data "50yr"
  delete_old_data "40yr"
  # Execute all even if one fails
  set +e
  execute_all_60yr
  execute_all_50yr
  execute_all_40yr
  set -e
}

# メイン処理
if [ -z "$1" ]; then
  echo "Usage: $0 [function_name]"
  echo "Example: $0 run_all"
  exit 1
fi

"$@"
