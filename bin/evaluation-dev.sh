#!/bin/bash

set -eu

PYTHON_EXEC=/opt/conda/bin/python

# ===
# PARAMETERS
# ===
# bench result
BENCH_RESULT_FILE_PATH=./tmp/evaluation/result_20241125.json
MAX_EPOCHS=10000

# ===
# VARIABLES
# ===
DATASETS=(
  "20241125-threshold"
  "20241125-scenario-ArtGallery"
  "20241125-scenario-AT"
  "20241125-scenario-BW"
  "20241125-scenario-CM"
)
LAYERS=(
  "1"
  "2"
  "3"
)


# ===
# BASE COMMANDS
# ===
function exec_simple_ct () {
  docker compose exec pytorch ${PYTHON_EXEC} main.py predict_calculation_time simple \
		--input-dir-path "$1" \
		--bench-result-file ${BENCH_RESULT_FILE_PATH} \
		--output-base-dir-path "$2"
}
function exec_simple_mu () {
  docker compose exec pytorch ${PYTHON_EXEC} main.py predict_memory_usage simple \
		--input-dir-path "$1" \
		--bench-result-file ${BENCH_RESULT_FILE_PATH} \
		--output-base-dir-path "$2"
}
function exec_gnn_ct () {
  docker compose exec pytorch ${PYTHON_EXEC} main.py predict_calculation_time gnn \
		--input-dir-path "$1" \
		--bench-result-file ${BENCH_RESULT_FILE_PATH} \
		--output-base-dir-path "$2" \
		--layer-num "$3" \
		--max-epochs "${MAX_EPOCHS}"
}
function exec_gnn_mu () {
  docker compose exec pytorch ${PYTHON_EXEC} main.py predict_memory_usage gnn \
		--input-dir-path "$1" \
		--bench-result-file ${BENCH_RESULT_FILE_PATH} \
		--output-base-dir-path "$2" \
		--layer-num "$3" \
		--max-epochs "${MAX_EPOCHS}"
}


###
# MAIN
###
echo "start evaluation"
for dataset in "${DATASETS[@]}"; do
  input_dir=./tmp/evaluation/"${dataset}"/input
  output_base_dir=./tmp/evaluation/"${dataset}"/output

  echo "executing dataset: ${dataset}"

  # simple
  echo "running exec_simple_ct"
  exec_simple_ct "${input_dir}" "${output_base_dir}/simple/calculation-time"
  sleep 10
  echo "running exec_simple_mu"
  exec_simple_mu "${input_dir}" "${output_base_dir}/simple/memory-usage"
  sleep 10
  # gnn
  echo "running exec_gnn_ct"
  for layer in "${LAYERS[@]}"; do
    echo "layer: ${layer}"
    exec_gnn_ct "${input_dir}" "${output_base_dir}/gnn/calculation-time" "${layer}"
    sleep 10
  done
  echo "running exec_gnn_mu"
  for layer in "${LAYERS[@]}"; do
    echo "layer: ${layer}"
    exec_gnn_mu "${input_dir}" "${output_base_dir}/gnn/memory-usage" "${layer}"
    sleep 10
  done
done

echo "done"
