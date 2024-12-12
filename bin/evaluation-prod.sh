#!/bin/bash

set -eu

PYTHON_EXEC=/opt/conda/bin/python
LOG_FILE_PATH=evaluation.log

# ===
# PARAMETERS
# ===
# bench result
BENCH_RESULT_FILE_PATH=./tmp/evaluation/result_20241125.json
MAX_EPOCHS=10000

# ===
# VARIABLES
# ===
DATASETS_CALCULATION_TIME=(
  "20241125_calculation-time"
  "20241125_calculation-time_ArtGallery"
  "20241125_calculation-time_AT"
  "20241125_calculation-time_BW"
  "20241125_calculation-time_CM"
)
DATASETS_MEMORY_USAGE=(
  "20241125_memory-usage"
  "20241125_memory-usage_ArtGallery"
  "20241125_memory-usage_AT"
  "20241125_memory-usage_BW"
  "20241125_memory-usage_CM"
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
  ${PYTHON_EXEC} main.py predict_calculation_time simple \
    --input-dir-path "$1" \
    --bench-result-file ${BENCH_RESULT_FILE_PATH} \
    --output-base-dir-path "$2"
}
function exec_simple_mu () {
  ${PYTHON_EXEC} main.py predict_memory_usage simple \
    --input-dir-path "$1" \
    --bench-result-file ${BENCH_RESULT_FILE_PATH} \
    --output-base-dir-path "$2"
}
function exec_gnn_ct () {
  ${PYTHON_EXEC} main.py predict_calculation_time gnn \
    --input-dir-path "$1" \
    --bench-result-file ${BENCH_RESULT_FILE_PATH} \
    --output-base-dir-path "$2" \
    --layer-num "$3" \
    --max-epochs "${MAX_EPOCHS}"
}
function exec_gnn_mu () {
  ${PYTHON_EXEC} main.py predict_memory_usage gnn \
    --input-dir-path "$1" \
    --bench-result-file ${BENCH_RESULT_FILE_PATH} \
    --output-base-dir-path "$2" \
    --layer-num "$3" \
    --max-epochs "${MAX_EPOCHS}"å
}


###
# MAIN
###
echo "[$(date -Iseconds)] start evaluation" >> ${LOG_FILE_PATH}

echo "[$(date -Iseconds)] target: calculation-time" >> ${LOG_FILE_PATH}
for dataset in "${DATASETS_CALCULATION_TIME[@]}"; do
  input_dir=./tmp/evaluation/datasets/"${dataset}"
  output_base_dir=./tmp/evaluation/results/"${dataset}"

  echo "[$(date -Iseconds)] executing dataset: ${dataset}" >> ${LOG_FILE_PATH}

  # simple
  echo "[$(date -Iseconds)] running exec_simple_ct" >> ${LOG_FILE_PATH}
  exec_simple_ct "${input_dir}" "${output_base_dir}/simple"
  sleep 10m
  # gnn
  echo "[$(date -Iseconds)] running exec_gnn_ct" >> ${LOG_FILE_PATH}
  for layer in "${LAYERS[@]}"; do
    echo "[$(date -Iseconds)] layer: ${layer}" >> ${LOG_FILE_PATH}
    exec_gnn_ct "${input_dir}" "${output_base_dir}/gnn-L${layer}" "${layer}"
    sleep 10m
  done
done

echo "[$(date -Iseconds)] target: memory-usage" >> ${LOG_FILE_PATH}
for dataset in "${DATASETS_MEMORY_USAGE[@]}"; do
  input_dir=./tmp/evaluation/datasets/"${dataset}"
  output_base_dir=./tmp/evaluation/results/"${dataset}"

  echo "[$(date -Iseconds)] executing dataset: ${dataset}" >> ${LOG_FILE_PATH}

  # simple
  echo "[$(date -Iseconds)] running exec_simple_mu" >> ${LOG_FILE_PATH}
  exec_simple_mu "${input_dir}" "${output_base_dir}/simple"
  sleep 10m
  # gnn
  echo "[$(date -Iseconds)] running exec_gnn_mu" >> ${LOG_FILE_PATH}
  for layer in "${LAYERS[@]}"; do
    echo "[$(date -Iseconds)] layer: ${layer}" >> ${LOG_FILE_PATH}
    exec_gnn_mu "${input_dir}" "${output_base_dir}/gnn-L${layer}" "${layer}"
    sleep 10m
  done
done

echo "[$(date -Iseconds)] done" >> ${LOG_FILE_PATH}
