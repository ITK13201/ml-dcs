#!/bin/bash

# variables
PYTHON_EXEC=/opt/conda/bin/python
INPUT_DIR_PATH=./tmp/evaluation/20241116-000000/input
BENCH_RESULT_FILE_PATH=./tmp/evaluation/result_20241116-000000.json
OUTPUT_DIR_SIMPLE_CT=/tmp/evaluation/20241116-000000/output/simple/calculation-time
OUTPUT_DIR_SIMPLE_MU=/tmp/evaluation/20241116-000000/output/simple/memory-usage
OUTPUT_DIR_GNN_CT=./tmp/evaluation/20241116-000000/output/gnn/calculation-time
OUTPUT_DIR_GNN_MU=./tmp/evaluation/20241116-000000/output/gnn/memory-usage
MAX_EPOCHS=10000

# exec-simple-ct:
${PYTHON_EXEC} main.py predict_calculation_time simple \
  --input-dir-path ${INPUT_DIR_PATH} \
  --bench-result-file ${BENCH_RESULT_FILE_PATH} \
  --output-base-dir-path ${OUTPUT_DIR_SIMPLE_CT}

# exec-simple-mu:
${PYTHON_EXEC} main.py predict_memory_usage simple \
  --input-dir-path ${INPUT_DIR_PATH} \
  --bench-result-file ${BENCH_RESULT_FILE_PATH} \
  --output-base-dir-path ${OUTPUT_DIR_SIMPLE_MU}

# exec-gnn-ct:
${PYTHON_EXEC} main.py predict_calculation_time gnn \
  --input-dir-path ${INPUT_DIR_PATH} \
  --bench-result-file ${BENCH_RESULT_FILE_PATH} \
  --output-base-dir-path ${OUTPUT_DIR_GNN_CT} \
  --max-epochs ${MAX_EPOCHS}


# exec-gnn-mu:
${PYTHON_EXEC} main.py predict_memory_usage gnn \
  --input-dir-path ${INPUT_DIR_PATH} \
  --bench-result-file ${BENCH_RESULT_FILE_PATH} \
  --output-base-dir-path ${OUTPUT_DIR_GNN_MU} \
  --max-epochs ${MAX_EPOCHS}
