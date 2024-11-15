DOCKER_EXEC				= docker compose exec pytorch
DOCKER_RUN				= docker compose run --rm
PYTHON_EXEC				= /opt/conda/bin/python
# === args ===
# input
INPUT_DIR_PATH			= ./tmp/evaluation/20241031-142305/input
BENCH_RESULT_FILE_PATH	= ./tmp/evaluation/result_20241031-142305.json
# output
OUTPUT_DIR_SIMPLE_CT	= ./tmp/evaluation/20241031-142305/output/simple/calculation-time
OUTPUT_DIR_SIMPLE_MU	= ./tmp/evaluation/20241031-142305/output/simple/memory-usage
OUTPUT_DIR_GNN_CT		= ./tmp/evaluation/20241031-142305/output/gnn/calculation-time
OUTPUT_DIR_GNN_MU		= ./tmp/evaluation/20241031-142305/output/gnn/memory-usage
# additional
MAX_EPOCHS				= 10


exec-simple-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time simple \
		--input-dir=$(INPUT_DIR_PATH) \
		--output-dir=$(OUTPUT_DIR_SIMPLE_CT) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH)
exec-simple-mu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage simple \
		--input-dir=$(INPUT_DIR_PATH) \
		--output-dir=$(OUTPUT_DIR_SIMPLE_MU) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH)
exec-gnn-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time gnn \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_GNN_CT) \
		--max-epochs $(MAX_EPOCHS)
exec-gnn-mu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage gnn \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_GNN_MU) \
		--max-epochs $(MAX_EPOCHS)
