DOCKER_EXEC				= docker compose exec pytorch
DOCKER_RUN				= docker compose run --rm
PYTHON_EXEC				= /opt/conda/bin/python
# === args ===
# input
INPUT_DIR_PATH			= ./tmp/evaluation/20241125-scenario-ArtGallery/input
BENCH_RESULT_FILE_PATH	= ./tmp/evaluation/result_20241125-scenario-ArtGallery.json
# output
OUTPUT_DIR_SIMPLE_CT	= ./tmp/evaluation/20241125-scenario-ArtGallery/output/simple/calculation-time
OUTPUT_DIR_SIMPLE_MU	= ./tmp/evaluation/20241125-scenario-ArtGallery/output/simple/memory-usage
OUTPUT_DIR_GNN_CT		= ./tmp/evaluation/20241125-scenario-ArtGallery/output/gnn/calculation-time
OUTPUT_DIR_GNN_MU		= ./tmp/evaluation/20241125-scenario-ArtGallery/output/gnn/memory-usage
# additional
LAYER_NUM				= 2
MAX_EPOCHS				= 10000


exec-simple-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time simple \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_SIMPLE_CT)
exec-simple-mu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage simple \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_SIMPLE_MU)
exec-gnn-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time gnn \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_GNN_CT) \
		--layer-num $(LAYER_NUM) \
		--max-epochs $(MAX_EPOCHS)
exec-gnn-mu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage gnn \
		--input-dir-path $(INPUT_DIR_PATH) \
		--bench-result-file $(BENCH_RESULT_FILE_PATH) \
		--output-base-dir-path $(OUTPUT_DIR_GNN_MU) \
		--layer-num $(LAYER_NUM) \
		--max-epochs $(MAX_EPOCHS)

exec-prepare-dataset:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py prepare_dataset \
		--input-dir ./tmp/prepare-dataset/input/20241125-000000 \
		--output-dir ./tmp/prepare-dataset/output/20241125_memory-usage_CM \
		--exclude-outliers-by-ratio \
		--target memory-usage \
		--testing-scenario CM

exec-get-lts:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py get_specific_lts_files \
		--lts-input-dir ./tmp/get_specific_lts_files/lts_files \
		--result-input-dir ./tmp/get_specific_lts_files/result_files \
		--output ./tmp/get_specific_lts_files/output
