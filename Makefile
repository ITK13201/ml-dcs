DOCKER_EXEC	= docker compose exec cuda
PYTHON_EXEC	= /home/ml-dcs/.conda/envs/ml-dcs/bin/python
INPUT_DIR	= ./tmp/20241013-185457/input
OUTPUT_DIR	= ./tmp/20241013-185457/output
MODE		= simple

run-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time -m $(MODE) --input-dir=$(INPUT_DIR) --output-dir=$(OUTPUT_DIR)
run-mmu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage -m $(MODE) --input-dir=$(INPUT_DIR) --output-dir=$(OUTPUT_DIR)
