DOCKER_EXEC = docker compose exec anaconda
PYTHON_EXEC = /home/ml-dcs/.conda/envs/ml-dcs/bin/python
INPUT_DIR	= ./tmp/2024-08-11/input
OUTPUT_DIR	= ./tmp/2024-08-11/output

run-ct:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_calculation_time --input-dir=$(INPUT_DIR) --output-dir=$(OUTPUT_DIR)
run-mmu:
	$(DOCKER_EXEC) $(PYTHON_EXEC) main.py predict_memory_usage --input-dir=$(INPUT_DIR) --output-dir=$(OUTPUT_DIR)
