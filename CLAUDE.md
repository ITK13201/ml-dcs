# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-DCS (Machine Learning for Discrete Controller Synthesis) is a research project that applies machine learning techniques to predict the performance characteristics (calculation time and memory usage) of discrete controller synthesis operations using MTSA (Modal Transition System Analyzer) results.

The project implements two types of prediction models:
- **Simple ML models**: Traditional machine learning algorithms (Linear Regression, Random Forest, Gradient Boosting, Decision Tree)
- **GNN models**: Graph Neural Networks using GAT (Graph Attention Networks) to leverage the LTS (Labeled Transition System) structure

## Development Environment

### Docker Setup

The project uses Docker Compose with GPU support for PyTorch:

```bash
# Start the development environment
docker compose up -d

# Execute commands in the container
docker compose exec pytorch /opt/conda/bin/python main.py <command>

# Or use the Makefile shortcuts
make exec-simple-ct
make exec-gnn-ct
```

### Python Environment

- Uses Conda for package management
- Export environment: `conda env export -n ml-dcs > environment.yaml`
- Python executable in container: `/opt/conda/bin/python`

## Common Commands

### Running Predictions

All commands follow the pattern: `python main.py <command> <subcommand> [options]`

#### Simple ML Model Predictions

```bash
# Calculation time prediction
python main.py predict_calculation_time simple \
  --input-dir-path <INPUT_DIR> \
  --bench-result-file <BENCH_RESULT_FILE> \
  --output-base-dir-path <OUTPUT_DIR>

# Memory usage prediction
python main.py predict_memory_usage simple \
  --input-dir-path <INPUT_DIR> \
  --bench-result-file <BENCH_RESULT_FILE> \
  --output-base-dir-path <OUTPUT_DIR>
```

#### GNN Model Predictions

```bash
# Calculation time prediction
python main.py predict_calculation_time gnn \
  --input-dir-path <INPUT_DIR> \
  --bench-result-file <BENCH_RESULT_FILE> \
  --output-base-dir-path <OUTPUT_DIR> \
  --layer-num 2 \
  --max-epochs 10000

# Memory usage prediction
python main.py predict_memory_usage gnn \
  --input-dir-path <INPUT_DIR> \
  --bench-result-file <BENCH_RESULT_FILE> \
  --output-base-dir-path <OUTPUT_DIR> \
  --layer-num 2 \
  --max-epochs 10000
```

### Dataset Preparation

```bash
python main.py prepare_dataset \
  --input-dir <INPUT_DIR> \
  --output-dir <OUTPUT_DIR> \
  --exclude-outliers-by-ratio \
  --target <calculation-time|memory-usage> \
  --testing-scenario <SCENARIO_NAME>
```

### Other Utilities

```bash
# Check GPU availability
python main.py check_gpu

# Get specific LTS files
python main.py get_specific_lts_files \
  --lts-input-dir <LTS_DIR> \
  --result-input-dir <RESULT_DIR> \
  --output <OUTPUT_DIR>
```

### Using Makefile

The Makefile provides shortcuts for common operations. Configure the variables at the top:

```makefile
INPUT_DIR_PATH=./tmp/evaluation/<experiment>/input
BENCH_RESULT_FILE_PATH=./tmp/evaluation/result_<experiment>.json
OUTPUT_DIR_SIMPLE_CT=./tmp/evaluation/<experiment>/output/simple/calculation-time
# ... etc
```

Then run:
```bash
make exec-simple-ct  # Simple model - calculation time
make exec-simple-mu  # Simple model - memory usage
make exec-gnn-ct     # GNN model - calculation time
make exec-gnn-mu     # GNN model - memory usage
make exec-prepare-dataset
make exec-get-lts
```

## Architecture

### Command Structure

The CLI uses a hierarchical command structure implemented in `ml-dcs/ml_dcs/cmd/`:
- `root.py`: Main command router using argparse
- `base.py`: Base command class
- `predict_calculation_time/`: Calculation time prediction commands
  - `root.py`: Parent command
  - `simple.py`: Simple ML implementation
  - `gnn.py`: GNN implementation
- `predict_memory_usage/`: Memory usage prediction commands (same structure)
- `prepare_dataset.py`: Dataset preparation utility
- `get_specific_lts_files.py`: LTS file extraction utility

### Domain Layer

Key domain models in `ml-dcs/ml_dcs/domain/`:
- `mtsa.py`: MTSA result models
- `dataset.py`: Dataset structures (training/validation/testing splits)
- `graph.py`: Graph representations for GNN
- `ml_simple.py`: Data structures for simple ML models
- `ml_gnn.py`: Data structures for GNN models
- `evaluation.py`: Evaluation target enums and results

### ML Implementation

**Simple ML** (`ml-dcs/ml_dcs/internal/ml/simple.py`):
- `MLSimpleDataUtil`: Data preparation and threshold filtering
- `RegressionAlgorithm`: Enum of supported algorithms (Linear Regression, Random Forest, GBDT, Decision Tree, Logistic Regression)
- `RegressionModel`: Unified interface for training and prediction

**GNN** (`ml-dcs/ml_dcs/internal/ml/gnn.py`):
- `LTSGNN`: Graph neural network using GAT (Graph Attention Network) layers
  - Supports 1-4 layer configurations
  - Node features: 5 dimensions
  - Edge features: 3 dimensions
  - Output: 32-dimensional embeddings
- `LTSRegressionModel`: Fully connected network for final prediction
- `GNNDataUtil`: Data preparation and graph construction

### Preprocessing

`ml-dcs/ml_dcs/internal/preprocessor/`:
- `MLSimplePreprocessor`: Feature extraction for simple ML (extracts numerical features from MTSA results)
- `LTSStructurePreprocessor`: Graph structure extraction for GNN

### Use Cases

`ml-dcs/ml_dcs/usecases/`:
- `simple_evaluator.py`: Evaluation pipeline for simple ML models
- `gnn_evaluator.py`: Training and evaluation pipeline for GNN models
- `accuracy_evaluator.py`: Accuracy metrics calculation
- `scenario_evaluator.py`: Scenario-based evaluation (known vs unknown scenarios)

### Configuration

`ml-dcs/ml_dcs/config/config.py`:
- Logging configuration
- Matplotlib settings for plots
- PyTorch device selection (CUDA/CPU)
- Random seed initialization (default: 42)
- Signal directory configuration

## Data Flow

1. **Input**: LTS files and MTSA benchmark results (JSON)
2. **Preprocessing**:
   - Simple ML: Extract numerical features (states, transitions, etc.)
   - GNN: Convert LTS to graph structure (nodes, edges, attributes)
3. **Training**:
   - Simple ML: Train regression models using scikit-learn
   - GNN: Train GAT-based model using PyTorch Geometric
4. **Evaluation**: Calculate prediction accuracy (MAE, MAPE, R², etc.)
5. **Output**: Evaluation results (JSON) and visualization notebooks

## Evaluation Scenarios

The project evaluates models on 4 scenarios with different complexity:

| Scenario | Full Name | Description | Parameter N | Parameter K | Instances |
|----------|-----------|-------------|-------------|-------------|-----------|
| **HC** | Headcount Control | Room occupancy management | # of rooms | Max people/room | 530 |
| **AT** | Air Traffic | Aircraft landing coordination | # of aircraft | # of landing processes | 1,958 |
| **BW** | Bidding Workflow | Application review system | # of review teams | # of resubmissions | 48 |
| **CM** | Cat and Mouse | Target capture in restricted area | # of targets | Area size | 619 |

**Known vs Unknown Scenarios**:
- **Known scenarios**: Included in training data - models learn from these
- **Unknown scenarios**: Not in training data - used to test generalization ability

## MTSA Execution Steps

MTSA (Modal Transition System Analyzer) performs DCS in 3 steps:
1. **Parallel Composition**: Compose environment models
2. **Modified Parallel Composition**: Modify composition with monitoring models
3. **Safety Game Solving**: Solve the safety game to synthesize controller

These steps are analyzed in the `mtsa-results/` notebooks to understand computational bottlenecks.

## Notebooks and Datasets

### Notebooks Structure

The `ml-dcs/notebooks/evaluation_20241220/` directory contains 30 notebooks organized as:
- **results/**: Main evaluation results (overall + known/unknown scenarios)
- **learning-curve/**: GNN training progress visualization
- **training/**: Training cost analysis
- **mtsa-results/**: MTSA benchmark internal analysis
- **biggest-mtsa-scores/**: Identify problems with max calculation time/memory
- **biggest-mtsa-results/**: Detailed analysis of identified max problems
- **unknown-scenario-results/**: Consolidated results for unknown scenarios

See `ml-dcs/notebooks/README.md` for detailed descriptions of each notebook.

### Dataset Distribution

Evaluation datasets are distributed as compressed files (not in Git):
- Download: `https://github.com/ITK13201/ml-dcs/releases/download/datasets-2024-12-20/evaluation_20241220_datasets.tar.gz`
- Size: ~55MB compressed, ~1.4GB extracted
- Extract to `ml-dcs/tmp/` directory

Dataset contains:
- **prod-results/**: Prediction accuracy results and trained models
- **prepare-dataset/**: Training/validation/test datasets used for model creation
- **mtsa-results/**: MTSA execution results (total calculation time and memory)

## Important Notes

- Default random seed is 42 for reproducibility
- GPU is used when available (configured via Docker Compose)
- Results are organized by timestamp and scenario
- Two prediction targets: `calculation-time` and `memory-usage`
- Supports threshold-based outlier filtering
- Dataset is split into training/validation/testing sets
- Notebooks must be run from `ml-dcs/` directory (working directory assumption)
