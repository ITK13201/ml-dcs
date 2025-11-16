# ML-DCS (Machine Learning for Discrete Controller Synthesis)

離散制御機合成の性能特性（計算時間とメモリ使用量）を機械学習を用いて予測する研究プロジェクトです。MTSA (Modal Transition System Analyser) の結果を利用して、LTS (Labeled Transition System) の構造から性能を予測します。

## 目次

1. [特徴](#特徴)
2. [必要要件](#必要要件)
3. [セットアップ](#セットアップ)
4. [使い方](#使い方)
   - [基本的なコマンド構造](#基本的なコマンド構造)
   - [Simple MLモデルでの予測](#simple-mlモデルでの予測)
   - [GNNモデルでの予測](#gnnモデルでの予測)
   - [データセットの準備](#データセットの準備)
   - [ユーティリティコマンド](#ユーティリティコマンド)
5. [Makefileの使用](#makefileの使用)
6. [プロジェクト構成](#プロジェクト構成)
7. [モデルの詳細](#モデルの詳細)
8. [データフロー](#データフロー)
9. [評価結果の可視化](#評価結果の可視化)
10. [開発者向け情報](#開発者向け情報)

---

## 特徴

- **2種類の予測モデル**:
  - **Simple MLモデル**: 従来の機械学習アルゴリズム（線形回帰、ランダムフォレスト、勾配ブースティング、決定木）
  - **GNNモデル**: グラフニューラルネットワーク（GAT: Graph Attention Networks）を使用し、LTS構造を活用

- **2種類の予測対象**:
  - 計算時間（Calculation Time）
  - メモリ使用量（Memory Usage）

- **GPU対応**: PyTorch + CUDA環境でのGPU高速化をサポート

## 必要要件

- Docker & Docker Compose
- NVIDIA GPU（オプション、GNN訓練時に推奨）
- NVIDIA Container Toolkit（GPU使用時）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd ml-dcs
```

### 2. Docker環境の起動

```bash
docker compose up -d
```

### 3. コンテナへの接続

```bash
docker compose exec pytorch bash
```

## 使い方

### 基本的なコマンド構造

```bash
python main.py <command> <subcommand> [options]
```

### Simple MLモデルでの予測

#### 計算時間の予測

```bash
python main.py predict_calculation_time simple \
  --input-dir-path ./tmp/evaluation/input \
  --bench-result-file ./tmp/evaluation/result.json \
  --output-base-dir-path ./tmp/evaluation/output/simple/calculation-time
```

#### メモリ使用量の予測

```bash
python main.py predict_memory_usage simple \
  --input-dir-path ./tmp/evaluation/input \
  --bench-result-file ./tmp/evaluation/result.json \
  --output-base-dir-path ./tmp/evaluation/output/simple/memory-usage
```

### GNNモデルでの予測

#### 計算時間の予測

```bash
python main.py predict_calculation_time gnn \
  --input-dir-path ./tmp/evaluation/input \
  --bench-result-file ./tmp/evaluation/result.json \
  --output-base-dir-path ./tmp/evaluation/output/gnn/calculation-time \
  --layer-num 2 \
  --max-epochs 10000
```

#### メモリ使用量の予測

```bash
python main.py predict_memory_usage gnn \
  --input-dir-path ./tmp/evaluation/input \
  --bench-result-file ./tmp/evaluation/result.json \
  --output-base-dir-path ./tmp/evaluation/output/gnn/memory-usage \
  --layer-num 2 \
  --max-epochs 10000
```

### データセットの準備

```bash
python main.py prepare_dataset \
  --input-dir ./tmp/prepare-dataset/input \
  --output-dir ./tmp/prepare-dataset/output \
  --exclude-outliers-by-ratio \
  --target memory-usage \
  --testing-scenario CM
```

### ユーティリティコマンド

#### GPU動作確認

```bash
python main.py check_gpu
```

#### 特定のLTSファイルの抽出

```bash
python main.py get_specific_lts_files \
  --lts-input-dir ./tmp/lts_files \
  --result-input-dir ./tmp/result_files \
  --output ./tmp/output
```

## Makefileの使用

プロジェクトルートにある`Makefile`を使用すると、よく使うコマンドを簡単に実行できます。

### Makefileの設定

`Makefile`の上部で以下の変数を編集してください:

```makefile
INPUT_DIR_PATH=./tmp/evaluation/20241125-scenario-ArtGallery/input
BENCH_RESULT_FILE_PATH=./tmp/evaluation/result_20241125-scenario-ArtGallery.json
OUTPUT_DIR_SIMPLE_CT=./tmp/evaluation/20241125-scenario-ArtGallery/output/simple/calculation-time
OUTPUT_DIR_SIMPLE_MU=./tmp/evaluation/20241125-scenario-ArtGallery/output/simple/memory-usage
OUTPUT_DIR_GNN_CT=./tmp/evaluation/20241125-scenario-ArtGallery/output/gnn/calculation-time
OUTPUT_DIR_GNN_MU=./tmp/evaluation/20241125-scenario-ArtGallery/output/gnn/memory-usage
LAYER_NUM=2
MAX_EPOCHS=10000
```

### Makefileコマンド

```bash
make exec-simple-ct        # Simple MLモデルで計算時間を予測
make exec-simple-mu        # Simple MLモデルでメモリ使用量を予測
make exec-gnn-ct          # GNNモデルで計算時間を予測
make exec-gnn-mu          # GNNモデルでメモリ使用量を予測
make exec-prepare-dataset # データセットを準備
make exec-get-lts         # LTSファイルを抽出
```

## プロジェクト構成

```
ml-dcs/
├── main.py                 # エントリーポイント
├── ml_dcs/
│   ├── cmd/               # CLIコマンド実装
│   │   ├── root.py        # コマンドルーター
│   │   ├── predict_calculation_time/  # 計算時間予測
│   │   └── predict_memory_usage/      # メモリ使用量予測
│   ├── domain/            # ドメインモデル
│   │   ├── mtsa.py        # MTSAデータモデル
│   │   ├── dataset.py     # データセット構造
│   │   ├── graph.py       # グラフ表現
│   │   ├── ml_simple.py   # Simple MLデータ構造
│   │   └── ml_gnn.py      # GNNデータ構造
│   ├── internal/          # 内部実装
│   │   ├── ml/            # 機械学習実装
│   │   │   ├── simple.py  # Simple MLアルゴリズム
│   │   │   └── gnn.py     # GNNモデル
│   │   ├── preprocessor/  # データ前処理
│   │   ├── graph/         # グラフ処理
│   │   └── mtsa/          # MTSAデータ処理
│   ├── usecases/          # ユースケース
│   │   ├── simple_evaluator.py    # Simple ML評価
│   │   ├── gnn_evaluator.py       # GNN評価
│   │   └── scenario_evaluator.py  # シナリオ別評価
│   └── config/            # 設定
│       └── config.py      # ログ、PyTorch、matplotlib設定
├── notebooks/             # Jupyter notebooks（結果可視化）
├── tmp/                   # 一時ファイル・評価結果
├── docker/               # Dockerファイル
├── compose.yaml          # Docker Compose設定
└── Makefile             # よく使うコマンドのショートカット
```

## モデルの詳細

### Simple MLモデル

使用可能なアルゴリズム:
- 線形回帰 (Linear Regression)
- ランダムフォレスト (Random Forest)
- 勾配ブースティング決定木 (Gradient Boosting Decision Tree)
- 決定木 (Decision Tree)
- ロジスティック回帰 (Logistic Regression)

### GNNモデル

- **アーキテクチャ**: GAT (Graph Attention Network)
- **レイヤー数**: 1〜4層（設定可能）
- **ノード特徴量**: 5次元
- **エッジ特徴量**: 3次元
- **埋め込み次元**: 32次元
- **回帰ネットワーク**: 全結合層（3層）

## データフロー

1. **入力**: LTSファイルとMTSAベンチマーク結果（JSON）
2. **前処理**:
   - Simple ML: 数値特徴量の抽出（状態数、遷移数など）
   - GNN: LTSをグラフ構造に変換（ノード、エッジ、属性）
3. **訓練**:
   - Simple ML: scikit-learnで回帰モデルを訓練
   - GNN: PyTorch GeometricでGATモデルを訓練
4. **評価**: 予測精度の計算（MAE、MAPE、R²など）
5. **出力**: 評価結果（JSON）と可視化ノートブック

## 評価結果の可視化

`notebooks/`ディレクトリのJupyter notebooksを使用して、評価結果を可視化・分析できます:

- 予測精度の可視化
- モデル間の比較
- シナリオ別の分析（既知シナリオ vs 未知シナリオ）
- 学習曲線の分析

## 開発者向け情報

### Conda環境のエクスポート

```bash
conda env export -n ml-dcs > environment.yaml
```

### デフォルト設定

- **乱数シード**: 42（再現性のため）
- **Pythonパス**: `/opt/conda/bin/python` (コンテナ内)
- **デバイス**: CUDA利用可能時は自動的にGPUを使用
- **ログファイル**: `ml-dcs.log`

### 設定ファイル

`ml-dcs/ml_dcs/config/config.py`で以下を設定:
- ロギング設定
- matplotlib設定
- PyTorchデバイス選択
- 乱数シード初期化
