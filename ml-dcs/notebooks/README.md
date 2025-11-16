# Notebooks ディレクトリ

このディレクトリには、ML-DCSプロジェクトの評価結果を可視化・分析するためのJupyter notebooksが含まれています。

## 目次

1. [最新の評価結果: evaluation_20241220](#最新の評価結果-evaluation_20241220)
2. [ディレクトリ構造](#ディレクトリ構造)
3. [ノートブックの種類と目的](#ノートブックの種類と目的)
   - [1. メイン評価結果 (results/)](#1-メイン評価結果-results)
   - [2. 学習曲線 (learning-curve/)](#2-学習曲線-learning-curve)
   - [3. 訓練時間分析 (training/)](#3-訓練時間分析-training)
   - [4. MTSA結果分析 (mtsa-results/)](#4-mtsa結果分析-mtsa-results)
   - [5. 最大スコア特定 (biggest-mtsa-scores/)](#5-最大スコア特定-biggest-mtsa-scores)
   - [6. 最大規模問題の詳細分析 (biggest-mtsa-results/)](#6-最大規模問題の詳細分析-biggest-mtsa-results)
   - [7. 未知シナリオ統合結果 (unknown-scenario-results/)](#7-未知シナリオ統合結果-unknown-scenario-results)
4. [使用方法](#使用方法)
5. [評価シナリオの詳細](#評価シナリオの詳細)
6. [評価指標の詳細](#評価指標の詳細)
7. [グラフの見方](#グラフの見方)
8. [よく使われるクラスとユーティリティ](#よく使われるクラスとユーティリティ)
9. [トラブルシューティング](#トラブルシューティング)
10. [推奨ワークフロー](#推奨ワークフロー)
11. [注意事項](#注意事項)
12. [データセットの配布と使用方法](#データセットの配布と使用方法)
13. [まとめ](#まとめ)

---

## 最新の評価結果: evaluation_20241220

最新かつ最も包括的な評価結果は`evaluation_20241220/`ディレクトリに格納されています（30ファイル）。

## ディレクトリ構造

```
notebooks/
└── evaluation_20241220/        # 最新評価（2024年12月20日実施）
    ├── results/                # メインの評価結果
    │   ├── calculation-time.ipynb          # 計算時間予測の全体結果
    │   ├── memory-usage.ipynb              # メモリ使用量予測の全体結果
    │   ├── known_scenarios/                # 既知シナリオでの評価
    │   │   ├── calculation-time_AT.ipynb   # ATシナリオ（計算時間）
    │   │   ├── calculation-time_BW.ipynb   # BWシナリオ（計算時間）
    │   │   ├── calculation-time_CM.ipynb   # CMシナリオ（計算時間）
    │   │   ├── calculation-time_HC.ipynb   # HCシナリオ（計算時間）
    │   │   ├── memory-usage_AT.ipynb       # ATシナリオ（メモリ）
    │   │   ├── memory-usage_BW.ipynb       # BWシナリオ（メモリ）
    │   │   ├── memory-usage_CM.ipynb       # CMシナリオ（メモリ）
    │   │   └── memory-usage_HC.ipynb       # HCシナリオ（メモリ）
    │   └── unknown_scenarios/              # 未知シナリオでの評価
    │       ├── calculation-time_AT.ipynb   # ATシナリオ（計算時間）
    │       ├── calculation-time_BW.ipynb   # BWシナリオ（計算時間）
    │       ├── calculation-time_CM.ipynb   # CMシナリオ（計算時間）
    │       ├── calculation-time_HC.ipynb   # HCシナリオ（計算時間）
    │       ├── memory-usage_AT.ipynb       # ATシナリオ（メモリ）
    │       ├── memory-usage_BW.ipynb       # BWシナリオ（メモリ）
    │       ├── memory-usage_CM.ipynb       # CMシナリオ（メモリ）
    │       └── memory-usage_HC.ipynb       # HCシナリオ（メモリ）
    │
    ├── learning-curve/         # GNN学習曲線の可視化
    │   ├── calculation-time.ipynb          # 計算時間予測の学習曲線
    │   └── memory-usage.ipynb              # メモリ使用量予測の学習曲線
    │
    ├── training/               # 訓練時間とエポック数の分析
    │   ├── calculation-time.ipynb          # 計算時間予測モデルの訓練
    │   └── memory-usage.ipynb              # メモリ使用量予測モデルの訓練
    │
    ├── mtsa-results/           # MTSAベンチマーク結果の分析
    │   ├── calculation-time.ipynb          # 計算時間のMTSA結果
    │   └── memory-usage.ipynb              # メモリ使用量のMTSA結果
    │
    ├── biggest-mtsa-scores/    # 最大スコア問題の分析
    │   ├── calculation-time.ipynb          # 計算時間（最大スコア）
    │   └── memory-usage.ipynb              # メモリ使用量（最大スコア）
    │
    ├── biggest-mtsa-results/   # 最大規模問題の評価
    │   ├── calculation-time.ipynb          # 計算時間（最大規模）
    │   └── memory-usage.ipynb              # メモリ使用量（最大規模）
    │
    └── unknown-scenario-results/  # 未知シナリオの統合結果
        ├── calculation-time.ipynb          # 計算時間（未知シナリオ統合）
        └── memory-usage.ipynb              # メモリ使用量（未知シナリオ統合）
```

## ノートブックの種類と目的

### 1. メイン評価結果 (`results/`)

#### 1.1 全体結果: `calculation-time.ipynb` / `memory-usage.ipynb`

**目的**: 全モデルの予測精度を比較し、論文・レポート用の図表を生成

**評価対象モデル**:
- **Simple ML**:
  - RF (Random Forest)
  - GBDT (Gradient Boosting Decision Tree)
  - DT (Decision Tree)
  - LR (Linear Regression)
- **GNN**:
  - L1 (1層GAT)
  - L2 (2層GAT)
  - L3 (3層GAT)

**出力内容**:
1. モデルごとの散布図（実測値 vs 予測値）
2. 評価指標の集計表:
   - MAE (Mean Absolute Error): 平均絶対誤差
   - RMSE (Root Mean Square Error): 二乗平均平方根誤差
   - MAPE (Mean Absolute Percentage Error): 平均絶対パーセント誤差
   - T_pred: 予測時間（ミリ秒）

**使用クラス**:
```python
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet
from ml_dcs.domain.ml_gnn import GNNTestingResult
from ml_dcs.domain.graph import Graph, PredictionAccuracyGraphData
from ml_dcs.internal.graph.graph import GraphUtil
```

**実行例**:
各セルでモデルごとに以下を実行:
1. 結果ファイル（JSON）の読み込み
2. データモデルへの変換
3. グラフデータの生成
4. 散布図の表示
5. 評価指標の集計

最終セルで全モデルの評価指標をDataFrameで一覧表示。

#### 1.2 既知シナリオ評価: `known_scenarios/*_XX.ipynb`

**目的**: 訓練データに含まれるシナリオでの予測精度を詳細に評価

**対象シナリオ**:
- **AT** (Air Traffic): 複数の飛行機を衝突させずに着陸させるシステム
- **BW** (Bidding Workflow): 申請書の審査・受理を行うシステム
- **CM** (Cat and Mouse): 制限エリア内で捕獲対象を捕獲するシステム
- **HC** (Headcount Control): 部屋内の人数に応じて入室制御を行うシステム

**特徴**:
- `SimpleScenarioEvaluator` / `GNNScenarioEvaluator`を使用
- シナリオ固有のスケールでグラフを表示
- 各モデルのシナリオ別予測精度を比較

**出力**:
- シナリオ別の散布図（7モデル分）
- MAE/RMSEの集計表

**パラメータ設定例**:
```python
SCENARIO = "AT"  # または BW, CM, HC
ORDER_OF_MAG = 6  # グラフのスケール（10^6）
LIM = (-0.2 * 10**ORDER_OF_MAG, 3 * 10**ORDER_OF_MAG)
```

#### 1.3 未知シナリオ評価: `unknown_scenarios/*_XX.ipynb`

**目的**: 訓練データに含まれないシナリオでモデルの汎化性能を評価

**重要性**:
実際の運用では未知のシナリオに対する予測が必要となるため、この評価はモデルの実用性を示す重要な指標となります。

**構造**: 既知シナリオと同様だが、データセットが異なる

**分析ポイント**:
- 既知シナリオと比較してMAE/RMSEがどの程度増加するか
- どのモデルが汎化性能に優れているか
- シナリオ間での性能のばらつき

### 2. 学習曲線 (`learning-curve/`)

**目的**: GNNモデルの訓練過程を可視化し、過学習の有無を確認

**内容**:
- X軸: エポック数
- Y軸: MSE損失（Mean Squared Error）
- 2本の曲線:
  - 訓練セット（train）
  - 検証セット（validation）

**分析対象**:
- GNN L1（1層）
- GNN L2（2層）
- GNN L3（3層）

**使用クラス**:
```python
from ml_dcs.domain.ml_gnn import GNNTrainingResult
from ml_dcs.domain.graph import Graph2, LearningCurveGraphData
from ml_dcs.internal.graph.graph import Graph2Util
```

**確認事項**:
- 訓練損失と検証損失の収束
- 過学習の兆候（検証損失が上昇）
- 最適なエポック数の判断

**パラメータ例**:
```python
X_LABEL = "Epoch"
Y_LABEL = "MSE Loss (s$^2$)"  # または (KB$^2$)
Y1_LEGEND = "train"
Y2_LEGEND = "validation"
mag = 3  # スケール係数
y_lim = [-0.1 * 10**mag, 2 * 10**mag]
```

### 3. 訓練時間分析 (`training/`)

**目的**: 各モデルの訓練コストを比較

**出力指標**:
- **T_train**: 訓練時間（秒）
- **Epochs**: 訓練に要したエポック数
  - Simple ML: -1（エポックの概念なし）
  - GNN: 実際のエポック数

**使用クラス**:
```python
from ml_dcs.domain.ml_simple import MLSimpleTrainingResultSet
from ml_dcs.domain.ml_gnn import GNNTrainingResult
```

**分析内容**:
1. 各モデルの訓練時間を計測
2. DataFrameで一覧表示
3. モデル間の訓練コスト比較

**典型的な結果パターン**:
- Simple ML: 数秒〜数分（高速）
- GNN: 数分〜数十分（レイヤー数とエポック数に依存）

### 4. MTSA結果分析 (`mtsa-results/`)

**目的**: MTSAベンチマーク実行結果の内部構造を分析

**分析内容**:
- MTSA実行の3つのステップ:
  1. **1st**: Parallel Composition
  2. **2nd**: Modified Parallel Composition
  3. **3rd**: Safety Game Solving
- 各ステップの時間と状態数・遷移数
- シナリオごとの特性分析

**データ構造**:
```python
from ml_dcs.domain.mtsa import MTSAResult
```

**出力例**:
```
lts                      1st    2nd     3rd     all     S_m    S_g    S_c    D_m    D_g    D_c
ArtGallery(N, 4 room)   31.4   31.3   305.3   367.9  277440  865669  865668  1514240  4777468  2208453
```

- S_m, S_g, S_c: 各ステップの状態数
- D_m, D_g, D_c: 各ステップの遷移数

**分析ポイント**:
- どのステップが最も時間を要するか
- 状態数・遷移数と実行時間の関係
- シナリオ間での計算複雑性の違い

### 5. 最大スコア特定 (`biggest-mtsa-scores/`)

**目的**: データセット内で総計算時間または総メモリ使用量が最大となるデータを特定

**分析内容**:
多数のMTSA実行結果の中から、総計算時間と総メモリ使用量が特に大きいデータを特定します。

**データソース**:
```python
DIR = "tmp/prepare-dataset/output/20241220_calculation-time"  # または memory-usage
```

**処理手順**:
1. `MTSADataUtil`でデータセットを読み込み
2. 計算時間/メモリ使用量でソート（降順）
3. 最大値を持つ問題を特定
4. シナリオ別（HC, AT, BW, CM）に集計

**出力例**:
```
                          lts_name  calculation_time
ArtGallery(N, 4 room)_000036          302335.0
ArtGallery(N, 4 room)_000091          291496.0
CM(3, 3)_000100                       289450.0
```

**用途**:
- `biggest-mtsa-results/`の分析対象を決定
- 大規模問題の特性を把握
- データセット全体の分布を理解

### 6. 最大規模問題の詳細分析 (`biggest-mtsa-results/`)

**目的**: `biggest-mtsa-scores/`で特定された総計算時間・総メモリ使用量が最大のデータについて、MTSA実行結果の詳細を分析

**分析内容**:
総計算時間と総メモリ使用量が特に大きいデータに対して、MTSA実行の内部構造を詳細に分析します。

**mtsa-resultsとの違い**:
- `mtsa-results/`: 全データのMTSA内部ステップを分析
- `biggest-mtsa-scores/`: 最大スコアを持つデータを特定
- `biggest-mtsa-results/`: **特定された最大スコアデータの詳細分析**

**分析対象**:
各シナリオの中で総計算時間・総メモリ使用量が最も大きい問題について：
- MTSA実行の3つのステップ（Parallel Composition, Modified Parallel Composition, Safety Game Solving）の詳細
- 各ステップの実行時間
- 各ステップの状態数・遷移数
- ボトルネックとなっているステップの特定

**重要性**:
大規模問題の特性を理解することで、予測モデルが最も困難なケースでどのように動作するかを評価できます。

### 7. 未知シナリオ統合結果 (`unknown-scenario-results/`)

**目的**: 複数の未知シナリオの結果を統合し、総合的な汎化性能を評価

**特徴**:
- `UnknownScenarioResult`クラスを使用
- 複数シナリオ（HC, AT, BW, CM）の結果を集約
- 訓練結果とテスト結果の両方を統合

**使用クラス**:
```python
from ml_dcs.domain.result import UnknownScenarioResult
```

**処理フロー**:
1. 各シナリオの結果を順次読み込み
2. `update_with_simple_from_testing()` / `update_with_gnn_from_testing()`で集約
3. `update_with_simple_from_training()` / `update_with_gnn_from_training()`で訓練情報を追加
4. 最終的にDataFrameで全指標を一覧表示

**出力指標**:
- Epochs: 平均エポック数（GNNのみ、重み付き平均）
- MAE, RMSE, MAPE: 予測精度指標
- T_train: 訓練時間
- T_pred: 予測時間

**重要性**:
論文やレポートで未知シナリオでの総合性能を示す際に使用する最重要データです。

## 使用方法

### 1. Jupyter Notebookの起動

```bash
# Dockerコンテナ内でJupyter Notebookを起動
docker compose exec pytorch jupyter notebook --ip=0.0.0.0 --port=8000 --no-browser --allow-root
```

ブラウザで `http://localhost:8000` にアクセスしてください。

### 2. ノートブックの実行手順

#### 基本的な流れ

1. **設定セルの確認**:
   最初のセルでパス設定とパラメータを確認
   ```python
   SIMPLE_INPUT_PATH = lambda algorithm: f"tmp/prod-results/.../testing-result_{algorithm}.json"
   GNN_INPUT_PATH_L1 = "tmp/prod-results/.../testing-result.json"
   GNN_INPUT_PATH_L2 = "tmp/prod-results/.../testing-result.json"
   GNN_INPUT_PATH_L3 = "tmp/prod-results/.../testing-result.json"
   ```

2. **順次実行**:
   セルを上から順番に実行（Shift + Enter）

3. **結果の確認**:
   グラフと評価指標の表を確認

#### results/のノートブック実行例

```python
# 1. 設定
from ml_dcs.config import config  # 自動的にPyTorch、matplotlibを初期化

SIMPLE_INPUT_PATH = lambda algorithm: f"tmp/prod-results/results-20241220/20241220_calculation-time/simple/20241218-223051/testing-result_{algorithm}.json"
GNN_INPUT_PATH_L1 = "tmp/prod-results/results-20241220/20241220_calculation-time/gnn-L1/20241218-224116/testing-result.json"
# ... (L2, L3も同様)

X_LABEL = "Actual values (ms)"
Y_LABEL = "Predicted values (ms)"
ORDER_OF_MAG = 5
LIM = (-0.2 * 10**ORDER_OF_MAG, 3 * 10**ORDER_OF_MAG)

results_mae = []
results_rmse = []
results_mape = []
results_t_pred = []

# 2. 各モデルの評価（RF, GBDT, DT, LR, GNN-L1, L2, L3）
# 例: Random Forest
from ml_dcs.internal.graph.graph import GraphUtil
import json
from ml_dcs.domain.ml_simple import MLSimpleTestingResultSet
from ml_dcs.domain.graph import Graph, PredictionAccuracyGraphData

with open(SIMPLE_INPUT_PATH("RF"), "r") as f:
    data_dict = json.load(f)
data_model = MLSimpleTestingResultSet(**data_dict)
data = PredictionAccuracyGraphData.from_ml_simple_class(data_model)
graph = Graph(
    data=data,
    x_label=X_LABEL,
    y_label=Y_LABEL,
    x_lim=LIM,
    y_lim=LIM,
    order_of_mag=ORDER_OF_MAG,
)
util = GraphUtil(graph)
util.show()

results_mae.append(data_model.result_at_best_accuracy.mae)
results_rmse.append(data_model.result_at_best_accuracy.rmse)
results_mape.append(data_model.result_at_best_accuracy.mape)
results_t_pred.append(data_model.result_at_best_accuracy.duration.total_seconds())

# 3. 最終集計
import pandas as pd

pd.DataFrame({
    "MAE": [v / 1000 for v in results_mae],
    "RMSE": [v / 1000 for v in results_rmse],
    "MAPE": results_mape,
    "T_pred": [v * 1000 for v in results_t_pred],
})
```

## 評価シナリオの詳細

### シナリオの概要

DCS（Discrete Controller Synthesis：離散コントローラ合成）の評価には、関連研究で用いられた4つのシナリオを使用しています。各シナリオには以下の2つのパラメータがあります：

- **N**: 環境モデルの数
- **K**: 環境モデルあたりの状態数

これらのパラメータを操作することで、異なる複雑性を持つ環境モデルを生成します。

### 4つの評価シナリオ

| シナリオ | 正式名称 | システムの説明 | パラメータN | パラメータK | データサイズ |
|---------|---------|-------------|-----------|-----------|------------|
| **HC** | Headcount Control | 部屋内の人数に応じて入室制御を行うシステム | 部屋の数 | 部屋内の最大人数 | 530インスタンス |
| **AT** | Air Traffic | 複数の飛行機を衝突させずに着陸させるシステム | 飛行機の数 | 着陸プロセスの数 | 1,958インスタンス |
| **BW** | Bidding Workflow | 申請書の審査・受理を行うシステム | 審査チームの数 | 再申請可能回数 | 48インスタンス |
| **CM** | Cat and Mouse | 制限エリア内で捕獲対象を捕獲するシステム | 捕獲対象の数 | エリアの広さ | 619インスタンス |

### 各シナリオの詳細

#### HC (Headcount Control)

**システム概要**: 部屋内の人数に応じて部屋への入室制御を行うシステム

**パラメータ**:
- **N**: 部屋の数（例: N=2〜5）
- **K**: 部屋内の最大人数

**表記例**:
- `ArtGallery(N, 2 room)`: 2部屋の構成
- `ArtGallery(N, 5 room)`: 5部屋の構成

**特性**:
- 部屋数Nが増えると、状態空間が指数関数的に増加
- 最大規模の問題では状態数が277,000以上に到達

#### AT (Air Traffic)

**システム概要**: 飛行機の着陸プロセスを指示することで、複数の飛行機を衝突させずに着陸させるシステム

**パラメータ**:
- **N**: 飛行機の数（例: N=2〜5）
- **K**: 飛行機の着陸プロセスの数（例: K=2〜10）

**表記例**:
- `AT(2, 2)`: 2機の飛行機、各2つの着陸プロセス
- `AT(5, 10)`: 5機の飛行機、各10の着陸プロセス

**特性**:
- パラメータ(N,K)が大きくなると通信の複雑性が増加
- `AT(5,10)`は最大規模で、状態数が約514,000に到達

#### BW (Bidding Workflow)

**システム概要**: 提出された申請書に対して審査チームで許可された申請書のみを受理するシステム

**パラメータ**:
- **N**: 審査チームの数（例: N=2〜5）
- **K**: 再申請可能回数（例: K=2〜5）

**表記例**:
- `BW(2, 2)`: 2つの審査チーム、再申請2回まで
- `BW(5, 5)`: 5つの審査チーム、再申請5回まで

**特性**:
- バッファサイズが大きくなると管理が複雑化
- `BW(5,5)`は最大規模で、状態数が約161,000に到達

#### CM (Cat and Mouse)

**システム概要**: 捕獲ロボットを操作して制限されたエリア内で動く捕獲対象を捕獲するシステム

**パラメータ**:
- **N**: 捕獲対象の数（例: N=2〜3）
- **K**: エリアの広さ（例: K=2〜4）

**表記例**:
- `CM(2, 2)`: 2つの捕獲対象、エリア広さ2
- `CM(3, 3)`: 3つの捕獲対象、エリア広さ3

**特性**:
- 最も複雑なシナリオで、`CM(3,3)`以上は非常に大規模
- `CM(3,3)`は状態数が約1,362,000に到達
- 計算時間が最も長く、最大で1,759秒（約29分）

### 監視モデルの生成

各シナリオには複数の監視モデル（仕様）が存在します。

**生成方法**:
1. 複数の監視モデルの組み合わせの冪集合を作成
2. その中から最大200個の監視モデルの組をランダムに選択
3. 環境モデルと監視モデルのペアを生成

この方法により、同じパラメータ(N,K)でも異なる監視モデルの組み合わせを持つ複数のインスタンスが生成されます。

**例**:
- `AT(2,2)_000000`: ATシナリオ(N=2,K=2)、監視モデル組み合わせ0番
- `AT(2,2)_000001`: ATシナリオ(N=2,K=2)、監視モデル組み合わせ1番

### 評価データセットの構成

evaluation_20241220で使用されているデータセットの内訳:

| シナリオ | インスタンス数 | パラメータ範囲 | 計算時間範囲 | 状態数範囲 |
|---------|-------------|--------------|------------|-----------|
| **HC** | 530 | N=2〜5部屋 | 0.5秒〜368秒 | 1,104〜865,669 |
| **AT** | 1,958 | N=2〜5, K=2〜10 | 0.3秒〜220秒 | 35〜514,098 |
| **BW** | 48 | N=2〜5, K=2〜5 | 0.2秒〜66秒 | 26〜222,105 |
| **CM** | 619 | N=2〜3, K=2〜4 | 1.1秒〜1,759秒 | 3,925〜1,362,074 |
| **合計** | **3,155** | - | - | - |

### 既知シナリオと未知シナリオ

**既知シナリオ**:
訓練データに含まれるシナリオ。モデルはこれらのシナリオのデータから学習しています。

**未知シナリオ**:
訓練データに含まれないシナリオ。モデルの汎化性能を評価するために使用されます。

例えば、HC, AT, BWで訓練したモデルに対して、CMシナリオでテストすることで、未知のシステムタイプに対する予測精度を評価できます。

## 評価指標の詳細

### 予測精度指標

| 指標 | 計算式 | 単位 | 解釈 |
|-----|-------|------|------|
| **MAE** | Σ\|y_true - y_pred\| / n | ms / KB | 予測誤差の平均。0に近いほど良い。 |
| **RMSE** | √(Σ(y_true - y_pred)² / n) | ms / KB | 大きな誤差を強調。MAEより敏感。 |
| **MAPE** | Σ\|y_true - y_pred\| / y_true / n × 100 | % | 相対誤差の平均。スケール非依存。 |
| **R²** | 1 - SS_res / SS_tot | - | 決定係数。1に近いほど良い。 |

### 性能指標

| 指標 | 説明 | 単位 | 備考 |
|-----|------|------|------|
| **T_train** | 訓練時間 | 秒 | モデル構築のコスト |
| **T_pred** | 予測時間 | 秒/ms | 1予測あたりの平均時間 |
| **Epochs** | 訓練エポック数 | - | GNNのみ。Simple MLは-1 |

## グラフの見方

### 散布図（実測値 vs 予測値）

- **X軸**: 実測値（Actual values）
- **Y軸**: 予測値（Predicted values）
- **対角線（y=x）**: 完全一致ライン
- **理想的な結果**: 全点が対角線上または近傍に分布
- **外れ値**: 対角線から大きく離れた点

**ORDER_OF_MAGの調整**:
```python
ORDER_OF_MAG = 5  # グラフを10^5スケールで表示
LIM = (-0.2 * 10**ORDER_OF_MAG, 3 * 10**ORDER_OF_MAG)
```

### 学習曲線

- **収束の確認**: 両曲線が平坦になっているか
- **過学習の検出**: 検証損失が上昇していないか
- **最適エポック**: 検証損失が最小になるポイント

## よく使われるクラスとユーティリティ

### データモデル（Pydantic）

```python
# Simple MLモデルの結果
from ml_dcs.domain.ml_simple import (
    MLSimpleTestingResultSet,   # テスト結果
    MLSimpleTrainingResultSet,  # 訓練結果
)

# GNNモデルの結果
from ml_dcs.domain.ml_gnn import (
    GNNTestingResult,   # テスト結果
    GNNTrainingResult,  # 訓練結果
)

# 統合結果
from ml_dcs.domain.result import UnknownScenarioResult

# MTSAデータ
from ml_dcs.domain.mtsa import MTSAResult
```

### グラフ可視化

```python
# グラフデータとユーティリティ
from ml_dcs.domain.graph import (
    Graph,                        # 散布図用
    Graph2,                       # 学習曲線用
    PredictionAccuracyGraphData,  # 予測精度グラフデータ
    LearningCurveGraphData,       # 学習曲線グラフデータ
)

from ml_dcs.internal.graph.graph import (
    GraphUtil,   # Graph用ユーティリティ
    Graph2Util,  # Graph2用ユーティリティ
)
```

### シナリオ評価

```python
from ml_dcs.usecases.scenario_evaluator import (
    SimpleScenarioEvaluator,  # Simple MLのシナリオ評価
    GNNScenarioEvaluator,     # GNNのシナリオ評価
)
```

### データ読み込み

```python
from ml_dcs.internal.mtsa.data_utils import MTSADataUtil
```

## トラブルシューティング

### 1. ファイルが見つからない

**エラー**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'tmp/prod-results/...'
```

**解決方法**:
ノートブック冒頭のINPUT_PATHを実際の評価結果ファイルのパスに修正してください。

### 2. グラフが正しく表示されない

**症状**: 全ての点が1点に集中している、または軸が適切でない

**解決方法**:
ORDER_OF_MAGとLIMパラメータを調整:
```python
# データの範囲に応じて調整
ORDER_OF_MAG = 6  # 10^6スケール
LIM = (-0.2 * 10**ORDER_OF_MAG, 4 * 10**ORDER_OF_MAG)
```

### 3. PyTorchデバイスのエラー

**エラー**:
```
RuntimeError: CUDA out of memory
```

**解決方法**:
1. Docker Composeのメモリ制限を増やす
2. または、GPUを使わずCPUで実行（自動切り替え）

### 4. JSON読み込みエラー

**エラー**:
```
json.decoder.JSONDecodeError: Expecting value
```

**解決方法**:
評価結果ファイル（JSON）が正しく生成されているか確認してください。

## 推奨ワークフロー

1. **全体結果の確認**:
   - `results/calculation-time.ipynb`
   - `results/memory-usage.ipynb`

2. **シナリオ別の詳細分析**:
   - `results/known_scenarios/` の各ノートブック
   - `results/unknown_scenarios/` の各ノートブック

3. **学習過程の確認**:
   - `learning-curve/calculation-time.ipynb`
   - `learning-curve/memory-usage.ipynb`

4. **訓練コストの評価**:
   - `training/calculation-time.ipynb`
   - `training/memory-usage.ipynb`

### DCS詳細分析

1. **ベンチマーク結果**: `mtsa-results/`
2. **最大規模問題**: `biggest-mtsa-results/`
3. **最大スコア**: `biggest-mtsa-scores/`

## 注意事項

1. **作業ディレクトリ**: ノートブックは`ml-dcs/`ディレクトリから実行することを前提としています
2. **設定の自動初期化**: `from ml_dcs.config import config`により、PyTorchデバイス、matplotlib、乱数シードが自動的に初期化されます
3. **メモリ使用量**: 大規模なグラフや表を含むセルは実行に時間がかかる場合があります
4. **パスの一貫性**: 全てのノートブックで相対パス（`tmp/prod-results/...`）を使用しています

## データセットの配布と使用方法

### データの保存場所

evaluation_20241220のノートブックを実行するために必要なデータは、圧縮ファイルとして配布され、以下のディレクトリ構造で提供されます：

```
tmp/
├── README.md                     # データセットの説明
├── prod-results/                 # 予測機の精度検証結果と訓練済みモデル
│   └── results-20241220/
│       ├── 20241220_calculation-time/
│       │   ├── simple/
│       │   │   └── 20241218-223051/
│       │   │       ├── testing-result_RF.json
│       │   │       ├── testing-result_GBDT.json
│       │   │       ├── testing-result_DT.json
│       │   │       ├── testing-result_LR.json
│       │   │       ├── training-result_RF.json
│       │   │       ├── training-result_GBDT.json
│       │   │       ├── training-result_DT.json
│       │   │       └── training-result_LR.json
│       │   ├── gnn-L1/
│       │   │   └── 20241218-224116/
│       │   │       ├── testing-result.json
│       │   │       └── training-result.json
│       │   ├── gnn-L2/
│       │   │   └── 20241218-230732/
│       │   │       ├── testing-result.json
│       │   │       └── training-result.json
│       │   └── gnn-L3/
│       │       └── 20241218-232205/
│       │           ├── testing-result.json
│       │           └── training-result.json
│       ├── 20241220_memory-usage/
│       │   └── (同様の構造)
│       ├── 20241220_calculation-time_AT/      # Air Traffic シナリオ
│       ├── 20241220_calculation-time_BW/      # Bidding Workflow シナリオ
│       ├── 20241220_calculation-time_CM/      # Cat and Mouse シナリオ
│       ├── 20241220_calculation-time_ArtGallery/
│       ├── 20241220_memory-usage_AT/
│       ├── 20241220_memory-usage_BW/
│       ├── 20241220_memory-usage_CM/
│       └── 20241220_memory-usage_ArtGallery/
│
├── prepare-dataset/              # 予測機作成用データセット
│   └── output/
│       ├── 20241220_calculation-time/
│       │   ├── training/
│       │   ├── validation/
│       │   └── testing/
│       ├── 20241220_memory-usage/
│       │   ├── training/
│       │   ├── validation/
│       │   └── testing/
│       ├── 20241220_calculation-time_AT/
│       ├── 20241220_calculation-time_BW/
│       ├── 20241220_calculation-time_CM/
│       ├── 20241220_calculation-time_ArtGallery/
│       ├── 20241220_memory-usage_AT/
│       ├── 20241220_memory-usage_BW/
│       ├── 20241220_memory-usage_CM/
│       └── 20241220_memory-usage_ArtGallery/
│
└── mtsa-results/                 # MTSAで実行したDCSの総計算時間・総メモリ量データ
    └── calculation-time/
        └── (MTSA実行結果JSONファイル群)
```

### ディレクトリ構成の説明

配布されるデータセット（`tmp/`ディレクトリ）の構成：

#### 1. `tmp/prod-results/results-20241220/`

**概要**: 予測機の精度検証結果と訓練済み機械学習モデル

**内容**:
- Simple MLモデル（RF, GBDT, DT, LR）の精度検証結果と訓練済みモデル
- GNNモデル（L1, L2, L3）の精度検証結果と訓練済みモデル
- 各結果には以下が含まれる：
  - `testing-result*.json`: テストセット予測結果（MAE, RMSE, MAPEなど）
  - `training-result*.json`: 訓練結果

**サイズ**: 約12MB

**使用ノートブック**:
- `results/calculation-time.ipynb`, `memory-usage.ipynb`
- `results/known_scenarios/`, `unknown_scenarios/`
- `learning-curve/`
- `training/`
- `unknown-scenario-results/`

#### 2. `tmp/prepare-dataset/output/`

**概要**: 予測機作成に使用したデータセット

**内容**:
- 予測機の作成（訓練・検証・テスト）に使用したデータセット
- 訓練データ（training）
- 検証データ（validation）
- テストデータ（testing）
- シナリオ別データセット（全体 + AT, BW, CM, ArtGallery）

**サイズ**: 約1.3GB

**使用ノートブック**:
- `biggest-mtsa-scores/`

#### 3. `tmp/mtsa-results/`

**概要**: MTSAで実行したDCSの総計算時間・総メモリ量データ

**内容**:
- MTSAで実行したDCSの結果の中で総計算時間・総メモリ量を示したデータ
- 各シナリオ（AT, BW, CM, ArtGallery）のMTSA実行結果（JSON）

**サイズ**: 約220KB

**使用ノートブック**:
- `mtsa-results/`
- `biggest-mtsa-results/`

#### 4. `tmp/README.md`

データセットの詳細説明、使用方法、評価シナリオの情報が記載されています。

### データセットの取得と配置

#### ステップ1: データセットのダウンロード

evaluation_20241220のデータセット（圧縮: 約55MB、展開後: 約1.4GB）は圧縮ファイルとして配布されます。以下の方法で取得してください：

**GitHub Releasesから取得**:
```bash
cd ml-dcs
wget https://github.com/ITK13201/ml-dcs/releases/download/datasets-2024-12-20/evaluation_20241220_datasets.tar.gz
```

**または、curlを使用**:
```bash
cd ml-dcs
curl -L -o evaluation_20241220_datasets.tar.gz https://github.com/ITK13201/ml-dcs/releases/download/datasets-2024-12-20/evaluation_20241220_datasets.tar.gz
```

#### ステップ2: データセットの展開

```bash
cd ml-dcs
tar -xzf evaluation_20241220_datasets.tar.gz
```

展開後、`tmp/`ディレクトリが作成されます：
```
ml-dcs/
└── tmp/
    ├── README.md                 # データセットの説明
    ├── prod-results/             # 予測結果
    ├── prepare-dataset/          # 準備済みデータセット
    └── mtsa-results/             # MTSAベンチマーク結果
```

**注意**: 既存の`tmp/`ディレクトリがある場合は、事前にバックアップしてください：
```bash
cd ml-dcs
# 既存のtmpをバックアップ
test -d tmp && mv tmp tmp_backup
# 展開
tar -xzf evaluation_20241220_datasets.tar.gz
```

#### ステップ3: ノートブックの実行

データセット展開後、ノートブックをそのまま実行できます：

```bash
cd ml-dcs
jupyter notebook notebooks/evaluation_20241220/
```

ノートブックは`tmp/`ディレクトリ内のデータを自動的に参照します。

#### データセットの削除

データセットが不要になった場合：

```bash
cd ml-dcs
# tmpディレクトリを削除
rm -rf tmp/

# 元のtmpディレクトリを復元（バックアップがある場合）
test -d tmp_backup && mv tmp_backup tmp

# 圧縮ファイルを削除
rm evaluation_20241220_datasets.tar.gz
```

### データサイズの見積もり

| ディレクトリ | 実測サイズ | ファイル数 | 説明 |
|------------|----------|-----------|------|
| `tmp/prod-results/` | 12 MB | 約160ファイル | 予測機の精度検証結果と訓練済みモデル |
| `tmp/prepare-dataset/` | 1.3 GB | 約31,000ファイル | 予測機作成用データセット（訓練/検証/テスト） |
| `tmp/mtsa-results/` | 220 KB | 4ファイル | MTSA実行結果（総計算時間・総メモリ量） |
| **合計（展開後）** | **約1.4 GB** | **約31,758ファイル** | - |
| **圧縮後（tar.gz）** | **約55 MB** | - | 圧縮率: 約96% |

## まとめ

`evaluation_20241220/`ディレクトリは、ML-DCSプロジェクトの最新かつ最も包括的な評価結果を含んでいます。30個のノートブックは以下のように整理されています:

- **results/**: メイン評価（全体 + 既知/未知シナリオ別）
- **learning-curve/**: GNN訓練過程の可視化
- **training/**: 訓練コストの分析
- **mtsa-results/**: MTSAベンチマーク内部分析
- **biggest-mtsa-scores/**: 最大スコア問題の評価
- **biggest-mtsa-results/**: 最大規模問題の評価
- **unknown-scenario-results/**: 未知シナリオ統合結果
