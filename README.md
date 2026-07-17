# SHAP Selective Inference Simulation

非線形モデルの SHAP 重要度で特徴量を選択した後、選択的推論によって第一種過誤率（FPR）を評価するシミュレーションです。

## ファイル構成

```text
si_shap/
├── README.md
├── pyproject.toml
├── src/
│   └── si_shap/
│       ├── __init__.py
│       ├── selection.py
│       ├── inference.py
│       ├── simulation.py
│       └── plotting.py
├── notebooks/
│   ├── naive_vs_random.ipynb
│   └── si_vs_naive_vs_random.ipynb
├── docs/
│   ├── naive_vs_random.md
│   └── si_vs_naive_vs_random.md
└── tests/
    ├── test_selection.py
    ├── test_inference.py
    └── test_simulation.py
```

- `selection.py`：Random Forest、Tree SHAP 重要度、決定的な上位 $k$ 特徴選択
- `inference.py`：スプライン効果基底、カイ統計量、AIS と収束診断
- `simulation.py`：データ生成、3 手法の実行、FPR 集計
- `plotting.py`：p 値ヒストグラムの可視化
- `notebooks/`：シミュレーションの実行例（`naive_vs_random.ipynb` は未知分散 F 検定と既知分散カイ二乗検定の両方に対応）
- `docs/`：各実験の仮定、数式、検定手順の説明
- `tests/`：特徴選択、統計量、集計処理の回帰テスト

## 実行方法

リポジトリルートでパッケージ、Notebook 用依存関係、テスト用依存関係をインストールします。

```powershell
python -m pip install -e ".[notebook,test]"
```

その後、`notebooks/si_vs_naive_vs_random.ipynb` を開いて上から実行します。AIS は候補応答ごとに Random Forest の再学習と SHAP の再計算を行うため、最初は `n_iters` を小さくして動作を確認してください。

テストは次のコマンドで実行できます。

```powershell
python -m pytest
```
