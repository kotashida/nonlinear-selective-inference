# SHAP Selective Inference Simulation

非線形モデルの SHAP 重要度で特徴量を選択した後、選択的推論によって第一種過誤率（FPR）を評価するシミュレーションです。

## ファイル構成

```text
si_shap/
├── README.md
├── requirements.txt
├── src/
│   └── si_shap/
│       ├── __init__.py
│       └── simulation.py
├── notebooks/
│   └── run_selective_inference_simulation.ipynb
├── docs/
│   └── selective_inference_method.md
└── outputs/
    └── legacy_simulation_output.html
```

- `src/si_shap/simulation.py`：データ生成、SHAP 選択、未知分散の部分 $F$ 検定、AIS、FPR 集計の実装
- `notebooks/run_selective_inference_simulation.ipynb`：シミュレーションの実行、診断、可視化
- `docs/selective_inference_method.md`：仮定、数式、選択事象、選択的 p 値の説明
- `requirements.txt`：検証済み Python 依存関係
- `outputs/legacy_simulation_output.html`：旧実装の出力。現在の AIS 実装の結果ではないため参考用

## 実行方法

リポジトリルートで依存関係をインストールします。

```powershell
python -m pip install -r requirements.txt
```

その後、`notebooks/run_selective_inference_simulation.ipynb` を開いて上から実行します。AIS は候補応答ごとに Random Forest の再学習と SHAP の再計算を行うため、最初は `n_iters` を小さくして動作を確認してください。
