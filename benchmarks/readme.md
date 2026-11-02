# 概要
- UCI machine learning repositoryから回帰に使えるデータセットをすべてサーベイして良さげなものをピックアップしました

# 含んでいるデータセット
1. Abalone: アワビの年齢を予測する
1. Concrete: コンクリートの強度を予測する
1. Heating Load: 建物の特性から熱負荷を予測する
1. Cooling Load: 建物の特性から冷房負荷を予測する
1. Gas Turbine: タービンの物理特性や排ガス特性から出力を予測する
1. Housing: 住宅価格を予測する
1. Red Wine: 赤ワインの品質を予測する
1. White Wine: 白ワインの品質を予測する

# 前処理
- 各特徴と目的変数はすべて標準化しています
- 離散的な特徴量に対しては小さな正規ノイズを乗せて連続値に変換しています

# 保存形式
- 辞書形式で保存しています
- keyは`["heating_load", "cooling_load", "gas_turbine", "red_wine", "white_wine", "abalone", "concrete", "housing"]`のいずれかです
- valueは`(X, y)`のタプルであり，それぞれのshapeは`(n_samples, n_features), (n_samples,)`です

# 使い方
```python
import pickle

with open("table_dataset.pkl", "rb") as f:
    datasets = pickle.load(f)

X, y = datasets["heating_load"]
print(X.shape, y.shape) # (n_samples, n_features), (n_samples,)
```
