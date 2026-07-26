# Random Forest の SHAP 選択変数に対する選択的推論：全 `examples` の数理理論と実装

本書は、`examples` フォルダーにある 4 本の Python スクリプトが実行する統計解析を包括的に説明する技術文書である。対象は、[`examples/run_selective_inference.py`](../examples/run_selective_inference.py) のグローバル帰無仮説シミュレーション、[`examples/plot_selection_regions.py`](../examples/plot_selection_regions.py) の選択領域可視化、[`examples/sweep_selection_region_settings.py`](../examples/sweep_selection_region_settings.py) の設定感度分析、[`examples/compare_selection_event_power.py`](../examples/compare_selection_event_power.py) の対比較された検出力実験である。

数理的な前提、Random Forest と Tree SHAP による選択写像、B スプライン射影検定、選択事象、Adaptive Importance Sampling（AIS）、明示的な選択領域、収束診断、多重性補正、偽陽性率、検出力とその標準誤差を導出し、各説明に対応する Python ファイル名と関数名を明記する。統計学、仮説検定、線形代数、SHAP、選択的推論に十分な予備知識がない読者も追えるように、用語と記号を定義してから数式を導入する。

## 1. 文書の対象と解析目的

本プログラムは、次のデータ解析手順に含まれる選択バイアスを調べる。

1. 同じデータを用いて柔軟な予測モデルを学習する。
2. SHAP 重要度が大きい特徴量を選択する。
3. 選択した特徴量と応答の関連を、再び同じ応答データで検定する。

真の関連が存在しない場合でも、有限個の観測には偶然の相関や非線形パターンが現れる。多数の特徴量の中から最も強く見えるものを選ぶと、偶然に強く見えた特徴量が残りやすい。選択後に通常の帰無分布をそのまま使用すると、選択前には正しかった検定でも、小さい $p$ 値を過剰に生成する。この現象が選択バイアスである。

実行例は次の 3 手法を同じ帰無データ上で比較する。

- **Random**: 応答を参照せず特徴量をランダムに選び、通常のカイ検定を行う。
- **Unadjusted SHAP**: SHAP で特徴量を選ぶが、選択を無視して通常のカイ検定を行う。
- **Selective SHAP (AIS)**: SHAP で選ばれた事実を条件に含め、選択条件付き $p$ 値を AIS で推定する。

データには真の信号を一切含めないため、帰無仮説の棄却はすべて偽陽性となる。シミュレーションを反復し、各手法の偽陽性率が指定した有意水準 $\alpha$ と整合するかを評価する。

**実装対応:** コマンドライン処理とファイル出力は [`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `main`、統計処理全体は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` に実装されている。

## 2. 基本用語と記号

### 2.1 統計学の基本用語

- **標本（sample）**: データの 1 行に相当する 1 観測。本書では標本数を $n$ とする。
- **特徴量（feature）**: 応答を説明または予測する候補変数。特徴量数を $d$ とする。
- **応答（response）**: 予測・検定の対象となる変数。本書ではベクトル $y$ で表す。
- **帰無仮説（null hypothesis）**: 検定で基準とする主張。本書では、特定の特徴量に対応するスプライン効果が存在しないことを表す。
- **検定統計量（test statistic）**: 帰無仮説との不整合の程度を 1 個の数値に要約したもの。本実装では値が大きいほど帰無仮説と整合しにくい。
- **$p$ 値**: 帰無仮説の下で、観測値以上に極端な検定統計量が得られる確率。帰無仮説が正しい確率ではない。
- **有意水準 $\alpha$**: $p<\alpha$ のとき帰無仮説を棄却するための閾値。既定値は $0.05$ である。
- **偽陽性**: 真である帰無仮説を誤って棄却すること。第一種過誤ともいう。
- **条件付け**: 特定の情報を固定した範囲内で確率分布を考えること。
- **i.i.d.**: 各確率変数が互いに独立で、同一の分布に従うこと。

### 2.2 記号

特徴量行列と応答を

$$
X=(X_{ij})\in\mathbb R^{n\times d},
\qquad
y=(y_1,\ldots,y_n)^\top\in\mathbb R^n
$$

とする。$X_{ij}$ は標本 $i$ の特徴量 $j$ の値である。実装と CSV に記録される特徴量 index は Python と同じ 0 始まりである。

$\mathcal N(0,1)$ は平均 0、分散 1 の標準正規分布、$I_n$ は $n\times n$ の単位行列、$\lVert v\rVert_2$ はベクトル $v$ のユークリッド長を表す。

**実装対応:** $n,d,k,\alpha$ などの入力は [`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `parse_args` で定義され、 [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_validate_inputs` で検証される。

## 3. 処理全体の構成

1 回のシミュレーション反復は、次の順序で進む。

1. グローバル帰無仮説の下で $(X,y)$ を生成する。
2. $(X,y)$ から Random Forest と Tree SHAP により上位 $k$ 特徴量を選ぶ。
3. 応答と独立に $k$ 特徴量をランダム選択し、通常のカイ検定を行う。
4. SHAP で選択した各特徴量について、通常のカイ $p$ 値を計算する。
5. 同じ SHAP 選択特徴量について、応答を 1 次元の候補族に分解する。
6. 各候補応答で SHAP 選択を最初から再実行し、選択事象を評価する。
7. AIS により選択条件付き $p$ 値を推定する。
8. 全反復の $p$ 値、偽陽性率、失敗率、Monte Carlo 診断を集計する。

**実装対応:** 上記の 1–8 は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` 本体に対応する。パッケージ直下から `from si_shap import run_simulation` として利用できるよう、 [`src/si_shap/__init__.py`](../src/si_shap/__init__.py) が公開している。

## 4. グローバル帰無仮説下のデータ生成

各反復で、特徴量と応答を次のように生成する。

$$
X_{ij}\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1),
\qquad
y_i\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1),
\qquad
X\perp y.
$$

$X\perp y$ は、特徴量行列と応答が独立であることを表す。生成後の $X$ を固定して考えると、条件付き分布は

$$
y\mid X\sim\mathcal N(0,\sigma^2I_n),
\qquad \sigma=1
$$

である。応答の各成分は平均 0、既知分散 1 で、互いに独立である。

すべての特徴量が応答と無関係なので、この設定をグローバル帰無仮説と呼ぶ。特徴量 $j$ に対する局所的な帰無仮説は、後述する射影行列 $P_j$ を用いて

$$
H_{0,j}:P_j\mu=0,
\qquad
\mu=\mathbb E[y\mid X]
$$

と書ける。シミュレーションでは $\mu=0$ なので、すべての $j$ について $H_{0,j}$ が成立する。

本実装では $\sigma=1$ を既知として使用し、データから分散を推定しない。したがって、後述するカイ分布の導出は、既知分散の球対称正規分布を前提とする。

**実装対応:** データ生成は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_generate_null_dataset`、既知標準偏差 $\sigma=1$ は同ファイルの `SIMULATION_SIGMA` に対応する。実データ API ではこの値を流用せず、利用者が既知の `sigma` を明示する。

## 5. SHAP による特徴選択写像

選択的推論では、「どの特徴量を選んだか」だけでなく、「どのアルゴリズムで選んだか」全体が選択事象を定義する。本実装の選択写像には、応答の丸め、 Random Forest の学習、Tree SHAP、重要度の集約、top-$k$ と同順位処理が含まれる。

### 5.1 応答の丸め

候補応答 $v\in\mathbb R^n$ を森林に入力する前に、各成分を `selection_decimals` 桁へ丸める。

$$
\widetilde v_i
=\operatorname{round}(v_i,m),
\qquad m=\texttt{selection\_decimals}.
$$

既定値は $m=10$ である。丸めは表示上の処理ではなく、選択写像そのものの一部である。観測応答にも AIS が生成するすべての候補応答にも同じ規則を適用するため、浮動小数点の微小差による不安定な再選択を抑える。

**実装対応:** 丸めは [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_tree_shap_importance_for_estimator` にある `np.round` に対応する。Random Forest shortcut の `_tree_shap_importance` と公開 selector の `ShapSelector.select` はいずれもこの関数へ委譲する。

### 5.2 Random Forest

丸めた応答を目的変数として Random Forest 回帰を学習する。$B$ 本の回帰木を $\widehat f_1,\ldots,\widehat f_B$ とすると、森林の予測は概念的に

$$
\widehat f(x)=\frac1B\sum_{b=1}^B\widehat f_b(x)
$$

で表される。既定のパラメータは

$$
\texttt{n\_estimators}=50,
\qquad
\texttt{max\_depth}=5,
\qquad
\texttt{random\_state}=42
$$

である。真の信号がなくても、柔軟な森林は有限標本に現れた偶然のパターンを学習できる。この偶然の適合が、SHAP 選択後のバイアスを発生させる。

固定された `random_state` は、同じ $(X,v)$ から同じ選択結果を得るために重要である。`random_state=None` などへ変更すると、候補応答が同じでも再学習ごとに森林が変化し得る。その場合、後述する `is_selected(z)` を決定的な選択事象とみなす理論がそのままでは成立しない。

**実装対応:** 既定値と上書きの統合は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `RF_PARAMS` と `_resolve_rf_params`、既定森林の構築は `ShapSelector.__init__` または `_tree_shap_importance`、clone と学習は `_tree_shap_importance_for_estimator` に対応する。コマンドラインの `NAME=VALUE` は [`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `_parse_rf_parameter` が JSON 型として解釈できる値を変換する。

### 5.3 Tree SHAP 値

標本 $i$ に対する特徴量 $j$ の SHAP 値を $\phi_{ij}$ とする。Shapley 値の一般的な考え方は、特徴量集合 $D$ の部分集合 $S$ を用いて

$$
\phi_{ij}
=\sum_{S\subseteq D\setminus\{j\}}
\frac{|S|!(d-|S|-1)!}{d!}
\left[v_i(S\cup\{j\})-v_i(S)\right]
$$

と表される。$v_i(S)$ は特徴量集合 $S$ を利用したときのモデル出力である。この式は、特徴量を追加したときの予測変化を、他の特徴量のさまざまな組合せについて平均することを表す。

実装は部分集合を直接列挙せず、決定木用の Tree SHAP を `feature_perturbation="tree_path_dependent"` で計算する。この設定では、 coalition の扱いが学習済み木の経路と標本数に依存する。本書の SHAP 値はこの実装上の定義を指し、因果効果を意味しない。

**実装対応:** `shap.TreeExplainer(...).shap_values(X)` の呼び出しと単一出力・配列形状の検証は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_tree_shap_importance_for_estimator` に対応する。

### 5.4 平均絶対 SHAP 重要度

標本ごとの SHAP 値を、特徴量ごとの大域的重要度へ集約する。

$$
I_j(X,v)
=\frac1n\sum_{i=1}^n
\left|\phi_{ij}(X,\widetilde v)\right|.
$$

絶対値を取ることで、予測を増加させる寄与と減少させる寄与の相殺を防ぐ。 $I_j$ は今回学習したモデルが特徴量 $j$ にどれほど依存したかを表す指標であり、母集団における真の効果や因果効果ではない。

**実装対応:** 上式は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_tree_shap_importance_for_estimator` にある `np.mean(np.abs(values), axis=0)` に対応する。

### 5.5 top-$k$ 選択と同順位規則

$k=$ `k_select` とし、重要度が大きい $k$ 特徴量を

$$
\widehat S_k(X,v)
=\operatorname{TopK}_{j\in\{0,\ldots,d-1\}} I_j(X,v)
$$

として選ぶ。重要度が同じ場合は、特徴量 index が小さいものを優先する。森林の乱数、応答丸め、Tree SHAP の設定、同順位規則を固定することにより、 $\widehat S_k(X,v)$ は入力 $(X,v)$ に対する決定的な写像となる。

選択された特徴量 $j$ の検定では、候補応答における $\widehat S_k$ と観測 top-$k$ の関係から選択事象を定める。既定の `exact_set` は順序を無視した top-$k$ 集合全体の一致、`feature_inclusion` は $j\in\widehat S_k$、`exact_ranking` は順序付き top-$k$ の一致であり、厳密な式と包含関係は 21.2 節で示す。どの mode でも選択特徴量ごとに $P_j$、$a_j$、$u_j$ が異なるため、feature-specific な候補応答経路と検定を作る。

**実装対応:** 降順と同順位処理は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_top_k`、SHAP 計算との合成は `ShapSelector.select` と後方互換 helper `_select_features`、事象判定は `selection_event_holds` に対応する。[`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` が観測 top-$k$ を作り、feature-specific な `is_selected` 内で selector 全体を再実行する。

## 6. 非線形効果を表す B スプライン空間

特徴量 $j$ と応答の関係を直線に限定せず、滑らかな非線形関係も捉えるために 3 次 B スプラインを使用する。

### 6.1 スプライン計画行列

特徴量列 $x_j=X_{:j}$ に Patsy の式

```text
bs(x, df=3, degree=3, include_intercept=False) - 1
```

を適用し、3 列の計画行列

$$
B_j\in\mathbb R^{n\times3}
$$

を作る。各列は $x_j$ の値に対する異なる基底関数を表し、その線形結合によって曲線を表現する。`- 1` は Patsy の式全体から切片を除き、 `include_intercept=False` はスプライン基底自体に定数成分を含めない設定である。

**実装対応:** Patsy の計画行列は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_spline_effect_basis` が `dmatrix` で生成する。

### 6.2 中心化

計画行列の各列から標本平均を引く。中心化行列を

$$
M_0=I_n-\frac1n\mathbf1\mathbf1^\top
$$

とすると、中心化後の行列は

$$
C_j=M_0B_j
$$

である。$\mathbf1$ は全成分が 1 のベクトルである。中心化により

$$
\mathbf1^\top C_j=0
$$

となるため、$C_j$ が張る空間は定数ベクトル、すなわち応答の全体平均と直交する。検定対象は切片ではなく、特徴量に沿って変化するスプライン効果となる。

**実装対応:** 中心化は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_spline_effect_basis` にある `design - design.mean(axis=0, keepdims=True)` に対応する。

### 6.3 SVD と数値ランク

中心化した 3 列が完全に独立とは限らないため、特異値分解（SVD）を行う。

$$
C_j=U_jD_jV_j^\top.
$$

$D_j$ の特異値は、行列に含まれる独立な方向の強さを表す。最大特異値を $d_1$ とし、

$$
\tau
=\max(n,3)\,\epsilon_{\mathrm{machine}}d_1
$$

以下の特異値を数値的なゼロとみなす。$\tau$ を超える特異値の数を数値ランク $r_j$ とする。対応する左特異ベクトルを並べた

$$
Q_j=U_{j,:,1:r_j}
\in\mathbb R^{n\times r_j}
$$

は、中心化スプライン効果空間の正規直交基底である。

$$
Q_j^\top Q_j=I_{r_j},
\qquad
Q_j^\top\mathbf1=0.
$$

通常は $r_j=3$ だが、実装は 3 と固定せず、データから数値ランクを求める。ランク 0 の場合は検定対象となる効果空間が存在しないため停止する。

**実装対応:** SVD、許容誤差、数値ランク、基底の抽出は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_spline_effect_basis` に対応する。

### 6.4 直交射影

$Q_j$ の列空間への直交射影行列は

$$
P_j=Q_jQ_j^\top
$$

である。$P_j$ は対称かつ冪等である。

$$
P_j^\top=P_j,
\qquad
P_j^2=P_j.
$$

実装は $n\times n$ の $P_j$ を明示的に作らず、$Q_j^\top y$ で座標を求め、 $Q_j(Q_j^\top y)$ で射影ベクトルを再構成する。結果は $P_jy$ と同じである。

**実装対応:** 基底は `_spline_effect_basis`、基底による射影は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_chi_statistic` に対応する。

## 7. 既知分散カイ検定

### 7.1 射影座標と検定統計量

応答のスプライン基底上の座標を

$$
c_j=Q_j^\top y
$$

とし、効果空間への射影ベクトルを

$$
p_j=Q_jc_j
=Q_jQ_j^\top y
=P_jy
$$

とする。正規直交基底は長さを保存するので、

$$
\lVert p_j\rVert_2
=\lVert c_j\rVert_2
$$

が成立する。既知標準偏差 $\sigma$ で標準化した検定統計量は

$$
T_j
=\frac{\lVert c_j\rVert_2}{\sigma}
=\frac{\lVert P_jy\rVert_2}{\sigma}
$$

である。$T_j$ が大きいほど、応答の大きな成分が特徴量 $j$ のスプライン効果空間に存在する。

**実装対応:** 座標、射影、統計量は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_chi_statistic` が計算し、`statistic` と `projected` として返す。

### 7.2 帰無分布の導出

$H_{0,j}:P_j\mu=0$ の下では $Q_j^\top\mu=0$ である。$X$ を固定すると、正規分布の線形変換により

$$
Q_j^\top y\mid X
\sim\mathcal N(0,\sigma^2I_{r_j})
$$

となる。共分散は

$$
\begin{aligned}
\operatorname{Var}(Q_j^\top y\mid X)
&=Q_j^\top(\sigma^2I_n)Q_j\\
&=\sigma^2Q_j^\top Q_j\\
&=\sigma^2I_{r_j}
\end{aligned}
$$

である。したがって、$c_j/\sigma$ は $r_j$ 個の独立な標準正規変数からなる。そのベクトルの長さは自由度 $r_j$ のカイ分布に従う。

$$
T_j
=\sqrt{\sum_{h=1}^{r_j}
\left(\frac{c_{jh}}{\sigma}\right)^2}
\sim\chi_{r_j}.
$$

カイ分布は正規ベクトルの長さの分布である。カイ二乗分布とは区別されるが、

$$
T_j^2\sim\chi^2_{r_j}
$$

の関係がある。

特徴量 $j$ が応答と独立に事前指定されている場合、通常の上側 $p$ 値は

$$
p_j^{\mathrm{ordinary}}
=\Pr(\chi_{r_j}\ge T_j)
=1-F_{\chi_{r_j}}(T_j)
$$

である。SciPy の survival function `sf` はこの上側確率を直接計算し、 `1 - cdf` より数値的に安定である。

**実装対応:** Random baseline の `stats.chi.sf(statistic, df=rank)` は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation`、Unadjusted SHAP の同じ survival function は [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` にある。統計量と自由度は両方とも [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_chi_statistic` と `_spline_effect_basis` から得られる。

## 8. 3 種類の推論方法

### 8.1 Random

Random は、$d$ 特徴量から重複なしで $k$ 個を一様に選ぶ。選択は $y$ を参照しないため、選ばれた index で条件付けても検定統計量の帰無分布は変化しない。したがって、通常のカイ $p$ 値をそのまま使用できる。

Random は優れた特徴選択法として導入されているのではなく、データ生成とカイ検定自体が正しく校正されているかを確認する基準である。

**実装対応:** ランダム選択は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` にある `random_selection_rng.choice(..., replace=False)`、検定は同関数内の `_spline_effect_basis`、`_chi_statistic`、`stats.chi.sf` に対応する。

### 8.2 Unadjusted SHAP

Unadjusted SHAP も

$$
p_j^{\mathrm{unadjusted}}
=\Pr(\chi_{r_j}\ge T_j)
$$

を使用する。ただし、特徴量 $j$ は同じ応答 $y$ を用いた SHAP 重要度によって選ばれている。この式は $j$ を $y$ を見る前に固定した場合には正しいが、選択後に必要な分布は無条件の $\chi_{r_j}$ ではなく、

$$
T_j\mid\{j\in\widehat S_k(X,y)\}
$$

の分布である。

SHAP 選択は、有限標本の偶然のパターンに適合した特徴量を残しやすい。そのため選択後の $T_j$ は無条件の場合より大きくなりやすく、選択を無視した $p$ 値は小さくなりやすい。

**実装対応:** [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` が観測応答で selector を実行し、選択特徴量ごとに `stats.chi.sf(t_obs, df=test_rank)` を `unadjusted_p_value` として保存する。[`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` はその列を `unadjusted_iteration` として集計する。この計算に選択条件がないことが Unadjusted SHAP の定義である。

### 8.3 Selective SHAP (AIS)

Selective SHAP は、特徴量 $j$ が SHAP top-$k$ に含まれたという事実を帰無分布へ反映する。高次元の応答 $y\in\mathbb R^n$ を直接積分せず、観測された情報の一部に条件付けて 1 個の非負スカラーだけを変化させる。選択条件付き上側確率を AIS で数値積分する。

**実装対応:** 1 次元化は [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` 内の `orthogonal`、`direction`、`is_selected`、AIS は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` に対応する。明示的な領域計算では [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` が同じ分解を使う。

## 9. 選択的推論の 1 次元化

### 9.1 応答の直交分解

特徴量 $j$ の観測射影を $p_j=P_jy$ とし、効果空間に直交する成分を

$$
a_j=y-p_j=(I-P_j)y
$$

とする。$\lVert p_j\rVert_2>0$ のとき、効果空間内の観測方向を

$$
u_j=\frac{p_j}{\lVert p_j\rVert_2}
$$

と定義する。$u_j$ は単位ベクトルなので $\lVert u_j\rVert_2=1$ である。 $T_j=\lVert p_j\rVert_2/\sigma$ より、観測応答は

$$
y=a_j+p_j
=a_j+\sigma u_jT_j
$$

と分解できる。この分解は、応答を次の 3 要素へ分ける。

- 効果空間に直交する成分 $a_j$
- 効果空間内の方向 $u_j$
- 方向に沿った非負の大きさ $T_j$

**実装対応:** [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` と [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` にある `orthogonal = y - projected`、`projected_norm`、`direction = projected / projected_norm` がそれぞれ $a_j$、$\lVert p_j\rVert_2$、$u_j$ に対応する。`zero_tolerance` 以下では方向を安定に定義できないため `FloatingPointError` を送出する。

### 9.2 候補応答族

$X,a_j,u_j$ を観測値に固定し、大きさだけを $z\ge0$ に置き換える。

$$
y_j(z)=a_j+\sigma u_jz.
$$

$z=T_j$ では、厳密な算術の下で観測応答を再構成する。

$$
\begin{aligned}
y_j(T_j)
&=a_j+\sigma u_jT_j\\
&=y-P_jy+P_jy\\
&=y.
\end{aligned}
$$

各 $z$ について、応答丸め、森林学習、Tree SHAP、平均絶対重要度、top-$k$ 選択をすべて再実行する。観測された順序付き top-$k$ を $M_{\mathrm{obs}}$ とする。既定の選択状態を

$$
s_j(z)
=\mathbf1\{\operatorname{set}(\widehat S_k(X,y_j(z)))
=\operatorname{set}(M_{\mathrm{obs}})\}
$$

と定義する。$\mathbf1\{A\}$ は事象 $A$ が真なら 1、偽なら 0 を返す指示関数である。1 次元の選択領域は

$$
\mathcal Z_j
=\{z\ge0:s_j(z)=1\}
$$

である。AIS では $\mathcal Z_j$ を区間の和として明示的に求める必要はなく、標本化した $z$ ごとに $s_j(z)$ を評価すればよい。

**実装対応:** 公開 API の `is_selected` closure が `candidate` を構成し、 [`src/si_shap/selection.py`](../src/si_shap/selection.py) の selector を介して全パイプラインを再実行する。`selection_event_holds` は API、シミュレーション、可視化で共通の事象定義を提供する。`is_selected(statistic)` は観測点で選択事象全体が再現されることを検証する。

### 9.3 条件付け後のカイ分布

帰無仮説の下で、標準化された効果空間座標 $Q_j^\top y/\sigma$ は球対称な標準多変量正規分布に従う。球対称正規ベクトルは長さと方向に分解でき、次の性質を持つ。

- 長さは $\chi_{r_j}$ に従う。
- 方向は単位球面上で一様である。
- 長さと方向は独立である。

さらに、正規分布では直交する射影の共分散が 0 なら独立である。 $P_jy$ と $(I-P_j)y$ は直交射影なので、効果空間内の成分と $a_j$ は独立である。したがって、$X,a_j,u_j$ で条件付けた後に残る確率変数は

$$
Z\sim\chi_{r_j}
$$

だけである。

さらに選択事象で条件付けると、$Z$ は $\mathcal Z_j$ 内に制限される。条件付き密度は

$$
f_{Z\mid\mathcal E_j}(z)
=\frac{
f_{\chi_{r_j}}(z)s_j(z)
}{
\int_0^\infty f_{\chi_{r_j}}(v)s_j(v)\,dv
},
\qquad z\ge0
$$

となる。分母は、この 1 次元 slice 上で特徴量 $j$ が選択される確率である。

**実装対応:** この分布を記号的に導出する単独関数は存在しない。導出は、[`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_chi_statistic`、[`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` にある `orthogonal` と `direction`、`_run_ais` 内の `stats.chi.logpdf` を組み合わせる理論的根拠である。

### 9.4 選択条件付き $p$ 値

観測統計量 $T_j$ に対する上側の選択条件付き $p$ 値は

$$
\begin{aligned}
p_j^{\mathrm{selective}}
&=\Pr\left(
Z\ge T_j
\mid s_j(Z)=1,X,a_j,u_j
\right)\\
&=
\frac{
\int_{T_j}^{\infty}
s_j(z)f_{\chi_{r_j}}(z)\,dz
}{
\int_0^{\infty}
s_j(z)f_{\chi_{r_j}}(z)\,dz
}.
\end{aligned}
$$

分子は「選択され、かつ観測値以上に大きい」確率、分母は「選択される」確率である。選択規則が大きな $z$ だけを許す場合、無条件には大きい $T_j$ でも、選択されたという条件の下では珍しくない可能性がある。この基準集団の変化を反映するのが選択的 $p$ 値である。

**実装対応:** 上式の比は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` が推定する。`tail = all_z >= t_obs` が $Z\ge T_j$ に対応し、 `is_selected` が真の標本だけが分子と分母に寄与する。

## 10. Importance Sampling

### 10.1 基本恒等式

$Z\sim\chi_{r_j}$ から直接標本化し、選択された上側標本を数える単純な Monte Carlo 法も考えられる。しかし、選択領域のカイ確率が小さい場合は、ほとんどの標本で $s_j(Z)=0$ となる。選択された上側確率がさらに小さい場合、実用的な精度を得るために非常に多くの標本が必要となる。

Importance Sampling は、必要な領域を訪れやすい提案密度 $q$ から $z$ を生成し、

$$
w(z)=\frac{f_{\chi_{r_j}}(z)}{q(z)}
$$

で分布の変更を補正する。任意の可積分関数 $h$ について

$$
\begin{aligned}
\mathbb E_{\chi}[h(Z)]
&=\int h(z)f_\chi(z)\,dz\\
&=\int h(z)\frac{f_\chi(z)}{q(z)}q(z)\,dz\\
&=\mathbb E_q[h(Z)w(Z)]
\end{aligned}
$$

が成立する。$h(z)=s_j(z)$ と $h(z)=s_j(z)\mathbf1\{z\ge T_j\}$ を用いると、選択的 $p$ 値の分母と分子を提案分布から推定できる。

**実装対応:** log importance weight は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_adapt_proposal` と `_run_ais` で `stats.chi.logpdf(...) - proposal_logpdf` として計算される。小さい密度による underflow を抑えるため、通常の密度ではなく log-density を用いる。

### 10.2 0 以上に切断した正規分布

カイ統計量は非負なので、AIS の正規 proposal も $[0,\infty)$ に制限する。平均 $\mu$、標準偏差 $s>0$ の正規分布を 0 以上に切断した密度は

$$
q_{\mathrm{TN}}(z;\mu,s)
=\frac{
\phi((z-\mu)/s)/s
}{
\Phi(\mu/s)
},
\qquad z\ge0
$$

である。$\phi$ は標準正規密度、$\Phi$ は標準正規 CDF である。 $\Phi(\mu/s)=\Pr(N(\mu,s^2)\ge0)$ が、残った半直線上で密度の積分を 1 に戻す正規化定数となる。

**実装対応:** 密度評価は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_truncated_normal_logpdf`、乱数生成は同ファイルの `_sample_truncated_normal` に対応する。

## 11. Adaptive Importance Sampling

### 11.1 pilot proposal の初期値

pilot proposal は観測統計量の近くから開始する。

$$
\mu_0=T_j,
\qquad
s_0=\max(1,0.25T_j+0.5).
$$

`pilot_iters=0` の場合は適応を行わず、この値を最終 mixture の adapted 成分に使用する。

**実装対応:** 初期平均と標準偏差は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_adapt_proposal` 冒頭で定義される。

### 11.2 pilot 標本による適応

各 pilot 反復で、$q_{\mathrm{TN}}(\cdot;\mu_t,s_t)$ から `pilot_samples` 個を生成し、それぞれの選択状態を再評価する。選択された標本に対して

$$
w_h
=\frac{
f_{\chi_{r_j}}(z_h)
}{
q_{\mathrm{TN}}(z_h;\mu_t,s_t)
}
$$

を計算する。重み付き平均と分散

$$
\widehat\mu_t
=\frac{\sum_hw_hz_h}{\sum_hw_h},
$$

$$
\widehat v_t
=\frac{
\sum_hw_h(z_h-\widehat\mu_t)^2
}{
\sum_hw_h
}
$$

は、選択領域に制限されたカイ標的分布の位置と広がりを近似する。

更新は新しい推定値へ一度に置き換えず、現在値と半分ずつ混合する。

$$
\mu_{t+1}
=0.5\mu_t+0.5\widehat\mu_t,
$$

$$
s_{t+1}
=\max\left(
0.25,
0.5s_t+0.5\sqrt{\max(\widehat v_t,0)}
\right).
$$

標準偏差の下限 0.25 は、proposal が極端に狭くなることを防ぐ。選択された pilot 標本が 1 個もない場合、または有限な重みが 1 個もない場合は

$$
s_{t+1}=1.5s_t
$$

として探索範囲を広げる。pilot 標本は proposal の調整だけに使用し、最終 $p$ 値の推定には再利用しない。

**実装対応:** pilot 標本、再選択、重み付き moment、減衰更新、標準偏差の下限、1.5 倍の拡張は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_adapt_proposal` に対応する。

### 11.3 defensive mixture

最終 proposal は 3 成分の混合分布である。

$$
q_{\mathrm{final}}(z)
=0.25f_{\chi_{r_j}}(z)
+0.375q_{\mathrm{obs}}(z)
+0.375q_{\mathrm{adapt}}(z).
$$

観測点を中心とする成分は

$$
q_{\mathrm{obs}}(z)
=q_{\mathrm{TN}}(z;T_j,s_{\mathrm{obs}}),
$$

$$
s_{\mathrm{obs}}
=\max(0.75,0.25T_j+0.5)
$$

であり、adapted 成分は

$$
q_{\mathrm{adapt}}(z)
=q_{\mathrm{TN}}
(z;\mu_{\mathrm{adapt}},s_{\mathrm{adapt}})
$$

である。$q_{\mathrm{obs}}$ は観測統計量とその上側近傍、 $q_{\mathrm{adapt}}$ は pilot で見つけた選択されやすい領域を重点的に探索する。

カイ成分を 25% 混ぜることにより、標的密度が正の位置では

$$
q_{\mathrm{final}}(z)
\ge0.25f_{\chi_{r_j}}(z)
$$

であり、したがって

$$
\frac{f_{\chi_{r_j}}(z)}{q_{\mathrm{final}}(z)}
\le4
$$

となる。このカイ成分を defensive component と呼ぶ。adapted proposal が標的分布の一部を十分に覆わなくても、標的の support を完全に失わず、厳密な算術では raw importance weight を 4 以下に保つ。

混合密度は、各成分の log-density に log mixture weight を加え、`logsumexp` で合成する。非常に小さい密度を早い段階で指数化することによる underflow を防ぐためである。

**実装対応:** mixture weight は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `DEFENSIVE_MIXTURE_WEIGHTS`、密度は `_defensive_mixture_logpdf`、標本化は `_sample_defensive_mixture` に対応する。

## 12. 最終 AIS 推定量

最終 proposal から

$$
z_1,\ldots,z_M
\overset{\mathrm{i.i.d.}}{\sim}
q_{\mathrm{final}}
$$

を生成し、

$$
w_m
=\frac{f_{\chi_{r_j}}(z_m)}{q_{\mathrm{final}}(z_m)}
$$

を計算する。選択的 $p$ 値の self-normalized estimator は

$$
\widehat p_j
=\frac{
\sum_{m=1}^M
w_ms_j(z_m)\mathbf1\{z_m\ge T_j\}
}{
\sum_{m=1}^M
w_ms_j(z_m)
}
$$

である。

実装は $s_j(z_m)=0$ の標本を保存しない。これは、保存したうえで重みに 0 を掛けることと代数的に同じである。

有限な log weight の最大値を引いてから指数化する。

$$
\widetilde w_m
=\exp\left(
\log w_m-\max_h\log w_h
\right).
$$

すべての重みに共通の正定数を掛けても、比率としての $\widehat p_j$ や ESS は変化しない。最大値を引く処理は overflow と underflow を抑えるための数値安定化である。

**実装対応:** 最終 batch loop、選択標本の保存、log weight の shift、 `sum(weights * tail) / sum(weights)` は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` に対応する。

## 13. AIS の収束診断

### 13.1 Effective Sample Size

Importance Sampling では、標本数が多くても、ごく少数の重みだけが支配的なら有効な情報量は小さい。重み $w_1,\ldots,w_m$ の Effective Sample Size（ESS）を

$$
\operatorname{ESS}(w_1,\ldots,w_m)
=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}
$$

と定義する。全重みが等しい場合は ESS $=m$ となり、1 個の重みが支配する場合は 1 に近づく。

実装は 2 種類の ESS を監視する。

- `denominator_ess`: 選択され、有限重みを持つ全標本の ESS
- `tail_ess`: さらに $z\ge T_j$ を満たす上側標本の ESS

分母の選択確率が安定していても、上側確率を評価する標本が不足する可能性があるため、両方を確認する。

**実装対応:** ESS の式は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_effective_sample_size`、2 種類の ESS の計算は `_run_ais` に対応する。

### 13.2 batch sampling と停止条件

最終段階は `final_batch_size` 個ずつ proposal を生成し、各 batch 後に

$$
\operatorname{ESS}_{\mathrm{den}}
\ge\texttt{min\_denominator\_ess}
$$

かつ

$$
\operatorname{ESS}_{\mathrm{tail}}
\ge\texttt{min\_tail\_ess}
$$

を確認する。両方を満たせば停止し、満たさない場合は `max_final_samples` に達するまで標本を追加する。

返される `status` は次の 3 種類である。

- `ok`: 両方の ESS 条件を満たし、有限な $p$ 値を返した。
- `no_selected_samples`: 最終 proposal に、選択されかつ有限重みを持つ標本がなかった。
- `insufficient_ess`: 選択標本はあったが、最大標本数までに ESS 条件を満たさなかった。

失敗時の $p$ 値は `NaN` であり、0 や 1 として解釈してはならない。proposal 数が多くても、重みが偏っている場合や上側標本が不足する場合は収束しない。

**実装対応:** `while` loop、停止条件、3 種類の status は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` に対応する。標本数と ESS 閾値の入力検証は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` 冒頭で行う。

### 13.3 Monte Carlo 標準誤差

収束した推定値について、Monte Carlo 分散を

$$
\widehat{\operatorname{Var}}_{\mathrm{MC}}(\widehat p_j)
=\frac{
\sum_iw_i^2
\left(
\mathbf1\{z_i\ge T_j\}-\widehat p_j
\right)^2
}{
(\sum_iw_i)^2
}
$$

で近似し、その平方根を `mc_se` とする。

`mc_se` は、観測データと選択写像を固定したときの有限回 AIS による数値誤差を表す。母集団効果の標準誤差ではなく、モデル誤指定や反復間変動も含まない。

**実装対応:** `mc_variance` と `mc_se` は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` が `status="ok"` の場合だけ計算・返却する。

## 14. 乱数 stream と再現性

最上位の整数 seed は NumPy の `SeedSequence` で階層的に分割される。各反復の子 seed は、さらに次の 3 stream へ分かれる。

1. $X$ と $y$ を生成する data stream
2. Random 法の特徴量を選ぶ random-selection stream
3. AIS に使用する stream

AIS stream は、SHAP で選択された特徴量ごとにさらに分割される。この構成により、論理的に異なる操作で意図せず同じ乱数列を共有することを避ける。

同一のコードと全設定を使えば結果を再現できる。ただし、`n_iters`、$k$、seed 階層などを変更すると、後続の子 stream も変わり得る。設定の一部を変えても以前の部分結果が必ず不変になるという意味ではない。

**実装対応:** [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` にある `SeedSequence(seed).spawn(n_iters)` と `iteration_seed.spawn(3)`、[`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` にある `SeedSequence(ais_seed).spawn(k_select)` に対応する。

## 15. 偽陽性率と反復集計

反復数を $R$、1 反復で検定する特徴量数を $k$ とする。手法 $m$ の反復 $r$、選択位置 $\ell$ の $p$ 値を $p_{m,r\ell}$ とすると、すべての反復が完了した場合の偽陽性率は

$$
\widehat{\mathrm{FPR}}_m
=\frac1R\sum_{r=1}^R
\left[
\frac1k\sum_{\ell=1}^k
\mathbf1\{p_{m,r\ell}<\alpha\}
\right]
$$

である。これは全 $Rk$ 仮説に対する棄却割合と一致する。

反復 $r$ の棄却割合を

$$
R_{m,r}
=\frac1k\sum_{\ell=1}^k
\mathbf1\{p_{m,r\ell}<\alpha\}
$$

とし、simulation standard error を

$$
\operatorname{SE}_{\mathrm{sim}}
=\frac{
\operatorname{sample\_sd}(R_{m,1},\ldots,R_{m,R})
}{\sqrt R}
$$

で計算する。同じデータセットから得た $k$ 個の $p$ 値は独立とは限らないため、個々の $p$ 値ではなく反復ごとの棄却割合を分散計算の単位とする。

$k=1$ では、この FPR は 1 反復で少なくとも 1 件を誤棄却する確率とも一致する。 $k>1$ では選択された仮説 1 件当たりの棄却割合であり、Family-wise Error Rate や False Discovery Rate と同一ではない。

### 15.1 AIS 失敗の扱い

ある手法について、1 反復内の $k$ 個の $p$ 値がすべて有限な場合だけ、その反復を complete とする。失敗率は

$$
\widehat f_{\mathrm{fail}}
=1-
\frac{
\#\{\text{complete iterations}\}
}{R}
$$

である。

1 回でも失敗がある場合、公式の `fpr` と `simulation_se` は `NaN` となる。収束しやすかったデータだけから性能を評価することを防ぐためである。 `converged_fpr` は complete な反復だけを使う診断値として残るが、収束しやすさがデータに依存する場合は偏る可能性があり、公式 FPR の代用にはならない。

**実装対応:** complete 判定、`failure_rate`、`fpr`、`simulation_se`、 `converged_fpr`、`n_pvalues` は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_method_summary` が計算する。`run_simulation` は Random、Unadjusted SHAP、 Selective SHAP (AIS) にそれぞれ適用する。

## 16. コマンドライン引数

[`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `parse_args` は次の引数を定義する。

| 引数 | 既定値 | 統計的・計算上の役割 |
|---|---:|---|
| `--n-iters` | 1 | 独立に生成するデータセット数 $R$ |
| `--n-samples` | 100 | 各データセットの標本数 $n$ |
| `--n-features` | 20 | 候補特徴量数 $d$ |
| `--k-select` | 1 | SHAP と Random がそれぞれ選ぶ特徴量数 $k$ |
| `--alpha` | 0.05 | FPR 集計で用いる棄却閾値 $\alpha$ |
| `--seed` | 123 | 乱数 stream 階層の root seed |
| `--selection-decimals` | 10 | 選択写像内での応答丸め桁数 |
| `--selection-event` | `exact_set` | `exact_set`、`feature_inclusion`、`exact_ranking` のいずれに条件付けるか |
| `--multiplicity` | `none` | featurewise 選択的 $p$ 値へ `none`、Holm、Bonferroni のどれを計算するか |
| `--pilot-iters` | 3 | proposal 適応反復数 |
| `--pilot-samples` | 40 | pilot 1 反復当たりの候補 $z$ 数 |
| `--final-batch-size` | 80 | 1 回の収束判定までに追加する最終標本数 |
| `--max-final-samples` | 800 | 選択特徴量 1 個当たりの最終 proposal 上限 |
| `--min-denominator-ess` | 80 | 分母推定に要求する ESS |
| `--min-tail-ess` | 15 | 選択上側推定に要求する ESS |
| `--rf-param NAME=VALUE` | なし | 選択写像を構成する Random Forest 引数の上書き |
| `--output-dir` | `outputs/selective_inference_ais` | 出力先。統計計算自体には影響しない |

`_validate_inputs` は $R\ge1$、$n>4$、$d\ge1$、$1\le k\le d$、 $0<\alpha<1$ を要求する。`run_simulation` はさらに、丸め桁数、pilot 数、最終標本数、ESS 閾値を検証する。

$n>4$ は 3 次元の中心化スプライン効果空間を扱うために十分な観測数を要求する条件である。ただし、実際のランクは特徴量の値に依存するため、実装は SVD から数値ランクを改めて計算する。

**実装対応:** 引数定義は `parse_args`、`NAME=VALUE` の変換は `_parse_rf_parameter`、統計入力の検証は [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_validate_inputs` と `run_simulation` 冒頭に対応する。

## 17. 出力ファイル

[`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `main` は 4 ファイルを保存する。

### 17.1 `selective_inference.csv`

`result["ais_diagnostics"]` を保存したもので、SHAP 選択特徴量 1 個につき 1 行を持つ。

| 列 | 意味と生成箇所 |
|---|---|
| `iteration` | 1 始まりの反復番号。`run_simulation` が追加する |
| `feature` | 0 始まりの選択特徴量 index。`run_simulation` が追加する |
| `rank` | `_spline_effect_basis` が求めた数値ランク $r_j$ |
| `t_obs` | `_chi_statistic` が求めた観測統計量 $T_j$ |
| `unadjusted_p_value` | 選択を無視した $\Pr(\chi_{r_j}\ge T_j)$ |
| `raw_selective_p_value`, `p_value` | `_run_ais` の featurewise 選択的 $p$ 値とその互換 alias。失敗時は `NaN` |
| `adjusted_selective_p_value` | `multiplicity` に従う none、Bonferroni、Holm の調整値 |
| `selection_event`, `selection_event_definition` | 条件付けた event mode と feature-specific な論理式 |
| `status` | `_run_ais` の `ok`、`no_selected_samples`、`insufficient_ess` |
| `proposals` | `_run_ais` が生成した最終 mixture 標本数 |
| `selected_samples` | 選択され、有限重みを持つ標本数 |
| `tail_samples` | さらに $z\ge T_j$ を満たす標本数 |
| `denominator_ess` | 選択された全有限重みの ESS |
| `tail_ess` | 選択された上側有限重みの ESS |
| `mc_se` | `status=ok` のときの AIS Monte Carlo 標準誤差 |

### 17.2 `summary.csv`

[`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_method_summary` が作る手法別の表である。Random、Unadjusted SHAP、 Selective SHAP (AIS) の各行に `fpr`、`simulation_se`、`converged_fpr`、 `failure_rate`、`n_pvalues` を記録する。

### 17.3 `converged_p_values.csv`

[`examples/run_selective_inference.py`](../examples/run_selective_inference.py) の `_p_values_frame` が、`result["p_values"]` の配列を `method` と `p_value` の long form に変換する。`_method_summary` が保持した complete な反復の値だけが含まれる。Random と Unadjusted は通常すべて有限であり、収束の問題は主に AIS に関係する。

### 17.4 `settings.json`

`main` がデータ次元、seed、選択事象、多重性、解決済み森林パラメータ、$\alpha$、AIS 設定を保存する。これらの変更は、選択事象、検定対象、または Monte Carlo 精度を変えるため、結果の再現に必要である。なお `run_simulation` の `summary.csv` にある FPR は常に `raw_selective_p_value` から計算され、`--multiplicity` で計算される `adjusted_selective_p_value` は `selective_inference.csv` の featurewise 結果に保存されるが FPR 集計には使われない。

## 18. 理論上および解釈上の制約

### 18.1 選択事象

既定の `exact_set` は、各特徴量固有の経路上で

$$
\operatorname{set}(\widehat S_k(X,y_j(Z)))
=\operatorname{set}(M_{\mathrm{obs}})
$$

に条件付ける。集合は順序を持たない。`feature_inclusion` はより弱い $j\in\widehat S_k(X,y_j(Z))$ に条件付け、`exact_ranking` は順序も含む完全一致に条件付ける。強い条件付けは選択確率と ESS を下げ、検出力を失う場合がある。いずれのモードでも $j$ ごとに経路が異なるため、$k$ 個の結果を返す。

**実装対応:** `selection_event_holds` が 3 モードを一元的に定義し、既定値は `exact_set` である。選択事象による条件付けだけでは多重検定は解決しないため、個別の raw p 値とは別に Holm または Bonferroni 補正を選択できる。

### 18.2 $X,a_j,u_j$ による条件付け

選択的 $p$ 値は、新しい $X$ やすべての応答方向について平均した無条件確率ではない。観測された $X$、直交成分 $a_j$、効果方向 $u_j$ を固定し、カイ半径 $Z$ だけを変化させた条件付き確率である。

**実装対応:** [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` と [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` にある `is_selected` closure が、固定された `X`、`orthogonal`、`direction` を保持する。

### 18.3 AIS の数値誤差

理論上の対象は決定的な選択写像に対する積分比であるが、`_run_ais` は有限個の標本で近似する。ESS と `mc_se` は数値精度を診断するが、誤差を完全には除去しない。`status` が `ok` でない結果は欠測であり、帰無仮説を支持または否定する証拠として扱わない。

**実装対応:** proposal 上限、ESS 閾値、status、`mc_se` は [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` が管理する。

### 18.4 決定的な選択写像

各 $z$ で同じ規則を再現する必要がある。応答丸め、固定森林 seed、Tree SHAP 設定、top-$k$、同順位規則はすべて選択事象の一部である。これらの変更は単なる計算設定の変更ではなく、条件付ける事象そのものを変更する。

`random_state=None` のような確率的森林を使用した場合、その追加乱数に条件付ける処理や積分する処理は現在の実装に含まれていない。

**実装対応:** 選択写像は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_tree_shap_importance_for_estimator`、`_top_k`、`ShapSelector.select` から構成される。`_tree_shap_importance` と `_select_features` は Random Forest 用の後方互換 shortcut である。

### 18.5 SHAP 重要度と因果効果の区別

本シミュレーションは、特定の SHAP 選択アルゴリズムを使用した後の統計検定を補正する。SHAP 値が因果効果であること、Random Forest が真のデータ生成機構を正しく表すこと、選択特徴量への介入が有効であることは示さない。

**実装対応:** [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_tree_shap_importance_for_estimator` は学習済み予測モデルの重要度だけを計算する。

### 18.6 正規性と既知分散

カイ半径と方向の独立性、および正確なカイ帰無分布は、球対称正規誤差と既知の $\sigma$ に依存する。相関誤差、不均一分散、非正規誤差、未知分散に現在の式をそのまま適用できない。別の帰無分布または選択的推論の拡張が必要となる。

**実装対応:** 正規帰無データは [`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `_generate_null_dataset` に対応する。実データ API は正の `sigma` を必須とし、 `variance_method="known_user_supplied"` と値を結果に保存する。理論的に検証済みの未知分散法は未実装であり、選択済みデータから暗黙に推定しない。

### 18.7 計算量

`is_selected(z)` を 1 回評価するたびに Random Forest を最初から学習し、Tree SHAP を再計算する。選択特徴量ごとに pilot 評価と最終 AIS 評価を行うため、計算時間は反復数、$k$、pilot 標本数、最終 proposal 数、木の本数、データサイズとともに増加する。

**実装対応:** [`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` が `is_selected` を繰り返し呼び、closure 内から [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_select_features` を実行する。

## 19. 関数と数理処理の対応表

| ファイルと関数 | 数理的・処理上の役割 |
|---|---|
| `examples/run_selective_inference.py::_parse_rf_parameter` | 森林引数を変換し、選択写像の設定に加える |
| `examples/run_selective_inference.py::parse_args` | シミュレーション、検定、AIS、出力の設定を定義する |
| `examples/run_selective_inference.py::_p_values_frame` | complete な反復の $p$ 値を long form に変換する |
| `examples/run_selective_inference.py::main` | シミュレーションを呼び、4 ファイルを保存し、AIS 診断を表示する |
| `src/si_shap/simulation.py::_generate_null_dataset` | グローバル帰無仮説下の独立な標準正規 $X,y$ を生成する |
| `src/si_shap/simulation.py::_validate_inputs` | 次元、選択数、有意水準を検証する |
| `src/si_shap/simulation.py::_method_summary` | FPR、simulation SE、失敗率、complete-case 診断を計算する |
| `src/si_shap/simulation.py::run_simulation` | 帰無データを反復し、`selective_inference` と Random baseline を呼び、3 手法の結果を集約する |
| `src/si_shap/selection.py::_resolve_rf_params` | 森林の既定値と利用者の上書きを統合する |
| `src/si_shap/selection.py::_tree_shap_importance` | 応答を丸め、森林を学習し、Tree SHAP の平均絶対重要度を返す |
| `src/si_shap/selection.py::_top_k` | 降順および index 昇順の同順位規則で上位 $k$ を選ぶ |
| `src/si_shap/selection.py::_select_features` | SHAP 重要度計算と top-$k$ 選択を合成する |
| `src/si_shap/inference.py::_spline_effect_basis` | 3 次 B スプラインを生成・中心化・正規直交化する |
| `src/si_shap/inference.py::_chi_statistic` | 応答を射影し、標準化された長さ $T_j$ を返す |
| `src/si_shap/inference.py::_truncated_normal_logpdf` | 0 以上に切断した正規 proposal の log-density を評価する |
| `src/si_shap/inference.py::_sample_truncated_normal` | 切断正規 proposal から非負の候補半径を生成する |
| `src/si_shap/inference.py::_defensive_mixture_logpdf` | カイ・観測点・adapted の最終 mixture 密度を安定に評価する |
| `src/si_shap/inference.py::_effective_sample_size` | importance weight の偏りを ESS に要約する |
| `src/si_shap/inference.py::_adapt_proposal` | 選択された pilot 標本で proposal の位置と尺度を調整する |
| `src/si_shap/inference.py::_sample_defensive_mixture` | 3 成分に標本数を割り当て、最終 proposal から標本化する |
| `src/si_shap/inference.py::_run_ais` | 選択的上側確率の比を推定し、ESS、status、Monte Carlo 誤差を返す |
| `src/si_shap/api.py::selective_inference` | 条件付き候補応答族と選択事象を構成し、選択特徴量ごとの unadjusted/raw/adjusted selective $p$ 値を返す |
| `src/si_shap/__init__.py` | `run_simulation`、`selective_inference`、selection-region、power、plotting API をパッケージ直下で公開する |

## 20. 数学的対象の要約

観測された SHAP 選択特徴量 $j$ について、実装は

$$
X,
\qquad
a_j=(I-P_j)y,
\qquad
u_j=\frac{P_jy}{\lVert P_jy\rVert_2}
$$

を固定し、

$$
y_j(z)=a_j+\sigma u_jz,
\qquad
Z\sim\chi_{r_j}
$$

を構成する。各 $z$ で選択写像を完全に再実行し、

$$
s_j(z)
=\mathbf1\{\operatorname{set}(\widehat S_k(X,y_j(z)))
=\operatorname{set}(M_{\mathrm{obs}})\}
$$

を評価する。推定対象は

$$
\boxed{
p_j^{\mathrm{selective}}
=\frac{
\mathbb E_{Z\sim\chi_{r_j}}
\left[
s_j(Z)\mathbf1\{Z\ge T_j\}
\right]
}{
\mathbb E_{Z\sim\chi_{r_j}}
\left[s_j(Z)\right]
}
}
$$

である。

[`src/si_shap/inference.py`](../src/si_shap/inference.py) の `_run_ais` は、 adapted truncated-normal 成分と defensive chi 成分を組み合わせた proposal によりこの比を推定する。その他の関数は、決定的な選択写像の構築、カイ分布の成立に必要な射影、数値信頼性の診断、帰無シミュレーションの反復集計を担当する。

## 21. 公開 API、3 種類の選択事象、多重性補正

### 21.1 `selective_inference` が検定する仮説

実データ用公開 API の featurewise 帰無仮説は、Random Forest の係数や SHAP 値が 0 という仮説ではない。固定した $X$ に対する条件付き平均を $\mu=\mathbb E[y\mid X]$ とすると、選択特徴量 $j$ について検定する対象は

$$
H_{0,j}:P_j\mu=0
\qquad\text{対}\qquad
H_{1,j}:P_j\mu\ne0
$$

であり、$P_j=Q_jQ_j^\top$ は特徴量 $X_{\cdot j}$ の中心化 3 次 B スプライン空間への射影である。これは周辺的な非線形関連の検定であり、他の特徴量を統制した条件付き効果でも、因果効果でも、SHAP 重要度そのものの検定でもない。

モデル仮定は $y\mid X\sim\mathcal N(\mu,\sigma^2I_n)$、既知かつ外部から与えられる $\sigma>0$、決定的な selector、固定された同順位規則と estimator 乱数である。`sigma` を選択済みの同じデータから暗黙に推定すると、ここで用いるカイ分布は正当化されない。

**実装対応:** 仮説と仮定の文章は [`src/si_shap/api.py`](../src/si_shap/api.py) の `HYPOTHESIS` と `ASSUMPTIONS`、入力条件は `_validate_data`、選択結果の形状と順位の検証は `_validate_selection_result`、一連の検定は `selective_inference` に対応する。selector の構築は [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `make_selector`、乱数固定の検証は `_check_estimator_reproducibility` が行う。

### 21.2 選択事象の厳密な定義

観測応答で得た順序付き top-$k$ を $M_{\mathrm{obs}}=(m_1,\ldots,m_k)$、特徴量 $j$ 固有の候補応答 $y_j(z)$ で得た順序付き top-$k$ を $M_j(z)$ と書く。各選択特徴量 $j$ に対して 3 種類の指示関数を選べる。

$$
s_j^{\mathrm{exact\ set}}(z)
=\mathbf1\{\operatorname{set}(M_j(z))
=\operatorname{set}(M_{\mathrm{obs}})\},
$$

$$
s_j^{\mathrm{inclusion}}(z)
=\mathbf1\{j\in M_j(z)\},
$$

$$
s_j^{\mathrm{exact\ ranking}}(z)
=\mathbf1\{M_j(z)=M_{\mathrm{obs}}\}.
$$

`exact_set` は選択集合全体を固定するが集合内の順位は固定せず、`feature_inclusion` は検定対象 $j$ が top-$k$ に残ることだけを固定し、`exact_ranking` は top-$k$ の特徴と順序をともに固定する。観測時に $j\in M_{\mathrm{obs}}$ なので同じ feature-specific path 上では

$$
\mathcal Z_j^{\mathrm{exact\ ranking}}
\subseteq
\mathcal Z_j^{\mathrm{exact\ set}}
\subseteq
\mathcal Z_j^{\mathrm{inclusion}}
$$

が成立する。したがって前二者ほど一般に選択確率が小さくなり得るが、条件付き $p$ 値や検出力の大小がこの包含関係だけで単調に決まるわけではない。$k=1$ では `exact_set` と `feature_inclusion` は同じ事象になる。

**実装対応:** [`src/si_shap/selection.py`](../src/si_shap/selection.py) の `_validate_selection_event` が選択肢を検証し、`selection_event_holds` が上記 3 指示関数を評価し、`selection_event_definition` が出力用の定義を返す。これらは [`src/si_shap/api.py`](../src/si_shap/api.py) の `selective_inference` と [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` の双方から使われる。

### 21.3 featurewise raw $p$ 値と多重性補正

選択事象で条件付けた $p_j^{\mathrm{sel}}$ は featurewise に有効であるが、$k$ 個を同時に検定する family-wise error を自動的には制御しない。`multiplicity="none"` は

$$
p_j^{\mathrm{adj}}=p_j^{\mathrm{sel}}
$$

を返す。Bonferroni は family size を $k$ として、有限な各値を

$$
p_j^{\mathrm{Bonf}}=\min\{1,kp_j^{\mathrm{sel}}\}
$$

へ変換する。AIS に失敗した仮説も family size $k$ には含めるため、失敗行を除いて小さい family を作る反保守的な処理はしない。

Holm 法では raw $p$ 値を $p_{(1)}\le\cdots\le p_{(k)}$ と並べ、順位 $i$ の調整値を

$$
p_{(i)}^{\mathrm{Holm}}
=\min\left\{1,
\max_{1\le h\le i}(k-h+1)p_{(h)}
\right\}
$$

とし、元の特徴順へ戻す。Holm は全 raw $p$ 値の順序を必要とするため、1 件でも AIS 失敗による `NaN` があれば family 全体の Holm 調整値を `NaN` とする。

**実装対応:** 上式は [`src/si_shap/api.py`](../src/si_shap/api.py) の `adjust_p_values` に対応し、`selective_inference` は `raw_selective_p_value`、互換 alias の `p_value`、`adjusted_selective_p_value` を `feature_results` に保存する。[`src/si_shap/power.py`](../src/si_shap/power.py) の `compare_selection_event_power` は `multiplicity="none"` なら raw 列、それ以外なら adjusted 列を棄却判定に使う。一方、[`src/si_shap/simulation.py`](../src/si_shap/simulation.py) の `run_simulation` が報告する FPR は raw 列を使う。

## 22. `plot_selection_regions.py`：選択領域を明示的に求める理論

### 22.1 AIS と選択領域 scan の違い

AIS は $s_j(z)$ を標本点で評価するだけで $\mathcal Z_j=\{z\ge0:s_j(z)=1\}$ の境界を求めない。可視化例は同じ feature-specific path を有限区間で密に走査し、選択領域を区間の和として数値近似する。この処理は選択的 $p$ 値の計算に必要な段階ではなく、選択事象がどれほど断片的または稀であるかを理解する診断である。

**実装対応:** [`examples/plot_selection_regions.py`](../examples/plot_selection_regions.py) の `main` が [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` を呼び、`selection_regions_frame` で CSV を作り、[`src/si_shap/plotting.py`](../src/si_shap/plotting.py) の `plot_selection_regions` で図を保存する。

### 22.2 走査上限と省略 tail

数値走査の上限は、指定した tail probability $\varepsilon$ と観測統計量 $T_j$ から

$$
z_{\max,j}
=\max\left\{
F_{\chi_{r_j}}^{-1}(1-\varepsilon),
1.05T_j
\right\}
$$

とする。これにより観測点を必ず区間内部に含め、かつ走査外に残る無条件カイ確率を

$$
\varepsilon_{\mathrm{omit},j}
=\Pr(\chi_{r_j}>z_{\max,j})
=\overline F_{\chi_{r_j}}(z_{\max,j})
\le\varepsilon
$$

に抑える。これは無条件帰無分布の省略質量であり、選択条件付き分布の相対的な省略割合ではない。

**実装対応:** [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` が `stats.chi.ppf` で $z_{\max,j}$、`stats.chi.sf` で `omitted_tail_probability` を計算する。[`examples/plot_selection_regions.py`](../examples/plot_selection_regions.py) の `main` は $\varepsilon=10^{-8}$ を渡す。

### 22.3 grid による遷移検出と二分法

$[0,z_{\max,j}]$ に `grid_size` 個の等間隔点を置き、anchor として $T_j$ を追加する。昇順かつ重複を除いた grid を $0=z_0<z_1<\cdots<z_G=z_{\max,j}$ とし、隣接状態 $s_j(z_g)$ と $s_j(z_{g+1})$ が異なる区間を境界候補とする。候補区間 $[L,R]$ では midpoint $M=(L+R)/2$ の状態を左端と比較して同じ側の端点を更新し、$R-L\le\delta$ になるまで二分する。返す境界は $(L+R)/2$ で、$\delta$ が `boundary_tol` である。

検出された境界を $0=b_0<b_1<\cdots<b_H=z_{\max,j}$ とし、状態が真の区間だけを集めると

$$
\widehat{\mathcal Z}_j
=\bigcup_{h\in\mathcal I_j}[b_h,b_{h+1}]
$$

を得る。等間隔 grid の同じ cell 内で状態が偽から真、さらに偽へ戻るような幅の狭い成分は、anchor がその内部にない限り見落とし得る。`boundary_tol` を小さくしても未検出成分は復元できないため、探索解像度は主に `grid_size` が決める。観測点 $T_j$ を anchor に入れることで、少なくとも観測点を含む非常に狭い成分を発見できる可能性を高める。

**実装対応:** [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `find_selection_intervals` が grid、anchor、状態変化、区間結合を担当し、`_refine_transition` が二分法を実行する。`compute_selection_regions` は `anchor_points=(t_obs,)` を渡す。[`examples/plot_selection_regions.py`](../examples/plot_selection_regions.py) の `main` は `grid_size=1001`、`boundary_tol=1e-8` を使用する。

### 22.4 選択確率と条件付き密度

近似領域を互いに重ならない区間 $\widehat{\mathcal Z}_j=\bigcup_{h=1}^{H_j}[L_{jh},U_{jh}]$ とすると、走査範囲内の選択確率は

$$
\widehat\pi_j
=\sum_{h=1}^{H_j}
\left[
F_{\chi_{r_j}}(U_{jh})
-F_{\chi_{r_j}}(L_{jh})
\right].
$$

図に描く条件付き密度は、走査で得た領域内で

$$
\widehat f_{j,\mathrm{cond}}(z)
=\frac{f_{\chi_{r_j}}(z)}{\widehat\pi_j}
\mathbf1\{z\in\widehat{\mathcal Z}_j\}
$$

である。走査外の選択領域を省略しているため、$\widehat\pi_j$ は厳密な全半直線上の選択確率ではなく、tail 誤差と grid 誤差を含む数値近似である。

**実装対応:** [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `selection_probability` が区間ごとのカイ CDF 差を合計し、`compute_selection_regions` が結果を `SelectionRegionResult.selection_probability` に保存する。[`src/si_shap/plotting.py`](../src/si_shap/plotting.py) の `_plot_selection_region_on_axis` が `null_density / selection_probability` を領域内だけ描く。

### 22.5 AIS proposal を同じ図に描く意味

最終 AIS proposal は 11.3 節の

$$
q_{\mathrm{final}}(z)
=0.25f_{\chi_{r_j}}(z)
+0.375q_{\mathrm{obs}}(z)
+0.375q_{\mathrm{adapt}}(z)
$$

である。図は無条件カイ密度、選択領域上の条件付き密度、観測 $T_j$、最終 proposal を重ねるため、proposal が選択領域と上側 tail をどの程度覆うかを確認できる。`show_proposal_components=True` の場合だけ観測中心と adapted の各切断正規成分も描く。

**実装対応:** [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `compute_selection_regions` が `_adapt_proposal` を呼び、`observed_proposal_sd=max(0.75,0.25T_j+0.5)` と adapted parameters を記録する。[`src/si_shap/plotting.py`](../src/si_shap/plotting.py) の `plot_selection_regions` が panel 配置を行い、`_plot_selection_region_on_axis` が `_truncated_normal_logpdf` と `_defensive_mixture_logpdf` から密度を描く。

### 22.6 `plot_selection_regions.py` の関数対応と出力

- [`examples/plot_selection_regions.py`](../examples/plot_selection_regions.py) の `_parse_rf_parameter` は `NAME=VALUE` を JSON として可能な限り型付き変換し、森林パラメータが定義する選択写像を変更する。
- 同ファイルの `parse_args` は dataset seed、$k$、選択事象、Random Forest 上書きを定義する。
- 同ファイルの `main` は各 seed について $n=100,d=20,m=10$ の帰無データを生成する設定で `compute_selection_regions` を呼び、`shap_selection_regions.csv` と `shap_selection_regions.png` を `outputs/shap_selection_regions` に保存する。
- [`src/si_shap/selection_regions.py`](../src/si_shap/selection_regions.py) の `selection_regions_frame` は区間和を文字列 `"[L, U] U ..."` に変換し、その他の `SelectionRegionResult` fields とともに 1 選択特徴量 1 行の表を作る。

## 23. `sweep_selection_region_settings.py`：設定感度分析の数理

### 23.1 実験行列

設定 sweep は top-$k$ 値の集合 $\mathcal K$、森林設定の集合 $\mathcal F$、選択事象の集合 $\mathcal E$ の直積

$$
\mathcal G=\mathcal K\times\mathcal F\times\mathcal E
$$

の各 cell $g=(k,f,e)$ について 22 節の選択領域を再計算する。同じ dataset seeds を各 cell で使うため、差はデータの違いではなく設定の違いとして比較しやすい。ただし cell 間の領域境界や選択特徴量は非線形かつ離散的に変わるので、通常の paired estimator や信頼区間をこのスクリプトは計算しない。

### 23.2 各軸が数理対象をどう変えるか

$k$ を変えると top-$k$ 写像 $\widehat S_k$ と選択事象 $s_j(z)$ が変わる。`exact_set` では $k$ 個すべての一致を要求するため、$k$ が大きいほど単純に事象が狭くなるとは限らないが、満たすべき集合一致が変わる。森林設定 $f$ は木の本数、深さ、leaf 最小標本数を通じて学習写像 $y\mapsto\widehat f_y$、Tree SHAP 値 $\phi_{ij}(y)$、重要度 $I_j(y)$、最終的な $\widehat S_k(y)$ のすべてを変える。選択事象 $e$ は 21.2 節のどの条件へ制限するかを変える。したがって sweep は単なる描画 style の比較ではなく、条件付ける統計的事象そのものの感度分析である。

### 23.3 preset と森林設定

`baseline`、`shallow`、`medium`、`flexible` はそれぞれ異なる $(B,\text{max depth},\text{min leaf size})$ を指定する。全設定で `random_state=42` を固定し、`n_jobs` は並列計算数だけを変えるため、同じソフトウェア環境における数学的な選択写像を決定的に保つ。`quick` は $\mathcal K=\{1,2\}$、森林 `baseline, medium`、`exact_set`、`recommended` は $\mathcal K=\{1,2,5,10\}$ へ広げ、`comprehensive` は 4 森林、$\{1,2,5,10\}$、`exact_set, feature_inclusion` の直積を使う。

**実装対応:** [`examples/sweep_selection_region_settings.py`](../examples/sweep_selection_region_settings.py) の `RF_CONFIGS` が森林集合 $\mathcal F$、`PRESETS` が直積の各軸を定義する。`resolve_experiments` が CLI override または preset を解決して $1\le k\le20$ を検証し、`run_experiment` が 1 cell の `compute_selection_regions`、CSV、PNG を担当し、`main` が直積を列挙して `all_selection_regions.csv` に結合する。`parse_args` は seed、preset、各軸、grid、並列数、出力先を定義する。

### 23.4 sweep 出力の解釈

各 experiment directory の `selection_regions.csv` と `selection_regions.png` は 22 節と同じ feature-specific 数学的対象を表す。追加列 `experiment` は $(k,f,e)$ の識別子、`rf_config` は $f$ の名前、`rf_params` は選択写像を再現する具体的パラメータである。root の `all_selection_regions.csv` は全 cell を縦結合した表であり、`selection_probability`、区間数や境界文字列、`omitted_tail_probability`、proposal parameters を設定間で比較できる。

## 24. `compare_selection_event_power.py`：対比較検出力実験

### 24.1 対立仮説下のデータ生成

反復 $r=1,\ldots,R$ で、まず

$$
X_{rij}\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1)
$$

を生成する。signal feature 集合を $S_0\subseteq\{0,\ldots,d-1\}$、signal strength を $\beta>0$ とする。各 signal feature $j$ について raw nonlinear effect は

$$
g(x)=x+\frac12(x^2-1)
$$

である。標準正規 $X$ に対する母平均は $0$ だが、実装は有限標本で厳密に中心化・標準化する。反復 $r$、特徴量 $j$ の値を

$$
\bar g_{rj}=\frac1n\sum_{i=1}^n g(X_{rij}),
\qquad
s_{rj}=\sqrt{\frac1n\sum_{i=1}^n
\{g(X_{rij})-\bar g_{rj}\}^2},
$$

$$
h_{rj,i}=\frac{g(X_{rij})-\bar g_{rj}}{s_{rj}}
$$

とし、応答を

$$
y_{ri}=\mu_{ri}+\varepsilon_{ri},
\qquad
\mu_{ri}=\beta\sum_{j\in S_0}h_{rj,i},
\qquad
\varepsilon_{ri}\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1)
$$

で生成する。各 $h_{rj}$ は標本平均 0、NumPy の `ddof=0` による標本標準偏差 1 なので、各 signal feature の寄与 $\beta h_{rj}$ は経験標準偏差 $\beta$ を持つ。複数 signal の有限標本相関は 0 と強制していないため、$\mu_r$ 全体の経験分散は一般に $|S_0|\beta^2$ と厳密には一致しない。noise standard deviation は `SIMULATION_SIGMA=1` である。

**実装対応:** [`src/si_shap/power.py`](../src/si_shap/power.py) の `_nonlinear_effect` が $g$ の経験中心化と `np.std` による標準化、`_generate_power_dataset` が $X$、$\mu$、$y$ を生成し、`_validate_power_inputs` が $S_0$、$\beta$、比較する選択事象を検証する。

### 24.2 選択、検定、棄却の指示変数

選択事象 mode を $e\in\mathcal E$ とする。同じ観測データに対する top-$k$ selector は mode $e$ に依存しないので、反復 $r$ の観測選択集合を $M_r$ と共通に書ける。signal feature $j\in S_0$ について

$$
A_{rje}=\mathbf1\{j\in M_r\}
$$

を選択指示変数とする。$A_{rje}=1$ のときだけ mode $e$ の選択的 $p$ 値 $p_{rje}$ が計算される。多重性が `none` なら raw selective 値、Holm または Bonferroni なら adjusted selective 値を用い、棄却指示変数を

$$
D_{rje}
=\mathbf1\{A_{rje}=1,\ p_{rje}<\alpha,
\ p_{rje}\text{ is finite}\}
$$

と定義する。未選択 signal は $D_{rje}=0$ であり、AIS が失敗した選択 signal も表上の `rejected=False` だが、後者は strict power を利用不能にする点で未選択とは区別される。

**実装対応:** [`src/si_shap/power.py`](../src/si_shap/power.py) の `compare_selection_event_power` が各 event について `selective_inference` を呼び、`p_value_column` を選び、`failed = ~isfinite(p_value_used)`、`rejected = p_value_used < alpha` を作る。signal が選択されなかった場合は `selected=False, failed=False, rejected=False` の record を明示的に追加する。

### 24.3 overall power の正確な計算

この例の主要な検出力は「真の signal が SHAP で選択され、かつ選択的検定で棄却される」確率であり、選択後だけの条件付き検出力ではない。signal 数を $s=|S_0|$ とし、すべての selected-signal AIS が成功した場合の反復別検出率を

$$
P_{re}=\frac1s\sum_{j\in S_0}D_{rje}
$$

とすると、報告する strict overall power は

$$
\widehat{\mathrm{Power}}_e
=\frac1R\sum_{r=1}^R P_{re}
=\frac1{Rs}\sum_{r=1}^R\sum_{j\in S_0}D_{rje}.
$$

したがって signal が選択されなかった反復も分母 $Rs$ に残り 0 として寄与する。これは「選択できる能力」と「選択後に棄却できる能力」の両方を含む end-to-end power である。

1 件でも selected-signal AIS が失敗すれば `power` と `simulation_se` は `NaN` になる。失敗を除外した値を主要結果と誤認しないための strict 方針である。診断用 `converged_power` は、selected signal の失敗が 1 件もない反復集合 $\mathcal C_e$ だけを使い、

$$
\widehat{\mathrm{Power}}_e^{\mathrm{conv}}
=\frac1{|\mathcal C_e|}
\sum_{r\in\mathcal C_e}P_{re}
$$

と計算する。未選択 signal は失敗ではなく 0 なので complete 判定を妨げない。収束可能性がデータや event に依存すれば complete-case 値は偏り得る。

**実装対応:** [`src/si_shap/power.py`](../src/si_shap/power.py) の `_summarize_event_power` が `iteration_complete`、`iteration_power`、`converged_power` を作り、全 selected-signal tests が成功した `strict_complete` の場合だけ同じ値を `power` として公開する。

### 24.4 power の simulation standard error

同じ反復の $s$ signal features は共通の $X,y$ と選択集合を持つので独立とみなさず、反復別 $P_{re}$ を単位に標準誤差を計算する。complete 反復数を $R_e^\star=|\mathcal C_e|$、その平均を $\bar P_e$ とすると

$$
\widehat{\mathrm{SE}}_{\mathrm{sim},e}
=\frac{
\sqrt{
\frac1{R_e^\star-1}
\sum_{r\in\mathcal C_e}(P_{re}-\bar P_e)^2
}
}{\sqrt{R_e^\star}}.
$$

$R_e^\star=1$ なら sample standard deviation を定義できないため `NaN` である。失敗が一件もなければ `simulation_se` と `converged_simulation_se` は同じ値になり、失敗があれば前者だけ `NaN` になる。

**実装対応:** [`src/si_shap/power.py`](../src/si_shap/power.py) の `_summarize_event_power` が `iteration_power.std(ddof=1) / sqrt(iteration_power.size)` を計算する。

### 24.5 selection rate、conditional power、failure rate

overall power を分解して読むため、signal selection rate は

$$
\widehat{\Pr}(A=1)
=\frac1{Rs}\sum_{r=1}^R\sum_{j\in S_0}A_{rje}
$$

と計算する。観測 selector は event mode より前に共通に実行されるため、同一の paired run では selection rate は mode 間で同じである。

selected かつ AIS が成功した signal test の集合を $\mathcal V_e$ とすると、診断用 conditional power は

$$
\widehat{\Pr}(D=1\mid A=1,\mathrm{converged})
=\frac{
\sum_{(r,j)\in\mathcal V_e}D_{rje}
}{|\mathcal V_e|}.
$$

strict `conditional_power` は selected-signal failure が全実験で 0 の場合だけ返し、`converged_conditional_power` は成功例だけから常に可能な範囲で返す。signal test failure rate は

$$
\widehat f_{\mathrm{signal\ fail},e}
=\frac{
\#\{(r,j):A_{rje}=1,\ p_{rje}\text{ is nonfinite}\}
}{
\#\{(r,j):A_{rje}=1\}
}
$$

であり、signal が一度も選択されなければ 0 とする。

`converged_null_rejection_rate` は候補となった全 null features に対する FPR ではない。top-$k$ に選択され `feature_results` に現れた null features のうち AIS に成功したものだけを分母として棄却割合を計算する診断値である。`null_test_failure_rate` も同じ selected-null 集合を分母にする。

**実装対応:** これらは [`src/si_shap/power.py`](../src/si_shap/power.py) の `_summarize_event_power` に対応する。`signal_results` はすべての $(r,e,j\in S_0)$ を含み、`feature_results` は実際に top-$k$ に選択されて検定された特徴量だけを含む。

### 24.6 paired power difference

各反復で全 event modes に同じ $(X_r,y_r)$、同じ決定的 selector、同じ AIS seed を使う。baseline event を $e_0$、比較 event を $e_1$ とし、双方で全 signal tests が complete な paired 反復集合を $\mathcal C_{01}$ とする。反復別差は

$$
\Delta_r
=\frac1s\sum_{j\in S_0}
\left(D_{rje_1}-D_{rje_0}\right),
$$

paired estimate と標準誤差は

$$
\widehat\Delta
=\frac1{|\mathcal C_{01}|}
\sum_{r\in\mathcal C_{01}}\Delta_r,
\qquad
\widehat{\mathrm{SE}}_{\mathrm{paired}}
=\frac{\operatorname{sample\_sd}(\Delta_r:r\in\mathcal C_{01})}
{\sqrt{|\mathcal C_{01}|}}.
$$

出力する近似 95% interval は

$$
\widehat\Delta\pm1.96\widehat{\mathrm{SE}}_{\mathrm{paired}}
$$

であり、[0,1] や [-1,1] へ clip しない正規近似 interval である。正の差は比較 event の overall power が baseline より高いことを示す。strict `power_difference` は両 event の strict power が有限な場合だけ差を取り、`converged_power_difference` は complete pairs から計算する。

同じ AIS seed は proposal 乱数を揃えて Monte Carlo noise を相関させ、event 差の分散を減らす common-random-numbers 設計である。ただし event により `is_selected` の真偽、適応結果、停止 batch が変わるので、最終的に使われる標本列が完全に同じになるとは限らない。

**実装対応:** [`src/si_shap/power.py`](../src/si_shap/power.py) の `compare_selection_event_power` が `iteration_seed.spawn(2)` から data seed と shared AIS seed を作り、観測 top-$k$ が event 間で一致することを検証する。`_paired_comparisons` が `iteration, feature` で pivot し、complete pairs、$\Delta_r$、平均、paired SE、95% interval を計算する。

### 24.7 power plot の値と error bar

bar の高さは strict `power` が有限ならそれを使い、strict 値が `NaN` なら `converged_power` へ fallback して label に `*` を付ける。両方 `NaN` なら高さ 0、label `NA*` とする。error bar は常に `converged_simulation_se` を使い、非有限なら 0 に置き換える。したがって `*` 付き bar は主要な strict estimate ではなく complete iterations だけの診断であり、failure rates と一緒に読む必要がある。水平線 $y=\alpha$ は参照線にすぎず、power が $\alpha$ を超えるかどうかを仮説検定しているわけではない。

**実装対応:** [`examples/compare_selection_event_power.py`](../examples/compare_selection_event_power.py) の `_plot_power_comparison` が fallback、asterisk、`converged_simulation_se`、$\alpha$ 線を実装する。

### 24.8 power example の関数対応と出力

- [`examples/compare_selection_event_power.py`](../examples/compare_selection_event_power.py) の `_parse_rf_parameter` は森林 parameter override を型付き値へ変換する。
- 同ファイルの `parse_args` は $R,n,d,k,S_0,\beta,\alpha$、選択事象列、多重性、AIS、森林、出力先を定義する。既定出力先は起動時の current directory ではなく repository root 下の `outputs/selection_event_power` である。
- 同ファイルの `main` は [`src/si_shap/power.py`](../src/si_shap/power.py) の `compare_selection_event_power` を呼び、4 CSV、1 JSON、1 PNG を保存して要約を表示する。
- `power_summary.csv` は event 別の strict/converged overall power、simulation SE、conditional power、selection rate、signal/null failure diagnostics を持つ。
- `paired_power_comparison.csv` は baseline に対する strict/converged paired difference、paired SE、近似 95% interval、complete pair 数を持つ。
- `signal_results.csv` はすべての反復・event・真の signal feature について `selected`、`p_value`、`failed`、`rejected` を持つため overall power の分母を監査できる。
- `feature_results.csv` は top-$k$ に選択され実際に検定された signal/null features の詳細と AIS diagnostics を持つ。
- `settings.json` は DGP、選択、検定、AIS、森林の再現設定を保存し、`power_comparison.png` は 24.7 節の要約を描く。

## 25. 4 本の example script と数理理論の完全対応表

| example の関数 | 呼び出す主要関数 | 対応する数理理論 |
|---|---|---|
| `examples/run_selective_inference.py::_parse_rf_parameter` | JSON parser | 森林 hyperparameter を決定的選択写像 $\widehat S_k$ の一部として定義する |
| `examples/run_selective_inference.py::parse_args` | `argparse` | 帰無 DGP の $R,n,d$、$k$、$\alpha$、選択事象、多重性、AIS 精度を定義する |
| `examples/run_selective_inference.py::_p_values_frame` | DataFrame construction | complete iterations の featurewise $p$ 値を method 別 long form にする |
| `examples/run_selective_inference.py::main` | `simulation.run_simulation` | 4–15 節の帰無実験、FPR、simulation SE、AIS failure policy を実行・保存する |
| `examples/plot_selection_regions.py::_parse_rf_parameter` | JSON parser | 可視化対象の森林選択写像を指定する |
| `examples/plot_selection_regions.py::parse_args` | `argparse` | dataset seeds、$k$、event を固定し feature-specific path を定義する |
| `examples/plot_selection_regions.py::main` | `compute_selection_regions`, `selection_regions_frame`, `plot_selection_regions` | 22 節の grid/bisection 領域、カイ質量、条件付き密度、AIS proposal を計算・描画する |
| `examples/sweep_selection_region_settings.py::parse_args` | `argparse` | 感度分析の直積 $\mathcal K\times\mathcal F\times\mathcal E$ と数値解像度を定義する |
| `examples/sweep_selection_region_settings.py::resolve_experiments` | `PRESETS`, `RF_CONFIGS` | preset/override から実験行列を解決し $k$ の範囲を検証する |
| `examples/sweep_selection_region_settings.py::run_experiment` | `compute_selection_regions`, `plot_selection_regions` | 直積の 1 cell で選択領域と選択確率を計算する |
| `examples/sweep_selection_region_settings.py::main` | Cartesian-product comprehension | 全 cell を列挙し結果を縦結合して設定感度を比較可能にする |
| `examples/compare_selection_event_power.py::_parse_rf_parameter` | JSON parser | power DGP で共通利用する決定的森林を指定する |
| `examples/compare_selection_event_power.py::parse_args` | `argparse` | $R,n,d,k,S_0,\beta,\alpha$、比較 event、多重性、AIS を定義する |
| `examples/compare_selection_event_power.py::_plot_power_comparison` | Matplotlib | strict power または明示された converged fallback と simulation SE を可視化する |
| `examples/compare_selection_event_power.py::main` | `power.compare_selection_event_power` | 24 節の nonlinear alternative、overall/conditional power、paired difference を実行・保存する |

## 26. core Python 関数と数式の拡張対応表

| core の関数 | 対応する数学的対象 |
|---|---|
| `src/si_shap/selection.py::_tree_shap_importance_for_estimator` | $y$ の丸め、estimator の clone/refit、Tree SHAP $\phi_{ij}$、$I_j=n^{-1}\sum_i|\phi_{ij}|$ |
| `src/si_shap/selection.py::ShapSelector.select` | 全特徴順位と $\widehat S_k$ |
| `src/si_shap/selection.py::selection_event_holds` | $s_j^{\mathrm{exact\ set}}$、$s_j^{\mathrm{inclusion}}$、$s_j^{\mathrm{exact\ ranking}}$ |
| `src/si_shap/api.py::selective_inference` | $H_{0,j}$、$y=a_j+\sigma u_jT_j$、$y_j(z)$、featurewise 選択的 $p$ 値 |
| `src/si_shap/api.py::adjust_p_values` | none、Bonferroni、Holm の $p_j^{\mathrm{adj}}$ |
| `src/si_shap/selection_regions.py::find_selection_intervals` | $\widehat{\mathcal Z}_j$ の grid 発見と二分境界 refinement |
| `src/si_shap/selection_regions.py::selection_probability` | $\sum_h[F_\chi(U_h)-F_\chi(L_h)]$ |
| `src/si_shap/selection_regions.py::compute_selection_regions` | null data、feature-specific path、$z_{\max}$、領域、proposal の統合 |
| `src/si_shap/selection_regions.py::selection_regions_frame` | `SelectionRegionResult` の監査可能な tabular representation |
| `src/si_shap/plotting.py::plot_selection_regions` | 複数 dataset/feature の panel layout |
| `src/si_shap/plotting.py::_plot_selection_region_on_axis` | $f_\chi$、$f_\chi/\pi_j$、$q_{\mathrm{final}}$、$T_j$、$\mathcal Z_j$ の重ね描き |
| `src/si_shap/power.py::_nonlinear_effect` | $g(x)=x+\tfrac12(x^2-1)$ の経験中心化・標準化 $h$ |
| `src/si_shap/power.py::_generate_power_dataset` | $y=\beta\sum_{j\in S_0}h_j(X_j)+\varepsilon$ |
| `src/si_shap/power.py::_summarize_event_power` | overall/conditional power、selection/failure/null diagnostics、simulation SE |
| `src/si_shap/power.py::_paired_comparisons` | $\Delta_r$、$\widehat\Delta$、paired SE、正規近似 95% interval |
| `src/si_shap/power.py::compare_selection_event_power` | 同一データ・selector・AIS seed による event modes の paired experiment |

## 27. 結果を正しく解釈するための最終チェック

1. `raw_selective_p_value` は固定 $X$、固定 $a_j,u_j$、指定した selection event の下での featurewise 条件付き $p$ 値であり、Random Forest の係数、SHAP 値、因果効果の検定ではない。
2. `adjusted_selective_p_value` は選択条件への補正とは別の多重性補正であり、`run_simulation` の FPR は raw 値、power example は `multiplicity` に応じた raw/adjusted 値を使う。
3. `status="ok"` で ESS と `mc_se` を確認できる結果だけを数値的に収束した AIS estimate とみなし、`NaN` を非棄却へ置換して strict FPR/power を計算しない。
4. `selection_probability` は有限 grid と有限 $z_{\max}$ による可視化用近似であり、非常に狭い未観測成分を見落とす可能性と `omitted_tail_probability` がある。
5. `power` は $\Pr(\text{signal selected and rejected})$ の end-to-end estimate であり、`conditional_power` は選択され AIS が成功した signal に限定される。両者の分母は異なる。
6. `converged_fpr`、`converged_power`、`converged_conditional_power` は失敗のない subset から得た診断値であり、strict 指標が `NaN` のときの無条件な代替ではない。
7. exact-set、inclusion、exact-ranking、$k$、森林 hyperparameters、丸め桁数、同順位規則の変更は、すべて条件付ける選択写像または事象を変える。
8. 全例のカイ理論は独立な正規誤差、既知 $\sigma$、決定的 selector に依存し、未知分散・相関誤差・不均一分散へそのまま拡張できない。
