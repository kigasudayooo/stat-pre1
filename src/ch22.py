import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(np):
    X = np.array([
        [2,2,3,1],
        [9,8,10,9],
        [8,3,2,7],
        [7,1,3,8],
        [2,9,8,2],
        [5,4,5,5],
    ])
    return (X,)


@app.cell
def _(np):
    def centering_matrix(X):
        """
        列単位の平均値を引いてセントリングする関数
        """
        n, d = X.shape

        I = np.eye(n)
        J = np.ones((n, n)) 
        C = I - (1/n) * J 

        X_centered = C @ X

        return X_centered
    return (centering_matrix,)


@app.cell
def _(X, centering_matrix):
    X_c = centering_matrix(X)
    return (X_c,)


@app.cell
def _(X, X_c):
    n ,_ = X.shape

    S = 1/(n-1) * X_c.T @ X_c
    S
    return S, n


@app.cell
def _(S, np):
    np.linalg.svd(S)
    return


@app.cell
def _(S, np):
    np.linalg.eig(S)
    return


@app.cell
def _(S, np):
    # 変数ごとの標準偏差を計算
    std_devs = np.sqrt(np.diag(S))
    std_devs
    return (std_devs,)


@app.cell
def _(np, std_devs):
    # 標準偏差の逆数で対角化した行列
    D_inv = np.diag(1/std_devs)
    D_inv
    return (D_inv,)


@app.cell
def _(D_inv, S):
    # 相関行列の計算: R = D^(-1) * S * D^(-1)
    R = D_inv @ S @ D_inv
    R
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 特異値分解と固有値分解の関係

        ## 基本概念

        ### 固有値分解 (Eigendecomposition)
        任意の正方行列 $A$ に対して、以下の式が成り立つとき、$\lambda$ を固有値、$v$ を対応する固有ベクトルと呼ぶ：

        $$A v = \lambda v$$

        固有値分解では、行列 $A$ を以下のように分解できる：

        $$A = P \Lambda P^{-1}$$

        ここで：
        - $P$ は固有ベクトルを列とする行列
        - $\Lambda$ は固有値を対角成分とする対角行列
        - $P^{-1}$ は $P$ の逆行列

        ただし、すべての行列が固有値分解できるわけではない。

        ### 特異値分解 (SVD: Singular Value Decomposition)
        一方、特異値分解はより正方行列ではない行列にも適用できる。$m \times n$ 行列 $A$ の特異値分解は：

        $$A = U \Sigma V^T$$

        ここで：
        - $U$ は $m \times m$ の直交行列（左特異ベクトル）
        - $\Sigma$ は $m \times n$ の対角行列（特異値を対角成分に持つ）
        - $V^T$ は $n \times n$ の直交行列の転置（右特異ベクトル）

        ## 両者の関係

        1. 行列 $A^T A$ の固有ベクトルは、$A$ の右特異ベクトル $V$ と一致する
        2. $A A^T$ の固有ベクトルは、$A$ の左特異ベクトル $U$ と一致する
        3. $A^T A$ および $A A^T$ の固有値は $\sigma_i^2$ となる（ここで $\sigma_i$ は $A$ の特異値）

        数式を見ると：

        $$A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^T \Sigma V^T = V \Sigma^2 V^T$$

        同様に：

        $$A A^T = U \Sigma V^T V \Sigma^T U^T = U \Sigma \Sigma^T U^T = U \Sigma^2 U^T$$

        これらは固有値分解の形式になっていて、$\Sigma^2$ が固有値を対角成分に持つ行列。このように、$A^T A$ の固有ベクトルが $V$ と一致し、$A A^T$ の固有ベクトルが $U$ と一致する。



        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
