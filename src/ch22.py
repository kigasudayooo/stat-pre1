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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
