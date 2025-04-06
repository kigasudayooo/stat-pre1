import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    import japanize_matplotlib
    return LinearRegression, japanize_matplotlib, np, pd, plt, sns, stats


@app.cell
def _(LinearRegression, np, plt):

    # 再現性のために乱数シードを設定
    np.random.seed(42)

    # シミュレーションデータの作成
    def generate_data(n=50, outlier=True):
        """
        基本的な線形回帰データを生成し、オプションで外れ値を追加
        """
        # 基本データを生成
        X = np.random.uniform(0, 10, n)
        # 真の関係: y = 2*x + 3 + ノイズ
        y = 2 * X + 3 + np.random.normal(0, 1, n)
    
        # 外れ値を追加（オプション）
        if outlier:
            # 説明変数空間で離れた位置に外れ値を追加（高leverage点）
            X = np.append(X, 15)
            # パターン1: 回帰線上に近い値（残差小）
            # y = np.append(y, 2 * 15 + 3 + np.random.normal(0, 1))
            # パターン2: 回帰線から離れた値（残差大）
            y = np.append(y, 2 * 15 + 3 + 10)  # 大きな残差を持つ外れ値
    
        return X.reshape(-1, 1), y  # sklearn用に形状を整える

    # leverageとクック距離を計算
    def compute_diagnostics(X, y):
        """
        線形回帰モデルの診断指標を計算
        - leverage (hat diagonal)
        - 残差
        - クック距離
        """
        # 線形回帰モデルをフィット
        model = LinearRegression()
        model.fit(X, y)
    
        # 予測値と残差を計算
        y_pred = model.predict(X)
        residuals = y - y_pred
    
        # Leverage (Hat行列の対角成分)の計算
        # H = X(X^TX)^{-1}X^T
        X_mat = np.array(X)
        n = len(X)
        hat_matrix = X_mat.dot(np.linalg.inv(X_mat.T.dot(X_mat))).dot(X_mat.T)
        leverage = np.diagonal(hat_matrix)
    
        # 標準化残差の計算
        mse = np.sum(residuals**2) / (n - X.shape[1] - 1)
        std_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
        # クック距離の計算
        p = X.shape[1] + 1  # パラメータ数（切片を含む）
        cooks_distance = (std_residuals**2 / p) * (leverage / (1 - leverage))
    
        return {
            'X': X.flatten(),  # 可視化用にflatten
            'y': y,
            'leverage': leverage,
            'residuals': residuals,
            'std_residuals': std_residuals,
            'cooks_distance': cooks_distance,
            'threshold_leverage': 2 * p / n,  # leverageのしきい値
            'threshold_cooks': 4 / n  # クック距離のしきい値
        }

    # データを可視化
    def plot_diagnostics(results, title="診断結果"):
        """
        診断結果を可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)
    
        # 1. データと回帰線
        ax = axes[0, 0]
        ax.scatter(results['X'], results['y'], alpha=0.7)
    
        # 回帰線を描画
        X_sorted = np.sort(results['X'])
        model = LinearRegression().fit(results['X'].reshape(-1, 1), results['y'])
        y_pred = model.predict(X_sorted.reshape(-1, 1))
        ax.plot(X_sorted, y_pred, 'r-', label='回帰線')
    
        # 影響点を強調表示
        influential = results['cooks_distance'] > results['threshold_cooks']
        if np.any(influential):
            ax.scatter(results['X'][influential], results['y'][influential], 
                      s=200, facecolors='none', edgecolors='red', label='影響点')
    
        ax.set_title('データと回帰線')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
    
        # 2. Leverage vs 標準化残差
        ax = axes[0, 1]
        sc = ax.scatter(results['leverage'], results['std_residuals'], 
                       c=results['cooks_distance'], cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('クック距離')
    
        # しきい値ライン
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=results['threshold_leverage'], color='red', linestyle='--', alpha=0.3)
    
        # 点の番号をプロット
        for i, (x, y) in enumerate(zip(results['leverage'], results['std_residuals'])):
            ax.annotate(str(i), (x, y), fontsize=8)
    
        ax.set_title('Leverage vs 標準化残差')
        ax.set_xlabel('Leverage (ヘッセ行列の対角成分)')
        ax.set_ylabel('標準化残差')
    
        # 3. クック距離のバープロット
        ax = axes[1, 0]
        ax.bar(range(len(results['cooks_distance'])), results['cooks_distance'])
        ax.axhline(y=results['threshold_cooks'], color='red', linestyle='--', label=f'しきい値 ({results["threshold_cooks"]:.3f})')
    
        # 点の番号をプロット
        for i, d in enumerate(results['cooks_distance']):
            ax.annotate(str(i), (i, d), fontsize=8)
    
        ax.set_title('クック距離')
        ax.set_xlabel('データポイント')
        ax.set_ylabel('クック距離')
        ax.legend()
    
        # 4. レバレッジとクック距離の関係
        ax = axes[1, 1]
        leverage_term = results['leverage'] / (1 - results['leverage'])**2
        ax.scatter(leverage_term, results['cooks_distance'], alpha=0.7)
    
        # 理論的関係を示す直線（シンプルな場合）
        x_range = np.linspace(min(leverage_term), max(leverage_term), 100)
        # 仮想的にresiduals^2/p*MSEの平均値で線を引く（視覚的な参考）
        # X.shape[1] ではなく、保存されているthreshold_leverageから逆算して求める
        n = int(2 / results['threshold_leverage'])  # 2*p/n から n を求める
        p = 2  # 線形回帰なので、傾きと切片の2つのパラメータ
        avg_std_resid_term = np.mean(results['std_residuals']**2) / p
        ax.plot(x_range, avg_std_resid_term * x_range, 'r--', 
                label=f'理論的関係 (slope={avg_std_resid_term:.3f})')
    
        # 点の番号をプロット
        for i, (x, y) in enumerate(zip(leverage_term, results['cooks_distance'])):
            ax.annotate(str(i), (x, y), fontsize=8)
    
        ax.set_title('レバレッジ項とクック距離の関係')
        ax.set_xlabel('レバレッジ項: h_ii/(1-h_ii)²')
        ax.set_ylabel('クック距離')
        ax.legend()
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig

    # シミュレーションを実行

    # 1. 基本データセット（外れ値なし）
    X_base, y_base = generate_data(n=50, outlier=False)
    results_base = compute_diagnostics(X_base, y_base)
    fig_base = plot_diagnostics(results_base, "基本データセット（外れ値なし）")

    # 2. 外れ値を含むデータセット
    X_out, y_out = generate_data(n=50, outlier=True)
    results_out = compute_diagnostics(X_out, y_out)
    fig_out = plot_diagnostics(results_out, "外れ値を含むデータセット")

    # 3. 理論的関係を検証（カスタムデータセット）
    # 異なるleverage値と残差を持つデータポイントを作成
    def create_custom_data():
        # 基本データ
        X = np.linspace(0, 10, 30).reshape(-1, 1)
        y_true = 2 * X.flatten() + 3
    
        # 残差を調整（標準正規分布から）
        residuals = np.random.normal(0, 1, 30)
    
        # leverageを人為的に調整するため、Xの一部を極端な値に
        X[25] = 20  # 高leverage点1 (離れた位置)
        X[26] = 25  # 高leverage点2 (さらに離れた位置)
        X[27] = 15  # 高leverage点3
    
        # 残差も人為的に調整
        residuals[25] = 0.5   # 小さな残差
        residuals[26] = 5.0   # 大きな残差
        residuals[27] = -3.0  # 中程度の負の残差
    
        # データセット作成
        y = y_true + residuals
    
        return X, y

    X_custom, y_custom = create_custom_data()
    results_custom = compute_diagnostics(X_custom, y_custom)
    fig_custom = plot_diagnostics(results_custom, "カスタムデータセット（レバレッジと残差の関係を検証）")

    # 結果を表示
    plt.show()
    return (
        X_base,
        X_custom,
        X_out,
        compute_diagnostics,
        create_custom_data,
        fig_base,
        fig_custom,
        fig_out,
        generate_data,
        plot_diagnostics,
        results_base,
        results_custom,
        results_out,
        y_base,
        y_custom,
        y_out,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
