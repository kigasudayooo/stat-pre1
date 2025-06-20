import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # 統計学入門ノートブック

        このノートブックでは、統計学の基本概念から始めて、統計検定準１級に必要な基礎知識を学習します。

        ## 目次
        1. 記述統計学の基礎
        2. 確率の基本概念
        3. 確率分布
        4. 統計的推定
        5. 統計的検定
        6. 相関と回帰
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats
    from scipy import stats
    import seaborn as sns
    
    # 日本語フォント設定
    try:
        import japanize_matplotlib
    except ImportError:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 表示設定
    np.random.seed(42)
    plt.style.use('default')
    return np, pd, plt, sns, stats


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. 記述統計学の基礎

        ### 1.1 データの種類
        - **量的データ**: 数値で表現される（身長、体重、気温など）
        - **質的データ**: カテゴリーで表現される（性別、血液型、職業など）

        ### 1.2 代表値（中心傾向の指標）
        """
    )
    return


@app.cell
def __(np, pd, plt):
    # サンプルデータ生成
    np.random.seed(42)
    sample_data = np.random.normal(50, 10, 1000)
    
    # 代表値の計算
    mean_val = np.mean(sample_data)
    median_val = np.median(sample_data)
    mode_val = stats.mode(sample_data.round()).mode[0]
    
    print(f"平均値 (Mean): {mean_val:.2f}")
    print(f"中央値 (Median): {median_val:.2f}")
    print(f"最頻値 (Mode): {mode_val:.2f}")
    
    # ヒストグラムで可視化
    plt.figure(figsize=(10, 6))
    plt.hist(sample_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均値: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中央値: {median_val:.2f}')
    plt.xlabel('値')
    plt.ylabel('頻度')
    plt.title('サンプルデータの分布と代表値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    sample_data, mean_val, median_val, mode_val
    return mean_val, median_val, mode_val, sample_data


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 1.3 散布度の指標
        
        データのばらつきを表す指標：
        - **範囲 (Range)**: 最大値 - 最小値
        - **分散 (Variance)**: $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$
        - **標準偏差 (Standard Deviation)**: $\sigma = \sqrt{\sigma^2}$
        - **四分位範囲 (IQR)**: Q3 - Q1
        """
    )
    return


@app.cell
def __(np, plt, sample_data):
    # 散布度の計算
    range_val = np.max(sample_data) - np.min(sample_data)
    variance_val = np.var(sample_data)
    std_val = np.std(sample_data)
    q1 = np.percentile(sample_data, 25)
    q3 = np.percentile(sample_data, 75)
    iqr_val = q3 - q1
    
    print(f"範囲 (Range): {range_val:.2f}")
    print(f"分散 (Variance): {variance_val:.2f}")
    print(f"標準偏差 (Standard Deviation): {std_val:.2f}")
    print(f"四分位範囲 (IQR): {iqr_val:.2f}")
    
    # 箱ひげ図
    plt.figure(figsize=(10, 6))
    plt.boxplot(sample_data, vert=False)
    plt.xlabel('値')
    plt.title('箱ひげ図')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    range_val, variance_val, std_val, iqr_val
    return iqr_val, q1, q3, range_val, std_val, variance_val


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. 確率の基本概念

        ### 2.1 確率の定義
        - **古典的確率**: $P(A) = \frac{\text{事象Aが起こる場合の数}}{\text{全ての場合の数}}$
        - **統計的確率**: $P(A) = \lim_{n \to \infty} \frac{\text{事象Aが起こった回数}}{n}$
        - **主観的確率**: 個人の信念や経験に基づく確率

        ### 2.2 確率の性質
        1. $0 \leq P(A) \leq 1$
        2. $P(\Omega) = 1$ (全事象の確率は1)
        3. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
        """
    )
    return


@app.cell
def __(np, plt):
    # サイコロの例
    def simulate_dice_roll(n_rolls):
        """サイコロ投げのシミュレーション"""
        rolls = np.random.randint(1, 7, n_rolls)
        return rolls
    
    # 大数の法則の確認
    n_trials = [10, 100, 1000, 10000]
    probabilities = []
    
    plt.figure(figsize=(12, 8))
    
    for i, n in enumerate(n_trials):
        rolls = simulate_dice_roll(n)
        prob_6 = np.sum(rolls == 6) / n
        probabilities.append(prob_6)
        
        plt.subplot(2, 2, i+1)
        plt.hist(rolls, bins=np.arange(0.5, 7.5, 1), alpha=0.7, color='lightblue', edgecolor='black')
        plt.title(f'試行回数: {n}, 6の確率: {prob_6:.3f}')
        plt.xlabel('出目')
        plt.ylabel('頻度')
        plt.xticks(range(1, 7))
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("大数の法則の確認:")
    for n, prob in zip(n_trials, probabilities):
        print(f"試行回数 {n}: 6の確率 = {prob:.4f}")
    
    probabilities
    return i, n, n_trials, prob_6, probabilities, rolls, simulate_dice_roll


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. 確率分布

        ### 3.1 離散確率分布

        #### ベルヌーイ分布
        - パラメータ: $p$ (成功確率)
        - 確率質量関数: $P(X = k) = p^k(1-p)^{1-k}$, $k \in \{0, 1\}$

        #### 二項分布
        - パラメータ: $n$ (試行回数), $p$ (成功確率)
        - 確率質量関数: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$
        """
    )
    return


@app.cell
def __(np, plt, stats):
    # 二項分布の例
    n, p = 20, 0.3
    x = np.arange(0, n+1)
    pmf = stats.binom.pmf(x, n, p)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, pmf, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('成功回数 (k)')
    plt.ylabel('確率')
    plt.title(f'二項分布 B({n}, {p})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 期待値と分散
    mean_binom = n * p
    var_binom = n * p * (1 - p)
    
    print(f"二項分布 B({n}, {p}):")
    print(f"期待値 E[X] = {mean_binom:.2f}")
    print(f"分散 Var(X) = {var_binom:.2f}")
    
    mean_binom, pmf, var_binom, x
    return mean_binom, pmf, var_binom, x


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 3.2 連続確率分布

        #### 正規分布
        - パラメータ: $\mu$ (平均), $\sigma^2$ (分散)
        - 確率密度関数: $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

        #### 標準正規分布
        - $\mu = 0$, $\sigma^2 = 1$ の正規分布
        - 標準化: $Z = \frac{X - \mu}{\sigma}$
        """
    )
    return


@app.cell
def __(np, plt, stats):
    # 正規分布の例
    mu, sigma = 0, 1
    x_norm = np.linspace(-4, 4, 1000)
    pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
    
    plt.figure(figsize=(12, 8))
    
    # 標準正規分布
    plt.subplot(2, 2, 1)
    plt.plot(x_norm, pdf_norm, 'b-', linewidth=2, label='N(0,1)')
    plt.fill_between(x_norm, pdf_norm, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('密度')
    plt.title('標準正規分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 異なるパラメータの正規分布
    plt.subplot(2, 2, 2)
    for mu_val, sigma_val, color, label in [(0, 1, 'blue', 'N(0,1)'), 
                                            (0, 2, 'red', 'N(0,4)'), 
                                            (2, 1, 'green', 'N(2,1)')]:
        x_range = np.linspace(-6, 6, 1000)
        pdf_vals = stats.norm.pdf(x_range, mu_val, sigma_val)
        plt.plot(x_range, pdf_vals, color=color, linewidth=2, label=label)
    plt.xlabel('x')
    plt.ylabel('密度')
    plt.title('異なったパラメータの正規分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 68-95-99.7 ルール
    plt.subplot(2, 2, 3)
    x_68 = np.linspace(-1, 1, 100)
    x_95 = np.linspace(-2, 2, 100)
    x_997 = np.linspace(-3, 3, 100)
    
    plt.plot(x_norm, pdf_norm, 'b-', linewidth=2)
    plt.fill_between(x_68, stats.norm.pdf(x_68, 0, 1), alpha=0.3, color='red', label='68% (μ±σ)')
    plt.fill_between(x_95, stats.norm.pdf(x_95, 0, 1), alpha=0.2, color='green', label='95% (μ±2σ)')
    plt.fill_between(x_997, stats.norm.pdf(x_997, 0, 1), alpha=0.1, color='blue', label='99.7% (μ±3σ)')
    plt.xlabel('x')
    plt.ylabel('密度')
    plt.title('68-95-99.7 ルール')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q プロット
    plt.subplot(2, 2, 4)
    sample_normal = np.random.normal(0, 1, 1000)
    stats.probplot(sample_normal, dist="norm", plot=plt)
    plt.title('Q-Q プロット（正規性の確認）')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 確率計算の例
    prob_68 = stats.norm.cdf(1, 0, 1) - stats.norm.cdf(-1, 0, 1)
    prob_95 = stats.norm.cdf(2, 0, 1) - stats.norm.cdf(-2, 0, 1)
    prob_997 = stats.norm.cdf(3, 0, 1) - stats.norm.cdf(-3, 0, 1)
    
    print(f"P(-1 ≤ X ≤ 1) = {prob_68:.4f}")
    print(f"P(-2 ≤ X ≤ 2) = {prob_95:.4f}")
    print(f"P(-3 ≤ X ≤ 3) = {prob_997:.4f}")
    
    pdf_norm, prob_68, prob_95, prob_997, sample_normal, x_norm
    return (
        color,
        label,
        mu_val,
        pdf_norm,
        pdf_vals,
        prob_68,
        prob_95,
        prob_997,
        sample_normal,
        sigma_val,
        x_68,
        x_95,
        x_997,
        x_norm,
        x_range,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 4. 統計的推定

        ### 4.1 点推定
        
        #### 推定量の性質
        1. **不偏性**: $E[\hat{\theta}] = \theta$
        2. **一致性**: $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$
        3. **有効性**: 分散が最小

        #### 推定方法
        - **最尤推定法**: 尤度関数を最大化
        - **モーメント法**: 標本モーメントと母集団モーメントを等しくおく
        """
    )
    return


@app.cell
def __(np, plt, stats):
    # 最尤推定の例（正規分布）
    def likelihood_normal(mu, sigma, data):
        """正規分布の尤度関数"""
        return np.prod(stats.norm.pdf(data, mu, sigma))
    
    def log_likelihood_normal(mu, sigma, data):
        """正規分布の対数尤度関数"""
        return np.sum(stats.norm.logpdf(data, mu, sigma))
    
    # サンプルデータ
    true_mu, true_sigma = 5, 2
    sample_size = 100
    sample = np.random.normal(true_mu, true_sigma, sample_size)
    
    # 最尤推定
    mle_mu = np.mean(sample)
    mle_sigma = np.sqrt(np.var(sample, ddof=0))  # 最尤推定では ddof=0
    
    print(f"真の値: μ = {true_mu}, σ = {true_sigma}")
    print(f"最尤推定値: μ̂ = {mle_mu:.3f}, σ̂ = {mle_sigma:.3f}")
    
    # 対数尤度関数の可視化
    mu_range = np.linspace(3, 7, 50)
    sigma_range = np.linspace(1, 3, 50)
    
    loglik_mu = [log_likelihood_normal(mu, true_sigma, sample) for mu in mu_range]
    loglik_sigma = [log_likelihood_normal(true_mu, sigma, sample) for sigma in sigma_range]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mu_range, loglik_mu, 'b-', linewidth=2)
    plt.axvline(mle_mu, color='red', linestyle='--', label=f'最尤推定値: {mle_mu:.3f}')
    plt.axvline(true_mu, color='green', linestyle='--', label=f'真の値: {true_mu}')
    plt.xlabel('μ')
    plt.ylabel('対数尤度')
    plt.title('μに関する対数尤度関数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sigma_range, loglik_sigma, 'b-', linewidth=2)
    plt.axvline(mle_sigma, color='red', linestyle='--', label=f'最尤推定値: {mle_sigma:.3f}')
    plt.axvline(true_sigma, color='green', linestyle='--', label=f'真の値: {true_sigma}')
    plt.xlabel('σ')
    plt.ylabel('対数尤度')
    plt.title('σに関する対数尤度関数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    likelihood_normal, log_likelihood_normal, loglik_mu, loglik_sigma, mle_mu, mle_sigma, mu_range, sample, sample_size, sigma_range, true_mu, true_sigma
    return (
        likelihood_normal,
        log_likelihood_normal,
        loglik_mu,
        loglik_sigma,
        mle_mu,
        mle_sigma,
        mu_range,
        sample,
        sample_size,
        sigma_range,
        true_mu,
        true_sigma,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 4.2 区間推定（信頼区間）

        母平均 $\mu$ の信頼区間（母分散 $\sigma^2$ が既知の場合）:
        
        $$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

        母平均 $\mu$ の信頼区間（母分散 $\sigma^2$ が未知の場合）:
        
        $$\bar{X} \pm t_{\alpha/2}(n-1) \frac{s}{\sqrt{n}}$$
        """
    )
    return


@app.cell
def __(np, plt, sample, stats):
    # 信頼区間の計算
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # 標本標準偏差（不偏推定）
    
    # 95%信頼区間（t分布）
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_of_error = t_critical * sample_std / np.sqrt(n)
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    print(f"標本サイズ: {n}")
    print(f"標本平均: {sample_mean:.3f}")
    print(f"標本標準偏差: {sample_std:.3f}")
    print(f"95%信頼区間: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"真の値 μ = {true_mu} は信頼区間に含まれる: {ci_lower <= true_mu <= ci_upper}")
    
    # 信頼区間の可視化
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=20, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    plt.axvline(sample_mean, color='red', linestyle='-', linewidth=2, label=f'標本平均: {sample_mean:.3f}')
    plt.axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label='95%信頼区間')
    plt.axvline(ci_upper, color='orange', linestyle='--', linewidth=2)
    plt.axvline(true_mu, color='green', linestyle='-', linewidth=2, label=f'真の値: {true_mu}')
    
    # 理論分布を重ねる
    x_theory = np.linspace(sample.min(), sample.max(), 1000)
    y_theory = stats.norm.pdf(x_theory, true_mu, true_sigma)
    plt.plot(x_theory, y_theory, 'g-', linewidth=2, alpha=0.7, label='理論分布')
    
    plt.xlabel('値')
    plt.ylabel('密度')
    plt.title('標本分布と95%信頼区間')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    alpha, ci_lower, ci_upper, confidence_level, margin_of_error, sample_mean, sample_std, t_critical, x_theory, y_theory
    return (
        alpha,
        ci_lower,
        ci_upper,
        confidence_level,
        margin_of_error,
        sample_mean,
        sample_std,
        t_critical,
        x_theory,
        y_theory,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 5. 統計的検定

        ### 5.1 検定の基本概念

        #### 仮説設定
        - **帰無仮説** $H_0$: 検定したい仮説（通常は「効果がない」）
        - **対立仮説** $H_1$: 帰無仮説に対する対案

        #### 検定の誤り
        - **第1種の誤り** (Type I Error): $H_0$ が真なのに棄却する（有意水準 $\alpha$）
        - **第2種の誤り** (Type II Error): $H_0$ が偽なのに採択する（確率 $\beta$）
        - **検出力** (Power): $1 - \beta$

        ### 5.2 一標本t検定
        
        $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$
        
        検定統計量: $t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}} \sim t(n-1)$
        """
    )
    return


@app.cell
def __(np, plt, sample, stats):
    # 一標本t検定の例
    mu_0 = 4.5  # 帰無仮説の値
    
    # 検定統計量の計算
    t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
    
    # p値の計算（両側検定）
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # 臨界値（有意水準5%）
    alpha_test = 0.05
    t_critical_test = stats.t.ppf(1 - alpha_test/2, df=n-1)
    
    print("一標本t検定")
    print(f"帰無仮説: μ = {mu_0}")
    print(f"対立仮説: μ ≠ {mu_0}")
    print(f"検定統計量: t = {t_stat:.3f}")
    print(f"自由度: {n-1}")
    print(f"p値: {p_value:.4f}")
    print(f"臨界値: ±{t_critical_test:.3f}")
    print(f"結論: {'帰無仮説を棄却' if p_value < alpha_test else '帰無仮説を採択'}")
    
    # t分布と検定統計量の可視化
    x_t = np.linspace(-4, 4, 1000)
    y_t = stats.t.pdf(x_t, df=n-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_t, y_t, 'b-', linewidth=2, label=f't分布 (df={n-1})')
    
    # 棄却域の表示
    x_reject_left = x_t[x_t <= -t_critical_test]
    x_reject_right = x_t[x_t >= t_critical_test]
    plt.fill_between(x_reject_left, stats.t.pdf(x_reject_left, df=n-1), 
                     alpha=0.3, color='red', label='棄却域')
    plt.fill_between(x_reject_right, stats.t.pdf(x_reject_right, df=n-1), 
                     alpha=0.3, color='red')
    
    # 検定統計量の位置
    plt.axvline(t_stat, color='orange', linestyle='-', linewidth=3, 
                label=f'検定統計量: {t_stat:.3f}')
    plt.axvline(-t_critical_test, color='red', linestyle='--', linewidth=2)
    plt.axvline(t_critical_test, color='red', linestyle='--', linewidth=2, 
                label=f'臨界値: ±{t_critical_test:.3f}')
    
    plt.xlabel('t値')
    plt.ylabel('密度')
    plt.title('一標本t検定の可視化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    alpha_test, mu_0, p_value, t_critical_test, t_stat, x_t, y_t
    return alpha_test, mu_0, p_value, t_critical_test, t_stat, x_t, y_t


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 6. 相関と回帰

        ### 6.1 相関係数

        ピアソンの相関係数:
        $$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

        - $-1 \leq r \leq 1$
        - $r > 0$: 正の相関
        - $r < 0$: 負の相関
        - $r = 0$: 無相関

        ### 6.2 単回帰分析

        回帰式: $y = a + bx + \varepsilon$

        - 回帰係数: $b = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$
        - 切片: $a = \bar{y} - b\bar{x}$
        """
    )
    return


@app.cell
def __(np, plt, stats):
    # 回帰分析の例
    np.random.seed(42)
    n_points = 100
    
    # データ生成
    x = np.random.uniform(0, 10, n_points)
    y = 2 + 3 * x + np.random.normal(0, 2, n_points)  # y = 2 + 3x + ε
    
    # 相関係数の計算
    correlation = np.corrcoef(x, y)[0, 1]
    
    # 回帰分析
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
    
    print(f"相関係数: r = {correlation:.3f}")
    print(f"回帰式: y = {intercept:.3f} + {slope:.3f}x")
    print(f"決定係数: R² = {r_value**2:.3f}")
    print(f"p値: {p_value_reg:.4f}")
    
    # 散布図と回帰直線
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, alpha=0.6, color='blue')
    plt.plot(x, intercept + slope * x, 'r-', linewidth=2, 
             label=f'y = {intercept:.2f} + {slope:.2f}x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('散布図と回帰直線')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差プロット
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('予測値')
    plt.ylabel('残差')
    plt.title('残差プロット')
    plt.grid(True, alpha=0.3)
    
    # 残差の正規性確認
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差のQ-Qプロット')
    plt.grid(True, alpha=0.3)
    
    # 残差のヒストグラム
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('頻度')
    plt.title('残差の分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 予測区間の可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, color='blue', label='観測値')
    
    # 回帰直線
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='回帰直線')
    
    # 予測区間（簡単な計算）
    mse = np.mean(residuals**2)
    se = np.sqrt(mse)
    plt.fill_between(x_line, y_line - 1.96*se, y_line + 1.96*se, 
                     alpha=0.2, color='red', label='95%予測区間（近似）')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('回帰分析と予測区間')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    correlation, intercept, mse, n_points, p_value_reg, r_value, residuals, se, slope, std_err, x, x_line, y, y_line, y_pred
    return (
        correlation,
        intercept,
        mse,
        n_points,
        p_value_reg,
        r_value,
        residuals,
        se,
        slope,
        std_err,
        x,
        x_line,
        y,
        y_line,
        y_pred,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## まとめ

        このノートブックでは、統計学の基本的な概念を学習しました：

        1. **記述統計**: データの要約と可視化
        2. **確率論**: 確率の基本概念と性質
        3. **確率分布**: 離散分布と連続分布
        4. **統計的推定**: 点推定と区間推定
        5. **統計的検定**: 仮説検定の基本
        6. **回帰分析**: 相関と回帰の基礎

        ### 次のステップ
        - より高度な統計手法（多変量解析、時系列分析など）
        - ベイズ統計学
        - 機械学習との関連
        - 実際のデータ分析プロジェクト

        ### 参考文献
        - 統計学入門（東京大学出版会）
        - 現代数理統計学の基礎（久保川達也）
        - Introduction to Statistical Learning
        """
    )
    return


if __name__ == "__main__":
    app.run()