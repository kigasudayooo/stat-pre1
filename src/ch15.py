import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import japanize_matplotlib

    def simulate_poisson_process(rate, T):
        """
        rate: 強度パラメータλ
        T: シミュレーション時間
        """
        times = []
        t = 0

        while t < T:
            # 指数分布から次の時間間隔を生成
            dt = np.random.exponential(1/rate)
            t += dt

            if t < T:
                times.append(t)

        return times

    # シミュレーション例
    events = simulate_poisson_process(rate=5, T=10)
    print(f"発生時刻: {[round(t, 2) for t in events]}")
    print(f"発生回数: {len(events)}")

    # プロット
    plt.figure(figsize=(10, 2))
    plt.scatter(events, [1]*len(events), marker='|', s=100)
    plt.xlabel('時間')
    plt.yticks([])
    plt.title('ポワソン過程のシミュレーション (λ=5)')
    plt.show()
    return events, japanize_matplotlib, np, plt, simulate_poisson_process


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
