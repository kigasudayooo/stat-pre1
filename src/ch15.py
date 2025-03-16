import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo


    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    import japanize_matplotlib

    return japanize_matplotlib, mo, np, plt, stats


app._unparsable_cell(
    r"""
    こんにちは！確率過程とブラウン運動について教えるね。じゃあ、友達同士で話すように進めていくよ。

    # 確率過程について

    確率過程って、簡単に言うと「時間とともに変化する確率的な現象」のことなんだ。例えば、株価の変動、気温の変化、粒子の動き、人の行列の長さなど、時間によって予測不可能な形で変化するものを数学的に表現するための道具なんだよ。

    ## 確率過程の基本的な考え方

    確率過程は、各時点 $t$ において確率変数 $X(t)$ が対応しているよ。これを数式で表すと：

    $\{X(t), t \in T\}$

    ここで $T$ は時間の集合（離散的か連続的か）を表しているんだ。

    例えば、毎日のサイコロを振る結果を記録するなら離散的な時間 $T = \{1, 2, 3, ...\}$ になるし、粒子の動きをずっと追うなら連続的な時間 $T = [0, \infty)$ になるよね。

    ## 確率過程の種類

    確率過程にはいろんな種類があるんだ：

    1. **マルコフ過程**：過去の状態は関係なく、現在の状態だけで将来が決まるような過程
    2. **定常過程**：統計的性質が時間によって変わらない過程
    3. **マルチンゲール**：期待値が一定の確率過程
    4. **ポアソン過程**：ランダムな事象の発生回数を表す過程
    5. **ブラウン運動**：連続的で細かいランダムな動きを表す過程

    # ブラウン運動について詳しく

    ブラウン運動（またはウィーナー過程とも呼ばれる）は、確率過程の中でも特に重要な位置を占めているんだ。これは1827年に植物学者のロバート・ブラウンが、水に浮かぶ花粉粒子がランダムに動く現象を観察したことに由来しているよ。

    ## ブラウン運動の定義

    数学的には、ブラウン運動 $B(t)$ は以下の性質を持つ確率過程として定義されるんだ：

    1. $B(0) = 0$ （スタート地点はゼロ）
    2. $B(t)$ は独立増分を持つ（異なる時間間隔での変化量は互いに独立）
    3. 任意の $0 \leq s < t$ に対して、増分 $B(t) - B(s)$ は平均0、分散 $t-s$ の正規分布に従う
    4. $B(t)$ は確率1で連続な経路を持つ（途中で飛びはない）

    これを数式で表すと、増分 $B(t) - B(s)$ は次の分布に従うよ：

    $B(t) - B(s) \sim N(0, t-s)$

    ここで $N(\mu, \sigma^2)$ は平均 $\mu$、分散 $\sigma^2$ の正規分布を表しているんだ。

    ## ブラウン運動の性質

    ブラウン運動には興味深い性質がたくさんあるよ：

    1. **自己相似性**：時間をスケーリングすると、確率分布が同じになる
       $B(at) \sim \sqrt{a} \cdot B(t)$

    2. **非差分可能性**：経路は連続だけど、どの点でも微分不可能（ザラザラしている）

    3. **二次変分**：時間区間 $[0,t]$ での二次変分は $t$ に等しい

    4. **マルチンゲール性**：ブラウン運動は平均0のマルチンゲール

    ## ブラウン運動の例

    身近な例を挙げると：

    1. **金融市場**：株価の変動はブラウン運動でモデル化されることが多いよ（厳密には幾何ブラウン運動）

    2. **拡散現象**：インクが水に広がる様子、熱の伝導などはブラウン運動と関連している

    3. **ノイズ**：電子回路のノイズなどもブラウン運動でモデル化できる

    ## ブラウン運動のシミュレーション方法

    ブラウン運動をシミュレーションするのは実は簡単なんだ！時間を細かく分割して、各ステップで正規分布に従う乱数を足していくだけ：

    1. 時間区間 $[0,T]$ を $n$ 等分する：$\Delta t = T/n$
    2. 各ステップで $\Delta B_i \sim N(0, \Delta t)$ の乱数を生成
    3. $B(t_i) = B(t_{i-1}) + \Delta B_i$ として累積していく

    Python でコードを書くとこんな感じになるよ：

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def brownian_motion(T=1, N=1000, paths=1):
        dt = T/N
        # 正規分布からランダムな増分を生成
        dB = np.random.normal(0, np.sqrt(dt), size=(paths, N))
        # 0から始まるようにするために、最初に0の列を追加
        B = np.hstack([np.zeros((paths, 1)), np.cumsum(dB, axis=1)])
        # 時間の格子点
        t = np.linspace(0, T, N+1)
        return t, B

    # シミュレーション実行
    t, paths = brownian_motion(T=1, N=1000, paths=5)

    # プロット
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(t, paths[i])
    plt.title('ブラウン運動のシミュレーション')
    plt.xlabel('時間')
    plt.ylabel('位置')
    plt.grid(True)
    plt.show()
    ```

    ## ブラウン運動の応用

    ブラウン運動は理論だけじゃなく、実用的な応用もたくさんあるんだ：

    1. **ブラック・ショールズモデル**：オプション価格を計算するための金融モデル

    2. **フォッカー・プランク方程式**：物理学での粒子の分布の時間発展を記述

    3. **カルマンフィルタ**：ノイズのある測定から状態を推定する方法

    4. **伊藤積分**：確率微分方程式を解くための積分

    ## 確率微分方程式とブラウン運動

    ブラウン運動を使うと、ランダムな要素を含む微分方程式である「確率微分方程式（SDE）」を扱えるようになるよ。一般的な形は：

    $dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dB_t$

    ここで：
    - $\mu(X_t, t)$ はドリフト項（確定的な変化）
    - $\sigma(X_t, t)$ は拡散項（ランダムな変化の大きさ）
    - $dB_t$ はブラウン運動の微小変化

    例えば、株価のモデルとしてよく使われる幾何ブラウン運動は：

    $dS_t = \mu S_t dt + \sigma S_t dB_t$

    これは株価 $S_t$ が平均的に率 $\mu$ で成長し、ボラティリティ $\sigma$ でランダムに変動することを表しているんだ。

    ## まとめ

    確率過程は不確実性を持つ時間変化現象を扱うための数学的な枠組みで、ブラウン運動はその中でも特に重要な基本的なモデルなんだ。ブラウン運動は：

    - 連続だけど微分不可能な経路を持つ
    - 正規分布に従う独立な増分を持つ
    - 金融、物理学、工学など様々な分野で応用されている

    いかがだった？もっと具体的な部分について知りたいことがあれば、遠慮なく聞いてね！例えば、マルチンゲールについて詳しく知りたいとか、確率微分方程式のシミュレーション方法とか、他の確率過程について知りたいとか、何でも対応するよ！
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(np, plt):


    def brownian_motion(T=1, N=1000, paths=1):
        dt = T/N
        # 正規分布からランダムな増分を生成
        dB = np.random.normal(0, np.sqrt(dt), size=(paths, N))
        # 0から始まるようにするために、最初に0の列を追加
        B = np.hstack([np.zeros((paths, 1)), np.cumsum(dB, axis=1)])
        # 時間の格子点
        t = np.linspace(0, T, N+1)
        return t, B

    # シミュレーション実行
    t, paths = brownian_motion(T=1, N=1000, paths=5)

    # プロット
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(t, paths[i])
    plt.title('ブラウン運動のシミュレーション')
    plt.xlabel('時間')
    plt.ylabel('位置')
    plt.grid(True)
    plt.show()
    return brownian_motion, i, paths, t


app._unparsable_cell(
    r"""
    もちろん！ブラウン運動の性質についてより丁寧に説明するね。数学的な用語の意味も含めて、証明のアイデアも示していくよ。

    # ブラウン運動の性質と証明

    ## 1. 自己相似性（Self-similarity）

    **用語の意味**:  
    自己相似性とは、時間をスケーリングしても確率的な性質が変わらないという性質だよ。具体的には、任意の定数 $a > 0$ に対して：

    $\{B(at), t \geq 0\} \stackrel{d}{=} \{\sqrt{a} \cdot B(t), t \geq 0\}$

    ここで $\stackrel{d}{=}$ は「確率分布が等しい」ことを表しているんだ。

    **証明のアイデア**:  
    両辺の確率過程が同じ確率分布を持つことを示せばいいんだ。新しい過程 $X(t) = B(at)/\sqrt{a}$ を考えると：

    1. $X(0) = B(0)/\sqrt{a} = 0$（原点からスタート）
    2. 任意の $0 \leq s < t$ に対して増分：
       $X(t) - X(s) = \frac{B(at) - B(as)}{\sqrt{a}}$

    この増分は正規分布 $N(0, \frac{at-as}{a}) = N(0, t-s)$ に従う。

    3. 増分の独立性も保たれる
    4. 経路の連続性も保たれる

    これらはブラウン運動の定義と一致するから、$X(t)$ もブラウン運動。つまり、$B(at) \stackrel{d}{=} \sqrt{a} \cdot B(t)$ が示せたよ。

    この性質は、ブラウン運動のグラフを「ズームイン」しても同じような不規則なパターンが見えることを意味している。フラクタル的な性質と言えるね。

    ## 2. 非差分可能性（Non-differentiability）

    **用語の意味**:  
    ブラウン運動の経路は至るところ連続だけど、どの点でも微分不可能なんだ。つまり、どんなに小さなスケールで見ても「なめらか」ではなく、常にギザギザしている。

    **証明のアイデア**:  
    $t$ での微分係数が存在すると仮定して矛盾を導くよ。微分係数が存在するなら：

    $\lim_{h \to 0} \frac{B(t+h) - B(t)}{h}$

    が存在するはず。しかし、自己相似性から、$h$ が小さくなるほど $\frac{B(t+h) - B(t)}{h}$ の分散は $\frac{h}{h^2} = \frac{1}{h}$ に比例して大きくなる。$h \to 0$ のとき、分散は無限大になるため、極限は存在しない。

    実は、もっと強い結果が成り立っていて、ブラウン運動の経路は確率1でヘルダー連続（Hölder continuous）の指数が $\frac{1}{2}-\varepsilon$（任意の小さな $\varepsilon > 0$ について）であることが示せる。これは経路が「ほぼ $\frac{1}{2}$ 次で連続」だが、それよりも滑らかではないことを意味しているんだ。

    ## 3. 二次変分（Quadratic Variation）

    **用語の意味**:  
    時間区間 $[0,t]$ を $n$ 等分して、その分割点を $0 = t_0 < t_1 < \cdots < t_n = t$ とする。このとき、ブラウン運動の二次変分は：

    $\sum_{i=1}^{n} [B(t_i) - B(t_{i-1})]^2$

    分割の最大幅を0に近づけるとき、この和は確率1で $t$ に収束する。

    **証明のアイデア**:  
    各増分 $B(t_i) - B(t_{i-1})$ は正規分布 $N(0, t_i - t_{i-1})$ に従うから、その二乗の期待値は $t_i - t_{i-1}$ になる。

    $E[(B(t_i) - B(t_{i-1}))^2] = t_i - t_{i-1}$

    二次変分の期待値は：

    $E\left[\sum_{i=1}^{n} [B(t_i) - B(t_{i-1})]^2\right] = \sum_{i=1}^{n} (t_i - t_{i-1}) = t$

    これは分割に依存しない。中心極限定理と大数の法則を使って、分割を細かくすると二次変分が確率1で $t$ に収束することが示せる。

    この性質は金融数学でとても重要で、例えば伊藤積分（Itô integral）の定義やブラック・ショールズ方程式の導出に使われるよ。

    ## 4. マルチンゲール性（Martingale Property）

    **用語の意味**:  
    マルチンゲールとは、「将来の条件付き期待値が現在の値に等しい」確率過程のことだよ。つまり、過去の情報をすべて知っていても、将来の期待値は現在値から変わらない。

    ブラウン運動 $B(t)$ は平均0のマルチンゲールであり、数式で表すと：

    $E[B(t) | \mathcal{F}_s] = B(s)$ for all $0 \leq s < t$

    ここで $\mathcal{F}_s$ は時刻 $s$ までの情報をすべて含むフィルトレーション（情報の集合）だよ。

    **証明のアイデア**:  
    増分の独立性と平均0の性質から：

    $E[B(t) | \mathcal{F}_s] = E[B(t) - B(s) + B(s) | \mathcal{F}_s]$
    $= E[B(t) - B(s) | \mathcal{F}_s] + E[B(s) | \mathcal{F}_s]$
    $= E[B(t) - B(s)] + B(s)$ （増分は過去と独立）
    $= 0 + B(s) = B(s)$ （増分の平均は0）

    これでマルチンゲール性が示せたね。この性質は確率論や金融数学で重要で、例えば無裁定価格理論の基礎になっているよ。

    ## 5. リフレクション原理（Reflection Principle）

    **用語の意味**:  
    リフレクション原理は、ブラウン運動が特定のレベル $a$ に到達する確率に関する性質だよ。「最初にレベル $a$ に到達した後のブラウン運動の経路を、そのレベルに関して反射させた経路」も同じ確率でブラウン運動になる。

    これにより、ブラウン運動が区間 $[0,t]$ で特定のレベル $a > 0$ に到達する確率が計算できるんだ：

    $P(\max_{0 \leq s \leq t} B(s) \geq a) = 2P(B(t) \geq a) = 2(1 - \Phi(a/\sqrt{t}))$

    ここで $\Phi$ は標準正規分布の累積分布関数だよ。

    **証明のアイデア**:  
    停止時刻 $\tau_a = \inf\{s \geq 0: B(s) = a\}$ を定義する（初めて $a$ に到達する時間）。
    ブラウン運動が時間 $t$ までに $a$ に到達するイベントを考える。

    このとき、新しいプロセス：

    $\tilde{B}(s) = 
    \begin{cases} 
    B(s) & \text{if } s < \tau_a \\
    2a - B(s) & \text{if } s \geq \tau_a
    \end{cases}$

    も同じ確率法則に従うことを示せる。この反射を使って、最大値の分布が計算できるんだ。

    ## 6. ストロング・マルコフ性（Strong Markov Property）

    **用語の意味**:  
    通常のマルコフ性は「未来の挙動は現在の状態だけで決まり、過去の履歴には依存しない」という性質だけど、ストロング・マルコフ性はこれを一般化したもので、「停止時刻」と呼ばれるランダムな時間でも成り立つよ。

    ブラウン運動 $B(t)$ と停止時刻 $\tau$ に対して、$\{B(t+\tau) - B(\tau), t \geq 0\}$ は元のブラウン運動と同じ確率法則に従い、$\tau$ までの情報と独立だ。

    **証明のアイデア**:  
    離散的な近似を使って、停止時刻が有限値の集合しか取らない場合を考え、それを一般の場合に拡張するという方法が使われるよ。証明の詳細は技術的だけど、直感的には「どのタイミングで見始めても、そこからのブラウン運動の振る舞いは同じ」ということを意味している。

    ## 7. スケーリング則（Scaling Law）

    **用語の意味**:  
    任意の定数 $c > 0$ に対して、過程 $\{X(t) = \frac{1}{c}B(c^2t), t \geq 0\}$ もブラウン運動になる。これはブラウン運動の時間と空間のスケーリングに関する性質だよ。

    **証明のアイデア**:  
    自己相似性の証明と同様に、新しい過程 $X(t)$ がブラウン運動の定義を満たすことを確認すればいいんだ。増分の正規性、独立性、連続性などを確認していく。

    ## 8. 保存則（Conservation Laws）

    **用語の意味**:  
    ブラウン運動には様々な保存則があり、代表的なものは「伊藤の公式」に関連する。例えば：

    $d(B^2(t)) = dt + 2B(t)dB(t)$

    これは普通の微積分の連鎖律の確率版と考えられるよ。

    **証明のアイデア**:  
    伊藤の公式を使って導出する。確率過程 $f(B(t),t)$ に対する伊藤の公式は：

    $df(B(t),t) = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial x}dB(t) + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}dt$

    $f(x,t) = x^2$ のケースを考えると上記の公式が得られる。

    ## 9. 標本経路の性質（Properties of Sample Paths）

    **用語の意味**:  
    ブラウン運動の標本経路（実現された１つの軌跡）は以下の性質を持つ：

    - ほぼすべての標本経路は連続だが、微分不可能
    - 局所的な最大値と最小値が無数にある
    - 標本経路はノーホエア密度（nowhere dense）：どの区間においても、無限回正と負を行き来する
    - 標本経路は零集合（set of measure zero）上で変化する

    **証明のアイデア**:  
    これらの性質の証明には、ブラウン運動の定義から導かれる確率的な性質と、実解析や測度論の道具が使われるよ。例えば、連続性はコルモゴロフの連続性定理から導かれる。

    ## 10. 相加ゼロクロス（Zero Crossings）

    **用語の意味**:  
    時間区間 $[0,t]$ においてブラウン運動が0をクロスする回数は無限大になる。これは、どんなに小さな時間区間でも、ブラウン運動は0を無限回横切ることを意味しているよ。

    **証明のアイデア**:  
    背理法を使って、有限回しか0をクロスしないと仮定して矛盾を導く。自己相似性とマルコフ性を使って、任意の小さな時間間隔でも0をクロスする確率が正であることを示し、これが無限回起こることを証明する。

    ## 11. ブラウン運動の期待値と分散

    ブラウン運動 $B(t)$ の期待値と分散は：

    $E[B(t)] = 0$
    $Var[B(t)] = t$

    これは定義から直接導かれる基本的な性質だね。

    **証明**:  
    ブラウン運動の定義から、$B(0) = 0$ で、増分 $B(t) - B(0)$ は $N(0,t)$ に従うから：

    $E[B(t)] = E[B(t) - B(0)] = 0$
    $Var[B(t)] = Var[B(t) - B(0)] = t$

    ## まとめ

    ブラウン運動の性質は互いに深く関連していて、確率論の美しさを表しているよ。これらの性質は金融数学、物理学、工学など様々な分野で応用されているんだ。

    例えば：
    - 自己相似性は様々なスケールでの現象のモデル化に使われる
    - 二次変分と伊藤積分は確率微分方程式の解法に重要
    - マルチンゲール性は金融商品の価格付けの理論基盤
    - リフレクション原理は初到達時間問題の解決に使われる

    それぞれの性質について、もっと詳しく知りたい部分はある？より深い数学的な証明や特定の応用について教えることもできるよ！
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ヘルダー連続（Hölder continuity）について説明するね！

        # ヘルダー連続性とは？

        ヘルダー連続性は、関数がどれくらい「滑らか」かを測る概念なんだ。通常の連続性よりも強い条件で、微分可能性よりは弱い条件になるよ。数学的に定義すると：

        ## 定義

        関数 $f: X \rightarrow Y$（ここで $X$ と $Y$ は距離空間）がヘルダー連続であるとは、あるパラメータ $\alpha \in (0,1]$ と定数 $C > 0$ が存在して、任意の $x, y \in X$ に対して以下の条件を満たすことを言うんだ：

        $$d_Y(f(x), f(y)) \leq C \cdot d_X(x, y)^\alpha$$

        ここで $d_X$ と $d_Y$ はそれぞれの空間での距離関数だよ。

        ## 直感的な意味

        - $\alpha = 1$ のとき：これはリプシッツ連続（Lipschitz continuity）と呼ばれ、関数の変化率が一定以下であることを意味する
        - $\alpha < 1$ のとき：関数は連続だけど、点と点の間の距離が小さくなるにつれて、関数値の差は距離の $\alpha$ 乗に比例して小さくなる

        $\alpha$ が大きいほど関数は「滑らか」になる。逆に $\alpha$ が小さいほど関数は「荒く」なるよ。

        ## ブラウン運動とヘルダー連続性

        ブラウン運動の標本経路は確率1で「$\alpha < 1/2$ のヘルダー連続」であることが知られているんだ。具体的には、任意の $\alpha < 1/2$ に対して、ある定数 $C > 0$ が存在して：

        $$|B(t) - B(s)| \leq C |t-s|^\alpha$$

        が確率1で成り立つ。

        でも、$\alpha = 1/2$ ではヘルダー連続ではない！ つまり、どんな定数 $C$ をとっても：

        $$|B(t) - B(s)| \leq C |t-s|^{1/2}$$

        は確率1で満たされない時間の組 $(s,t)$ が存在するんだ。

        ## 証明のアイデア

        ブラウン運動がヘルダー連続であることの証明は、コルモゴロフの連続性定理を使うよ。この定理は、確率過程の増分の期待値に関する条件から経路の連続性を導き出すものなんだ。

        ブラウン運動の増分に対して：

        $$E[|B(t) - B(s)|^p] = E[|N(0, t-s)|^p] = C_p \cdot |t-s|^{p/2}$$

        ここで $C_p$ は $p$ に依存する定数。これをコルモゴロフの定理に適用すると、任意の $\alpha < p/2 - 1/p$ についてヘルダー連続であることが分かる。$p$ を十分大きくとれば、$\alpha$ を $1/2$ に任意に近づけることができるけど、厳密に $1/2$ にはできないんだ。

        ## ヘルダー連続性の例

        具体的な例をいくつか挙げると：

        1. **滑らかな関数**: $f(x) = x^2$ は $\alpha = 1$ でヘルダー連続（リプシッツ連続）
        2. **ルート関数**: $f(x) = \sqrt{x}$ は $\alpha = 1/2$ でヘルダー連続
        3. **ブラウン運動**: 確率1で $\alpha < 1/2$ でヘルダー連続
        4. **フラクタル関数**: ワイエルシュトラス関数などのフラクタル関数は特定の $\alpha$ でのみヘルダー連続

        ## ヘルダー連続性の応用

        ヘルダー連続性は単に数学的な概念にとどまらず、様々な応用があるよ：

        1. **偏微分方程式論**: 解の正則性（滑らかさ）を調べるのに使われる
        2. **フラクタル理論**: フラクタル次元を特徴づける
        3. **数値解析**: 近似アルゴリズムの収束率を解析する
        4. **確率過程**: ブラウン運動などの確率過程の経路の特性を記述する

        ## ブラウン運動が正確に $\alpha = 1/2$ でヘルダー連続でない理由

        ブラウン運動が正確に $\alpha = 1/2$ でヘルダー連続ではないことは、「反復対数の法則」という結果から導かれるんだ。この法則によると：

        $$\limsup_{h \to 0} \frac{|B(t+h) - B(t)|}{\sqrt{2h \log \log(1/h)}} = 1$$

        が確率1で成立する。これは、十分小さな $h$ に対して、$|B(t+h) - B(t)|$ は $\sqrt{h}$ より若干大きくなる場合があることを示している。もし $\alpha = 1/2$ でヘルダー連続ならば、上記の極限は有限であるはずだけど、実際には1に収束するから矛盾するんだね。

        ## まとめ

        ヘルダー連続性は関数の「滑らかさ」を測る重要な概念で、ブラウン運動の経路は「ほぼ $1/2$ 次で連続」だけれども、厳密には $1/2$ 未満のヘルダー指数しか持たないということが分かるよ。これは、ブラウン運動の経路が「予想よりも荒い」ことを数学的に表現していて、その非常に特殊な性質を浮き彫りにしているんだ。

        何か他にも質問があれば、遠慮なく聞いてね！
        """
    )
    return


app._unparsable_cell(
    r"""
    もちろん！独立定常増分（Independent and Stationary Increments）について詳しく説明するね。これは確率過程、特にブラウン運動の重要な性質だよ。

    # 独立定常増分について

    ## 独立増分（Independent Increments）とは？

    確率過程 $\{X(t), t \geq 0\}$ が「独立増分」を持つとは、異なる時間区間での変化量が確率的に独立であることを意味するんだ。

    **数学的定義**:  
    任意の時点 $0 \leq t_1 < t_2 < \cdots < t_n$ に対して、確率変数の列
    $X(t_2) - X(t_1), X(t_3) - X(t_2), \ldots, X(t_n) - X(t_{n-1})$
    が互いに独立であるとき、その確率過程は独立増分を持つと言うよ。

    **直感的な意味**:  
    ある時間区間での過程の変化は、他の時間区間での変化と何の関係もない。過去の変化を知っていても、将来の変化を予測するのに役立たない。

    **例**:  
    サイコロを振り続けて、合計値を記録する確率過程を考えよう。新しいサイコロの目は前のサイコロの目と無関係だから、この過程は独立増分を持つよ。

    ## 定常増分（Stationary Increments）とは？

    確率過程 $\{X(t), t \geq 0\}$ が「定常増分」を持つとは、増分の確率分布が時間のシフトに対して不変であることを意味するんだ。

    **数学的定義**:  
    任意の時間 $s, t \geq 0$ と $h > 0$ に対して、
    $X(t+h) - X(s+h)$ と $X(t) - X(s)$
    は同じ確率分布に従う。

    **直感的な意味**:  
    増分の統計的性質（平均、分散、分布形など）は、いつ観測するかによらず同じ。朝の変化も夜の変化も同じような振る舞いをする。

    **例**:  
    ポアソン過程では、任意の長さ $t$ の時間区間で発生する事象の数は平均 $\lambda t$ のポアソン分布に従う。これは時間区間の長さだけに依存して、その区間の開始時点には依存しない。

    ## 独立定常増分を持つ確率過程の例

    1. **ブラウン運動**:  
       - **独立増分**: $B(t) - B(s)$ と $B(v) - B(u)$ は、時間区間 $[s,t]$ と $[u,v]$ が重ならなければ独立
       - **定常増分**: $B(t+h) - B(s+h)$ は $N(0, t-s)$ に従い、$h$ には依存しない

    2. **ポアソン過程**:  
       - **独立増分**: 異なる時間区間での事象発生数は独立
       - **定常増分**: 長さ $t$ の任意の時間区間での事象発生数は平均 $\lambda t$ のポアソン分布に従う

    3. **複合ポアソン過程**:  
       事象が発生するとランダムな大きさのジャンプが起こる過程。ジャンプのタイミングがポアソン過程に従い、ジャンプの大きさは独立同一分布に従う。

    4. **レヴィ過程**:  
       独立定常増分を持つ確率過程の一般的なクラス。連続部分（ブラウン運動的）とジャンプ部分を持つことができる。

    ## ブラウン運動における独立定常増分の証明

    ブラウン運動 $\{B(t), t \geq 0\}$ が独立定常増分を持つことを示してみよう。

    **独立増分の証明**:  
    ブラウン運動の定義から、任意の $0 \leq t_1 < t_2 < \cdots < t_n$ に対して、確率変数
    $B(t_1), B(t_2) - B(t_1), \ldots, B(t_n) - B(t_{n-1})$
    は互いに独立な正規分布に従う。

    具体的には、$B(t_i) - B(t_{i-1}) \sim N(0, t_i - t_{i-1})$ で互いに独立。

    **定常増分の証明**:  
    任意の $s < t$ と $h > 0$ に対して：

    $B(t+h) - B(s+h) \sim N(0, (t+h) - (s+h)) = N(0, t-s)$
    $B(t) - B(s) \sim N(0, t-s)$

    したがって、$B(t+h) - B(s+h)$ と $B(t) - B(s)$ は同じ確率分布に従う。

    ## 独立定常増分の重要性

    1. **計算の簡単化**:  
       これらの性質により、確率過程の多くの特性（期待値、分散、相関など）の計算が大幅に簡単になる。

    2. **マルコフ性との関係**:  
       独立増分を持つ過程は通常マルコフ過程（未来は現在だけで決まり、過去には依存しない）にもなる。

    3. **確率積分の構築**:  
       独立定常増分の性質は、伊藤積分などの確率積分を構築する際の基礎となる。

    4. **モデリングの柔軟性**:  
       現実の多くの現象（株価変動、粒子の運動など）は、この性質を近似的に持つためモデル化しやすい。

    ## 独立定常増分の制約

    独立定常増分を持つ過程には、いくつかの制約も存在するよ：

    1. **長期依存性の欠如**:  
       独立増分を持つ過程は、長期的な記憶効果（長期依存性）を表現できない。例えば、金融市場でよく観測される「ボラティリティクラスタリング」（変動が大きい期間が続く傾向）は表現できない。

    2. **変化率の一貫性**:  
       定常増分を持つ過程は、時間とともに変化率が変わるような現象を直接表現できない。

    3. **レヴィ=ヒンチンの定理による制約**:  
       独立定常増分を持つ確率過程の特性関数（確率分布のフーリエ変換）は特定の形に制限される。

    ## 独立増分と定常増分の拡張

    実世界の現象をより正確にモデル化するため、これらの概念は様々に拡張されているよ：

    1. **分数ブラウン運動**:  
       長期依存性を持つブラウン運動の拡張。独立増分は持たないが、自己相似性は保持する。

    2. **時間変化レヴィ過程**:  
       定常性を緩和し、時間とともに変化率が変わるよう拡張したモデル。

    3. **セミマルチンゲール**:  
       独立増分を持たないが、確率積分が定義できるより広いクラスの確率過程。

    ## まとめと応用例

    **まとめ**:  
    独立定常増分は確率過程の非常に強力な性質で、特にブラウン運動やポアソン過程といった基本的な確率過程の中心的な特徴となっている。これらの性質により、複雑な確率現象の分析が大幅に簡略化されるんだ。

    **応用例**:

    1. **金融工学**:  
       ブラック・ショールズモデルなど、多くの金融モデルは資産価格がこれらの性質を持つと仮定している。

    2. **待ち行列理論**:  
       顧客の到着過程がポアソン過程（独立定常増分を持つ）と仮定されることが多い。

    3. **通信工学**:  
       ノイズモデルとしてブラウン運動（ガウスノイズ）がよく使われる。

    4. **物理学**:  
       拡散現象やランダムウォークのモデル化に独立定常増分を持つ過程が活用される。

    この性質をよく理解すると、確率過程の理論と応用の両方で大きな助けになるよ！他に知りたいことがあれば、遠慮なく聞いてね。
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        この問題について丁寧に証明していきますね！

        # 問題の証明

        ## 問題の整理

        証明すべきことは：$W = (W_t)_{t \geq 0}$ を標準ブラウン運動としたとき、$B_t = \mu t + \sigma W_t$ もブラウン運動であることを示す必要があります。

        ここで、$\mu$ と $\sigma$ はパラメータで、$\sigma > 0$ と仮定できます。

        ## ブラウン運動の定義の確認

        まず、ブラウン運動の定義を確認しましょう。確率過程 $X = (X_t)_{t \geq 0}$ がブラウン運動であるためには、以下の条件を満たす必要があります：

        1. $X_0 = 0$ （初期値が0）
        2. 独立増分を持つ
        3. 定常増分を持つ
        4. 連続な標本経路を持つ

        標準ブラウン運動 $W$ は、さらに任意の $0 \leq s < t$ に対して増分 $W_t - W_s \sim N(0, t-s)$ が成り立ちます。

        ## $B_t = \mu t + \sigma W_t$ がブラウン運動であることの証明

        $B_t = \mu t + \sigma W_t$ をブラウン運動の定義に沿って検証していきます。

        ### 1. 初期値の確認

        $t = 0$ のとき：
        $B_0 = \mu \cdot 0 + \sigma W_0 = 0 + 0 = 0$

        ただし、$W_0 = 0$ （標準ブラウン運動の定義より）。

        したがって、$B_0 = 0$ が成り立ちます。

        ### 2. 独立増分の確認

        任意の時点 $0 \leq t_1 < t_2 < \cdots < t_n$ に対して、増分
        $B_{t_2} - B_{t_1}, B_{t_3} - B_{t_2}, \ldots, B_{t_n} - B_{t_{n-1}}$
        が互いに独立であることを示す必要があります。

        各増分を計算すると：
        $B_{t_i} - B_{t_{i-1}} = \mu t_i + \sigma W_{t_i} - (\mu t_{i-1} + \sigma W_{t_{i-1}})$
        $= \mu(t_i - t_{i-1}) + \sigma(W_{t_i} - W_{t_{i-1}})$

        標準ブラウン運動 $W$ の独立増分性により、$W_{t_i} - W_{t_{i-1}}$ は互いに独立です。
        そして $\mu(t_i - t_{i-1})$ は定数項なので、$B_{t_i} - B_{t_{i-1}}$ も互いに独立となります。

        したがって、$B_t$ は独立増分を持ちます。

        ### 3. 定常増分の確認

        任意の時間 $s, t \geq 0$ と $h > 0$ に対して、$B_{t+h} - B_{s+h}$ と $B_t - B_s$ が同じ確率分布に従うことを示します。

        まず、$B_t - B_s$ を計算します：
        $B_t - B_s = \mu t + \sigma W_t - (\mu s + \sigma W_s)$
        $= \mu(t-s) + \sigma(W_t - W_s)$

        $W$ の定常増分性により、$W_t - W_s \sim N(0, t-s)$ です。

        同様に、$B_{t+h} - B_{s+h}$ を計算します：
        $B_{t+h} - B_{s+h} = \mu(t+h) + \sigma W_{t+h} - (\mu(s+h) + \sigma W_{s+h})$
        $= \mu(t-s) + \sigma(W_{t+h} - W_{s+h})$

        $W$ の定常増分性により、$W_{t+h} - W_{s+h} \sim N(0, t-s)$ です。

        したがって、$B_t - B_s$ と $B_{t+h} - B_{s+h}$ は同じ確率分布に従います。
        つまり、$B_t$ は定常増分を持ちます。

        ### 4. 増分の確率分布

        任意の $0 \leq s < t$ に対して、増分 $B_t - B_s$ の確率分布を特定します：

        $B_t - B_s = \mu(t-s) + \sigma(W_t - W_s)$

        $W_t - W_s \sim N(0, t-s)$ なので、$\sigma(W_t - W_s) \sim N(0, \sigma^2(t-s))$ です。

        さらに、定数 $\mu(t-s)$ を加えると：
        $B_t - B_s \sim N(\mu(t-s), \sigma^2(t-s))$

        したがって、$B_t$ の増分は正規分布に従いますが、平均 $\mu(t-s)$ と分散 $\sigma^2(t-s)$ を持ちます。

        ### 5. 連続な標本経路

        $W_t$ が連続な標本経路を持ち、$\mu t$ も連続関数なので、それらの和 $B_t = \mu t + \sigma W_t$ も連続な標本経路を持ちます。

        ## 結論

        $B_t = \mu t + \sigma W_t$ はブラウン運動の定義を満たしていますが、標準ブラウン運動ではありません。具体的には：

        - 初期値：$B_0 = 0$
        - 独立増分：確認済み
        - 定常増分：確認済み
        - 連続な標本経路：確認済み
        - 増分の分布：$B_t - B_s \sim N(\mu(t-s), \sigma^2(t-s))$

        したがって、$B_t = \mu t + \sigma W_t$ はドリフト項 $\mu$ と拡散係数 $\sigma$ を持つブラウン運動です。これは「ドリフト付きブラウン運動」または「一般化されたブラウン運動」とも呼ばれます。

        $B_t$ は標準ブラウン運動とは異なり、増分が平均 $\mu(t-s)$ を持ちます。これは長期的には $\mu$ の方向に確定的なトレンドを持ちながら、$\sigma$ の大きさに応じたランダムな揺らぎを伴って動く確率過程となります。

        この結果は、金融モデルや物理現象のモデル化において非常に重要で、例えば株価のモデル化によく使われる幾何ブラウン運動の基礎となっています。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        もちろん、$B_t$ の分布を各部分でより詳しく導出していきますね。

        # $B_t$ の分布の詳細な導出

        ## $B_t - B_s$ の分布

        $B_t - B_s = \mu(t-s) + \sigma(W_t - W_s)$ という式があります。

        ここで分布を求めるには、各項の確率的性質を考慮します：

        1. $\mu(t-s)$ は定数項
        2. $W_t - W_s$ は標準ブラウン運動の増分で、$N(0, t-s)$ に従う

        確率変数 $X$ が $N(a, \sigma_X^2)$ に従い、定数 $c$ と $d$ があるとき、$Y = c + dX$ は $N(c + da, d^2\sigma_X^2)$ に従うという性質を使います。

        今回の場合：
        - $X = W_t - W_s \sim N(0, t-s)$
        - $c = \mu(t-s)$
        - $d = \sigma$

        したがって：
        $B_t - B_s = \mu(t-s) + \sigma(W_t - W_s) \sim N(\mu(t-s), \sigma^2(t-s))$

        つまり、$B_t - B_s$ は平均 $\mu(t-s)$、分散 $\sigma^2(t-s)$ の正規分布に従います。

        この分布の確率密度関数（PDF）は：

        $$f_{B_t-B_s}(x) = \frac{1}{\sqrt{2\pi\sigma^2(t-s)}} \exp\left(-\frac{(x-\mu(t-s))^2}{2\sigma^2(t-s)}\right)$$

        ## $B_{t+h} - B_{s+h}$ の分布

        同様に、$B_{t+h} - B_{s+h} = \mu(t-s) + \sigma(W_{t+h} - W_{s+h})$ について考えます。

        ここで、$W$ の定常増分性により $W_{t+h} - W_{s+h} \sim N(0, t-s)$ が成り立ちます。
        （これは、時間差 $t-s$ が同じなら、開始時点に関わらず増分の分布は同じであるという性質です）

        同じ計算手順で：
        - $X = W_{t+h} - W_{s+h} \sim N(0, t-s)$
        - $c = \mu(t-s)$
        - $d = \sigma$

        したがって：
        $B_{t+h} - B_{s+h} = \mu(t-s) + \sigma(W_{t+h} - W_{s+h}) \sim N(\mu(t-s), \sigma^2(t-s))$

        つまり、$B_{t+h} - B_{s+h}$ も平均 $\mu(t-s)$、分散 $\sigma^2(t-s)$ の正規分布に従います。

        この分布の確率密度関数（PDF）も同様に：

        $$f_{B_{t+h}-B_{s+h}}(x) = \frac{1}{\sqrt{2\pi\sigma^2(t-s)}} \exp\left(-\frac{(x-\mu(t-s))^2}{2\sigma^2(t-s)}\right)$$

        ## 確率分布の等価性の証明

        上記の計算から、$B_t - B_s$ と $B_{t+h} - B_{s+h}$ は同じ確率分布 $N(\mu(t-s), \sigma^2(t-s))$ に従うことが分かりました。

        この事実を形式的に示すには、両確率変数の特性関数（characteristic function）を比較する方法もあります：

        $B_t - B_s$ の特性関数は：
        $$\phi_{B_t-B_s}(u) = \exp\left(iu\mu(t-s) - \frac{1}{2}u^2\sigma^2(t-s)\right)$$

        $B_{t+h} - B_{s+h}$ の特性関数も：
        $$\phi_{B_{t+h}-B_{s+h}}(u) = \exp\left(iu\mu(t-s) - \frac{1}{2}u^2\sigma^2(t-s)\right)$$

        これら二つの特性関数が等しいことから、二つの確率変数は同一の確率分布に従うことが証明されます。

        ## ブラウン運動の定常増分性の確認

        以上の結果から、$B_t = \mu t + \sigma W_t$ は定常増分を持つことが確認できました。

        具体的には、任意の $s, t \geq 0$ と $h > 0$ に対して：
        - $B_t - B_s \sim N(\mu(t-s), \sigma^2(t-s))$
        - $B_{t+h} - B_{s+h} \sim N(\mu(t-s), \sigma^2(t-s))$

        これらは同一の確率分布であり、時間のシフト $h$ に依存していません。つまり、増分の分布は時間のシフトに対して不変（定常）であることが示されました。

        ## $B_t$ 自体の分布

        補足として、$B_t$ 自体の分布についても確認しておきましょう。

        $B_t = \mu t + \sigma W_t$ で、$W_t \sim N(0, t)$ なので：
        $B_t \sim N(\mu t, \sigma^2 t)$

        したがって、時刻 $t$ における $B_t$ は平均 $\mu t$、分散 $\sigma^2 t$ の正規分布に従います。この分布は $t$ に依存しているため、$B_t$ 自体は非定常過程ですが、その増分は定常という性質を持ちます。

        以上が、$B_t = \mu t + \sigma W_t$ の分布に関する詳細な導出です。確率過程 $B_t$ はドリフト項 $\mu$ を持つブラウン運動であり、標準ブラウン運動を平行移動・スケーリングした形になっています。
        """
    )
    return


app._unparsable_cell(
    r"""
    やあ！ポワソン過程について教えるよ。これは確率論の中でもとても重要な概念なんだ。できるだけわかりやすく説明していくね！

    # ポワソン過程とは？

    ポワソン過程は「ランダムに起こる事象」を数学的にモデル化したものだよ。例えば：
    - ウェブサイトへのアクセス
    - お店への客の来店
    - 交通事故の発生
    - 放射性物質の崩壊

    こういった「いつ起こるかわからないけど、ある平均的な頻度で起こる現象」を扱うときに使うんだ。

    ## ポワソン過程の特徴

    ポワソン過程には3つの重要な特徴があるよ：

    1. **独立増分性**：異なる時間区間で起こる事象の数は互いに独立
    2. **定常増分性**：事象の発生確率は時間の経過に関係なく一定
    3. **希薄性**：極めて短い時間では、最大でも1つの事象しか起こらない

    ## 数学的な定義

    時間 $t$ における事象の発生回数を $N(t)$ と表すとき、ポワソン過程は以下の性質を持つよ：

    1. $N(0) = 0$（最初は何も起こっていない）
    2. 任意の $s < t$ に対して、$N(t) - N(s)$ は平均 $\lambda(t-s)$ のポワソン分布に従う
    3. 互いに重ならない時間区間での増分 $N(t_2) - N(t_1)$, $N(t_4) - N(t_3)$, ... は互いに独立

    ここで $\lambda$ は**強度**または**レート**と呼ばれるパラメータで、単位時間あたりの事象の平均発生回数を表すよ。

    ## ポワソン分布との関係

    時間区間 $[0,t]$ で発生する事象の数 $N(t)$ は、パラメータ $\lambda t$ のポワソン分布に従うんだ：

    $$P(N(t) = k) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}, \quad k = 0, 1, 2, \ldots$$

    つまり、時間 $t$ の間に $k$ 回の事象が発生する確率はこの式で計算できるよ！

    ## 指数分布との関係

    ポワソン過程ではもう一つ重要な性質があって、連続する事象間の時間間隔は**指数分布**に従うんだ。

    事象間の時間間隔を $X$ とすると：

    $$P(X > t) = e^{-\lambda t}$$

    また、確率密度関数は：

    $$f_X(t) = \lambda e^{-\lambda t}, \quad t \geq 0$$

    この平均は $1/\lambda$ になるよ。だから、事象の発生レートが大きいほど、次の事象までの待ち時間は短くなるんだ。

    ## 具体例で理解しよう

    例えば、あるカフェでは平均して1時間に10人の客が来るとしよう（$\lambda = 10$）。

    - 2時間の間に15人の客が来る確率は？
      $$P(N(2) = 15) = \frac{e^{-20} \cdot 20^{15}}{15!} \approx 0.0516$$

    - 次の客が5分以内に来る確率は？
      $$P(X \leq 5/60) = 1 - e^{-10 \cdot 5/60} \approx 0.5654$$

    ## ポワソン過程の応用

    ポワソン過程は様々な分野で活用されているよ：

    - **待ち行列理論**：銀行やコールセンターでの顧客到着モデル
    - **信頼性工学**：機器の故障発生モデル
    - **保険数理**：保険金請求の発生モデル
    - **神経科学**：ニューロンの発火パターンモデル
    - **通信工学**：ネットワークトラフィックモデル

    ## 非定常ポワソン過程

    実際の現象では、事象の発生率が時間によって変わることもあるよね。例えば、カフェへの来客数はランチタイムに多くなったりする。

    そういった場合は、レートパラメータ $\lambda$ を時間の関数 $\lambda(t)$ として扱う**非定常ポワソン過程**（非均質ポワソン過程）を使うよ。時間区間 $[s,t]$ での事象発生回数は平均 $\int_s^t \lambda(u) du$ のポワソン分布に従うんだ。

    ## ポワソン過程をシミュレーションしてみよう

    ポワソン過程をシミュレーションする一番簡単な方法は、指数分布に従う時間間隔を次々と生成していくことだよ：

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def simulate_poisson_process(rate, T):
        \"\"\"
        rate: 強度パラメータλ
        T: シミュレーション時間
        \"\"\"
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
    print(f\"発生時刻: {[round(t, 2) for t in events]}\")
    print(f\"発生回数: {len(events)}\")

    # プロット
    plt.figure(figsize=(10, 2))
    plt.scatter(events, [1]*len(events), marker='|', s=100)
    plt.xlabel('時間')
    plt.yticks([])
    plt.title('ポワソン過程のシミュレーション (λ=5)')
    plt.show()
    ```

    ## まとめ

    ポワソン過程は、ランダムな事象の発生を扱う最も基本的で重要な確率過程の一つだよ。その特徴は：

    - 事象の発生は互いに独立
    - 発生レート $\lambda$ が一定（定常過程の場合）
    - 時間区間 $[0,t]$ での事象数はポワソン分布に従う
    - 事象間の時間間隔は指数分布に従う

    実世界の多くの現象がポワソン過程でモデル化できるから、統計学や機械学習、オペレーションズリサーチなど様々な分野で応用されているんだ。

    何か特定の部分についてもっと知りたいことがあれば、気軽に聞いてね！
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(np, plt):


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
    return events, simulate_poisson_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        そうそう、鋭い質問だね！ポワソン過程はガンマ分布とも深い関係があるんだ。説明していくね！

        # ポワソン過程とガンマ分布の関係

        ポワソン過程において、事象間の時間間隔は指数分布に従うということは既に説明したよね。でも「最初から数えて$n$番目の事象が発生するまでの時間」を考えるとどうなるかな？これがガンマ分布と関係してくるんだ！

        ## ガンマ分布とは

        まず簡単にガンマ分布について説明すると、パラメータ $\alpha$ (形状パラメータ) と $\beta$ (尺度パラメータ) を持つ連続確率分布で、確率密度関数は：

        $$f(x; \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}, \quad x > 0$$

        ここで $\Gamma(\alpha)$ はガンマ関数で、$\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1}e^{-t}dt$ だよ。

        ## ポワソン過程における$n$番目の事象発生時刻

        さて、ポワソン過程で強度 $\lambda$ のとき、「$n$番目の事象が発生するまでの時間」を $S_n$ とすると、$S_n$ はパラメータ $\alpha = n$, $\beta = \lambda$ のガンマ分布に従うんだ！

        つまり：

        $$f_{S_n}(t) = \frac{\lambda^n}{\Gamma(n)}t^{n-1}e^{-\lambda t}, \quad t > 0$$

        これは直感的に理解できるよ：
        - $n=1$ のとき、これは単に指数分布になる（最初の事象までの待ち時間）
        - $n>1$ のとき、$S_n$ は $n$ 個の独立な指数分布に従う確率変数（各事象間の時間間隔）の和になる

        ## なぜガンマ分布になるのか？

        指数分布に従う独立な確率変数 $X_1, X_2, \ldots, X_n$ の和 $S_n = X_1 + X_2 + \ldots + X_n$ の分布を考えてみよう。

        各 $X_i$ は平均 $1/\lambda$ の指数分布に従うから、モーメント母関数は：

        $$M_{X_i}(t) = \frac{\lambda}{\lambda - t}, \quad t < \lambda$$

        独立な確率変数の和のモーメント母関数は個々のモーメント母関数の積になるから：

        $$M_{S_n}(t) = \prod_{i=1}^{n} M_{X_i}(t) = \left(\frac{\lambda}{\lambda - t}\right)^n, \quad t < \lambda$$

        これはパラメータ $\alpha = n$, $\beta = \lambda$ のガンマ分布のモーメント母関数と同じなんだ！

        ## エルラン分布との関係

        特に、$\alpha$ が正の整数の場合のガンマ分布は**エルラン分布**と呼ばれるよ。だから、ポワソン過程の「$n$番目の事象発生時刻」はエルラン分布に従うとも言えるんだ。

        ## 具体例で理解しよう

        例えば、平均して1時間に5件の注文が入るオンラインショップを考えよう（$\lambda = 5$）。

        - 最初の注文が来るまでの時間は指数分布（$\lambda = 5$）に従う
        - 3件目の注文が来るまでの時間はガンマ分布（$\alpha = 3$, $\beta = 5$）に従う

        具体的な計算をしてみると：
        - 最初の注文が30分以内に来る確率：
          $$P(X_1 \leq 0.5) = 1 - e^{-5 \cdot 0.5} \approx 0.9179$$

        - 3件目の注文が1時間以内に来る確率：
          $$P(S_3 \leq 1) = \int_0^1 \frac{5^3}{\Gamma(3)}t^{2}e^{-5t}dt \approx 0.7619$$

        ## シミュレーションで確認

        ポワソン過程をシミュレーションして、$n$番目の事象発生時刻がガンマ分布に従うことを確認してみよう：

        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        # パラメータ
        lam = 5  # 強度パラメータ
        n = 3    # n番目の事象
        num_simulations = 10000  # シミュレーション回数

        # n番目の事象発生時刻をシミュレーション
        Sn_samples = []
        for _ in range(num_simulations):
            # n個の指数分布からのサンプルを生成して和をとる
            exponential_samples = np.random.exponential(scale=1/lam, size=n)
            Sn = np.sum(exponential_samples)
            Sn_samples.append(Sn)

        # 理論的なガンマ分布
        x = np.linspace(0, 2, 1000)
        gamma_pdf = stats.gamma.pdf(x, a=n, scale=1/lam)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.hist(Sn_samples, bins=50, density=True, alpha=0.7, label='シミュレーション結果')
        plt.plot(x, gamma_pdf, 'r-', linewidth=2, label=f'ガンマ分布(α={n}, β={lam})')
        plt.xlabel('時間')
        plt.ylabel('確率密度')
        plt.title(f'ポワソン過程における{n}番目の事象発生時刻の分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        ```

        ## まとめ：ポワソン過程、指数分布、ガンマ分布の関係

        ポワソン過程では：
        1. 時間間隔 $[0,t]$ での事象発生回数 $N(t)$ は平均 $\lambda t$ のポワソン分布に従う
        2. 連続する事象間の時間間隔は平均 $1/\lambda$ の指数分布に従う
        3. $n$番目の事象発生時刻 $S_n$ はパラメータ $\alpha = n$, $\beta = \lambda$ のガンマ分布に従う

        このように、ポワソン過程、ポワソン分布、指数分布、ガンマ分布はすべて密接に関係していて、同じ現象の異なる側面を表現しているんだ！

        実は確率論や統計学ではこういった「異なる分布の間の関係性」が多くあって、それを理解すると様々な問題を異なる角度から解決できるようになるよ。

        他に質問があれば、気軽に聞いてね！
        """
    )
    return


@app.cell
def _(np, plt, stats):

    # パラメータ
    lam = 5  # 強度パラメータ
    n = 3    # n番目の事象
    num_simulations = 10000  # シミュレーション回数

    # n番目の事象発生時刻をシミュレーション
    Sn_samples = []
    for _ in range(num_simulations):
        # n個の指数分布からのサンプルを生成して和をとる
        exponential_samples = np.random.exponential(scale=1/lam, size=n)
        Sn = np.sum(exponential_samples)
        Sn_samples.append(Sn)

    # 理論的なガンマ分布
    x = np.linspace(0, 2, 1000)
    gamma_pdf = stats.gamma.pdf(x, a=n, scale=1/lam)

    # プロット
    plt.figure(figsize=(10, 6))
    plt.hist(Sn_samples, bins=50, density=True, alpha=0.7, label='シミュレーション結果')
    plt.plot(x, gamma_pdf, 'r-', linewidth=2, label=f'ガンマ分布(α={n}, β={lam})')
    plt.xlabel('時間')
    plt.ylabel('確率密度')
    plt.title(f'ポワソン過程における{n}番目の事象発生時刻の分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return (
        Sn,
        Sn_samples,
        exponential_samples,
        gamma_pdf,
        lam,
        n,
        num_simulations,
        x,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
