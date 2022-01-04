$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\L}{\mathcal{L}}
\newcommand{\M}{\mathcal{M}}
\renewcommand{\P}{\mathbb{P}}
\renewcommand{\S}{\mathcal{S}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Yes}{\text{Yes}}
\newcommand{\iid}{\overset{\text{iid}}{\sim}}
\newcommand{\Lap}{\operatorname{Lap}}
\newcommand{\OPT}{\operatorname{OPT}}
$$



# Differential Privacy

## Definitions

A **database** $x$ is a collection of records from a universe $\X$. It can be represented as a **histogram** $x \in \N^{|\X|}$, where $x_i$ is the number of elements of type $i \in \X$ in the database $x$. The total size of the database (i.e. number of records) is $\|x\|_1$. Distances between databases can be measured by $\|x-y\|_1$.

A **randomized algorithm** $\M$ with domain $A$ and (by assumption, discrete) range $B$ is associated with a mapping $M : A \to \Delta(B)$. Given input $a$, $\M$ outputs $b \in B$ with probability $(M(a))_b$.

A randomized algorithm $\M$ is said to be **$(\varepsilon, \delta)$-differentially private** if for all $\S \subseteq \operatorname{Range}(\M)$ and all $x, y$ such that $\|x-y\|_1 \leq 1$,
$$
\P(\M(x) \in \S) \leq \exp(\varepsilon)\P(\M(y) \in \S) + \delta
$$

For brevity, we say that an $(\varepsilon, 0)$-differentially private is just $\varepsilon$-differentially private. To prove $\varepsilon$-DP, one bounds the probability ratio $\frac{\P(\M(x) \in \S)}{\P(\M(y) \in \S)}$ by $\exp(\varepsilon)$.

The **privacy loss** incurred by observing $\xi$ is
$$
\L_{\M(x) \| \M(y)}^{\xi} = \log\left(\frac{\P(\M(x) = \xi)}{\P(\M(y) = \xi)}\right)
$$

The **$\ell_1$-sensitivity** of a function $f : \N^{|\X|} \to \R^k$  is
$$
\Delta f = \max_{x,y \in \N^{|\X|} :\, \|x-y\|_1=1} \|f(x) - f(y)\|_1
$$
Sensitivity captures the magnitude by which a single individual’s data can change the function in the worst case, and therefore, intuitively, the uncertainty in the response that we must introduce in order to hide the participation of a single individual.

## Basic results

Post-processing: If $\M : \N^{|\X|} \to R$ is $(\varepsilon, \delta)$-DP and $f$ is a deterministic or randomized mapping $R \to R'$, then $f \circ \M : \N^{|\X|} \to R'$ is $(\varepsilon, \delta)$-DP.

*Proof*. First consider a deterministic $f$. Let $x, y$ satisfy $\|x-y\|_1 \leq 1$ and fix an event $S \subseteq R'$. Let $T$ be the preimage of $S$ under $f$, i.e. $T = f^{-1}(S)= \{r \in R : f(r) \in S\}$, and observe
$$
\begin{align*}
\P(f(\M(x)) \in S) &= \P(\M(x) \in T) \\
&\leq \exp(\varepsilon) \P(\M(y) \in T) + \delta \\
&= \exp(\varepsilon) \P(f(\M(y)) \in S) + \delta \\
\end{align*}
$$
To extend to randomized $f$, consider that any randomized function can be written as a convex combination of deterministic functions, and a convex combination of DP mechanisms is DP.



## Basic mechanisms

### Randomized response

Used to ask people about a potentially embarrassing or incriminating property $P$. The answerer should do the following:

1. Flip a coin.
2. If tails, respond truthfully.
3. If heads, flip another coin, and respond Yes if heads, No if tails

Privacy is maintained by plausible deniability – a Yes could have come from two heads (1/4 chance), regardless of whether or not the invididual has property $P$.

At the same time, since we control the generation process, we can estimate summary statistics of the true response in aggregate. Observe that
$$
\E[\#\Yes] = \frac{3}{4} (\#P) + \frac{1}{4}(\#\lnot P)
$$
In other words, if $p$ is the true fraction of $P$, the expected fraction of Yes's is $\frac{3}{4}p + \frac{1}{4}(1-p) = \frac{p}{2} + \frac{1}{4}$. We can estimate $p$ by plugging in the observed fraction of Yes's and solving for $p$.

Why do we need randomness? Consider a non-trivial deterministic algorithm. An adversary can search, by perturbing a single row, for two input databases which differ by only a single row but yield different results when queried. This reveals information about the unknown row.

Randomized response is $(\log 3,0)$-DP.

### Laplace mechanism

The Laplace mechanism applies to query functions of the form $f : \N^{|\X|} \to \R^k$. The mechanism is defined as
$$
\M(x; f, \varepsilon) = f(x) + (Y_1, \dots, Y_k) \quad \text{where} \quad Y_i \iid \Lap\left(\frac{\Delta f}{\varepsilon}\right)
$$
The Laplace mechanism is $(\varepsilon,0)$-DP.

*Proof*. Let $p_x$ denote the density of $\M(x; f, \varepsilon)$ for any $x$. Then for any $x, y$ with $\|x-y\|_1 \leq 1$,
$$
\begin{align*}
\frac{p_x(z)}{p_y(z)} &= \prod_{i=1}^k \frac{\exp(-\frac{\varepsilon|f(x)_i-z_i|}{\Delta f})}{\exp(-\frac{\varepsilon|f(y)_i-z_i|}{\Delta f})} \\
&= \exp\left(\frac{\varepsilon}{\Delta f} \sum_{i=1}^k |f(y)_i-z_i| - |f(x)_i-z_i|\right) \\
&\le \exp\left(\frac{\varepsilon}{\Delta f} \sum_{i=1}^k |f(y)_i-f(x)_i|\right) \\
&\le \exp\left(\frac{\varepsilon \|f(x)-f(y)\|_1}{\Delta f}\right) \\
&\le \exp(\varepsilon)
\end{align*}
$$
We can also obtain a high-probability bound the error of the Laplace mechanism. This depends on the fact that a Laplace random variable $Y \sim \Lap(b)$ satisfies $\P(|Y| \ge tb) = \exp(-t)$ , i.e. $\P(|Y| \ge t) = \exp(-\frac{t}{b})$. Denoting $y = \M(x; f, \varepsilon)$, observe that
$$
\begin{align*}
\P(\|f(x) - y\|_\infty \ge t) &= \P(\max_{i \in [k]} |f(x)_i - y_i| \ge t) \\
&\le k \P(|Y_i| \ge t) \\
&= k \exp(-\frac{t}{\Delta f / \varepsilon}) \\
&= k \exp(-\frac{t\varepsilon}{\Delta f})
\end{align*}
$$
Setting this quantity equal to $\delta$ and solving for $t$ gives
$$
\begin{align*}
\delta &= k\exp(-\frac{t\varepsilon}{\Delta f}) \\
\frac{\delta}{k} &= \exp(-\frac{t\varepsilon}{\Delta f}) \\
\log(\frac{\delta}{k}) &= -\frac{t\varepsilon}{\Delta f} \\
t &= -\frac{\Delta f}{\varepsilon} \log(\frac{\delta}{k}) = \frac{\Delta f}{\varepsilon} \log(\frac{k}{\delta})
\end{align*}
$$

### Exponential mechanism

Consider a utility function $u : \N^{|\X|} \times R \to \R$ where $R$ is an arbitrary range of values. We extend sensitivity to this case by taking max over $R$:
$$
\Delta u = \max_{r \in R} \max_{\|x-y\|_1 \leq 1} |u(x,r) - u(y,r)|
$$
The **exponential mechanism** samples $r$ with probability proportional to $\exp(\frac{\varepsilon u(x,r)}{2 \Delta u})$.

The exponential mechanism is $(\varepsilon, 0)$-DP.

*Proof*. Left as exercise ;)

We can also obtain a high-probability bound on the suboptimality of the sampled result. Let $\OPT(x) = \max_r u(x,r)$ and $R_{\OPT}(x) = \{r \in R : u(x,r) =\OPT(x)\}$. For a given $c \ge 0$, the total unnormalized probability mass of elements $r \in R$ such that $u(x,r) \le c$ is at most $|R|\exp(\frac{\varepsilon c}{2\Delta u})$, while the total unnormalized mass of all elements (normalizing factor) is at least $|R_{\OPT}| \exp(\frac{\varepsilon \OPT(x)}{2\Delta u})$. Thus
$$
\P(u(x, \M(x; u, R)) \le c) \le \frac{|R|\exp(\frac{\varepsilon c}{2\Delta u})}{|R_{\OPT}|\exp(\frac{\varepsilon\OPT(x)}{2\Delta u})} = \frac{|R|}{|R_{\OPT}|} \exp(\frac{\varepsilon (c - \OPT(x))}{2 \Delta u})
$$
By setting the RHS = $e^{-t}$, we obtain
$$
\P\left(u(x, \M(x; u, R)) \le \OPT(x) - \frac{2\Delta u}{\varepsilon}\left(\log(\frac{|R|}{|R_{\OPT}|}) + t\right)\right) \le e^{-t}
$$

## Composition of privacy

If $\M_1$ is $\varepsilon_1$-DP and $\M_2$ is $\varepsilon_2$-DP, then their combination $\M_{1,2}(x) = (\M_1(x), \M_2(x))$ is $(\varepsilon_1 + \varepsilon_2)$-DP.

*Proof*. If $\|x-y\|_1 \le 1$, then
$$
\frac{\P(\M_{1,2}(x) = (r_1, r_2))}{\P(\M_{1,2}(y) = (r_1, r_2))} = \frac{\P(\M_1(x) = r_1)}{\P(\M_1(y) = r_1))} \frac{\P(\M_2(x) = r_2)}{\P(\M_2(y) = r_2)} \leq \exp(\varepsilon_!) \exp(\varepsilon_2) = \exp(\varepsilon_1 + \varepsilon_2)
$$


Applying the previous result repeatedly, we obtain that if $\M_i$ is $\varepsilon_i$-DP for $i \in [k]$, then the combination $\M_{[k]}(x) = (\M_1(x), \dots, \M_k(x))$ is $(\sum_{i=1}^k \varepsilon_i)$-DP. More generally, if $\M_i$ is $(\varepsilon_i, \delta_i)$-DP, then $\M_{[k]}$ is $(\sum_{i=1}^k \varepsilon_i, \sum_{i-1}^k \delta_i)$-DP.
