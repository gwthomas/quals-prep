$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\Z}{\mathcal{Z}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\H}{\mathcal{H}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\iid}{\overset{\text{iid}}{\sim}}
\newcommand{\A}{\mathcal{A}}
\renewcommand{\O}{\mathcal{O}}
\newcommand{\poly}{\operatorname{poly}}
\newcommand{\VC}{\operatorname{VC}}
$$



# Learning Theory

## PAC learning framework

An algorithm $\A$ is **probably approximately correct** (PAC) for a learning problem if for any $\delta, \epsilon > 0$, given $p = \poly(1/\delta, 1/\epsilon)$ samples from a distribution $P$, with probability at least $1 - \delta$, $\A$ outputs a hypothesis with at most $\epsilon$ error on $P$.

## Concentration inequalities

### Hoeffding’s inequality

Let $X_i \in [a_i, b_i]$, then
$$
\P(\bar{X} \ge \E[\bar{X}] + t) \le \exp(-\frac{2n^2t^2}{\sum_i (b_i - a_i)^2})
$$

## Generalization bound for finite hypothesis class

Let $\ell(h, z) \in [0,1]$ be the loss of a hypothesis $h \in \H$ on example $z \in \mathcal{Z}$.  We assume $n$ samples $z_i \iid P$, with average empirical loss $\hat{L}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h, z_i)$ and true loss $L(h) = \E_{z \sim P}[\ell(h, z)]$. The ERM solution is $\hat{h} = \arg\min \hat{L}(h)$.

We want to show that w.h.p. the **excess risk** $L(\hat{h}) - L(h^*)$ of the ERM solution $\hat{h}$ is low:
$$
\P(L(\hat{h}) - L(h^*) \ge \epsilon) \le \delta
$$
We use the following decomposition and bound:
$$
L(\hat{h}) - L(h^*) = L(\hat{h}) - \hat{L}(\hat{h}) + \underbrace{\hat{L}(\hat{h}) - \hat{L}(h^*)}_{\le 0} + \hat{L}(h^*) - L(h^*) \leq 2 \sup_{h \in \H} |\hat{L}(h) - L(h)|
$$
It follows that
$$
\P(L(\hat{h}) - L(h^*) \ge \epsilon)) \le \P\left(\sup_{h \in \H} |\hat{L}(h) - L(h)| \ge \frac{\epsilon}{2}\right)
$$
For a finite hypothesis class $\H = \{h_1, \dots, h_n\}$, the uniform convergence can be obtained by a simple union bound:
$$
\P\left(\sup_{h \in \H} |\hat{L}(h) - L(h)| \ge \frac{\epsilon}{2}\right) = \P\left(\bigcup_{h \in \H} \left\{|\hat{L}(h) - L(h)| \ge \frac{\epsilon}{2}\right\}\right)  \leq \sum_{h \in \H} \P\left(|\hat{L}(h) - L(h)| \ge \frac{\epsilon}{2}\right)
$$
If we can show that for all $h \in \H$, $\P(|\hat{L} - L(h)| \ge \epsilon/2) \le \delta/|\H|$, then we have the desired bound.

Two-sided Hoeffding tells us that for all $h$,
$$
\P(|\hat{L}(h) - L(h)| \ge t) \leq 2\exp(-2nt^2)
$$
Thus, putting it all together,
$$
\begin{align*}
\P(L(\hat{h}) - L(h^*) \ge \epsilon) &\leq \P\left(\sup_{h \in \H} |\hat{L}(h) - L(h)| \ge \frac{\epsilon}{2}\right) \\
&\leq |\H| \cdot 2 \exp\left(-2n\left(\frac{\epsilon}{2}\right)^2\right) \\
&= 2|\H| \exp(-\frac{n\epsilon^2}{2})
\end{align*}
$$
Setting this upper bound to $\delta$ and solving, we obtain
$$
\begin{align*}
2|\H| \exp(-\frac{n\epsilon^2}{2}) &= \delta \\
-\frac{n\epsilon^2}{2} &= \log\frac{\delta}{2|\H|} \\
\epsilon &= \sqrt{\frac{2}{n} \log \frac{2|\H|}{\delta}} = \O\left(\sqrt{\frac{\log|\H|}{n}}\right)
\end{align*}
$$


## Measures of complexity

### VC dimension

Let $\F$ be a set of functions mapping $\Z$ to a finite set. Then the **shattering coefficient** of $\F$ with $n$ points is
$$
s(\F, n) = \max_{z_1, \dots, z_n \in \Z} |\{f(z_1), \dots, f(z_n) : f \in \F\}|
$$
If $\F$ is a collection of boolean functions and $s(\F, n) = 2^n$, we say that $\F$ *shatters* $n$ points.

The **VC dimension** of a hypothesis class $\H$ is the maximum number of points that can be shattered by $\H$.
$$
\VC(\H) = \sup \{s(\H, n) = 2^n\}
$$

#### Examples

* Intervals: $\H = \{z \mapsto 1_{z \in [a,b]} : a, b \in \R\}$ have $\VC(\H) = 2$ because if there are three points on the line with labels e.g. $(1, 0, 1)$, there is no way to isolate the middle point
* Similarly, if $\H$ is the class of rectangle indicators, $\VC(\H) = 4$
* Let $\F$ be a function class with a finite basis. Let $\H = \{1_{f(\cdot) \ge 0} : f \in \F\}$. Then $\VC(\H) \le \dim(\F)$.
  * As a corollary, half-spaces in $d$ dimensions have VC dimension $d$.

### Rademacher complexity

Let $\F$ be a class of functions $\Z \to \R$. The **Rademacher complexity** of $\F$ on $n$ points is
$$
R_n(\F) = \E_{\sigma, z}\left[ \sup_{f \in \F} \frac{1}{n} \sum_{i=1}^n \sigma_i f(z_i) \right]
$$
where $\sigma_i \iid \operatorname{Unif}\{-1,1\}$ are Rademacher random variables and $z_i \iid P$. Similarly, the **empirical Rademacher complexity** is the same quantity without expectation over the $\{z_i\}$, which are fixed:
$$
\hat{R}_n(\F) = \E_\sigma\left[ \sup_{f \in \F} \frac{1}{n} \sum_{i=1}^n \sigma_i f(z_i) \right]
$$
Intuitively, if $\F$ is a very rich function class (i.e. high complexity), then for any assignment of the $\{\sigma_i, z_i\}$ we can find an $f \in \F$ such that $\{f(z_i)\}$ is highly correlated with $\sigma_i$.

#### Basic properties

* $R_n(\{f\}) = 0$
* If $\F_1 \subseteq \F_2$, then $R_n(\F_1) \le R_n(\F_2)$
* $R_n(\F_1 + \F_2) = R_n(\F_1) + R_n(F_2)$
* $R_n(c\F) = |c|R_n(\F)$
* Talagrand's inequality: $R_n(\phi \circ \F) \le L_\phi R_n(\F)$ where $L_\phi$ is the Lipschitz constant of $L$

#### Examples

* Linear functions with weights bounded in $L_2$ ball:
  * Let $\F = \{z \mapsto w \cdot z : \|w\|_2 \le B_2\}$
  * Assume $\E\|z\|_2^2 \le C_2^2$
  * Then $R_n(\F) \le \frac{B_2C_2}{\sqrt{n}}$
* Linear functions with weights bounded in $L_1$ ball:
  * Let $\F = \{z \mapsto w \cdot z : \|w\|_1 \le B_1\}$
  * Assume $\|z\|_\infty \le C_\infty$ w.p. 1
  * Then $R_n(\F) \le \frac{B_1C_\infty \sqrt{2\log(2d)}}{\sqrt{n}}$

#### Generalization bound based on Rademacher complexity

Let $\F = \{z \mapsto \ell(h, z): h \in \H\}$ be the loss class. Then w.p. $\ge 1-\delta$,
$$
L(\hat{h}) - L(h^*) \leq 4 R_n(\F) + \sqrt{\frac{2\log\frac{2}{\delta}}{n}}
$$
**Massart's finite lemma** states that, if $\F$ is finite and $M^2$ is a bound on the second moment, i.e. $\sup_{f \in \F} \frac{1}{n} \sum_{i=1}^n f(z_i)^2 \le M^2$, then the empirical Rademacher complexity is bounded as
$$
\hat{R}_n(\F) \le \sqrt{\frac{2M^2\log|\F|}{n}}
$$

### Metric entropy

