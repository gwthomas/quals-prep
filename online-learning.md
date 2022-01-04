$$
\newcommand{\T}{^{\top}}
\newcommand{\Unif}{\operatorname{Unif}}
$$



# Online Learning

Repeated game.

For $t = 1, \dots, T$:

* The player chooses $x_t$ from a convex loss set $\X$
* The adversary gives a loss $\ell_t = \ell_t(x_t)$
* Optional: $\ell_t$ is revealed to the agent 

## Regret setting

The *regret* of an algorithm $\A$ which chooses actions $x^\A_t$ according to the observed history
$$
R_T(\A) = \sup_{\ell_1, \dots, \ell_T} \left\{ \sum_{t=1}^T \ell_t(x^\A_t) - \min_x \sum_{t=1}^T \ell_t(x) \right\}
$$
We want regret that is sublinear in $T$.

## Online gradient descent

Online gradient just runs gradient descent in an online fashion:
$$
x_{t-1} = \Pi_{\X}(x_t - \eta_t \nabla \ell_t(x_t))
$$
Remarkably, despite the functions $\ell_t$ being potentially totally different, OGD with step sizes $\eta = \frac{D}{G\sqrt{t}}$ obtains sublinear regret:
$$
R_T \leq \frac{3}{2} GD\sqrt{T}
$$
where $G$ is an upper bound on the gradient norm and $D$ is an upper bound on the (Euclidean) diameter of $\X$.

Indeed, there is a lower bound of $\Omega(GD\sqrt{T})$ on the regret of any OCO algorithm, so OGD is optimal (up to constant factors)!

However, better results are possible if more is known about the losses. If the losses are $\alpha$-strongly convex, then OGD with step sizes $\eta = \frac{1}{\alpha T}$ achieves regret bounded like
$$
R_T \le \frac{G^2}{\alpha} (1 + \log T)
$$

### Corollary: stochastic optimization

The previous result for OGD yields a regret bound for stochastic optimization as well. Run OGD with stochastic gradients for $T$ steps and let $\bar{x} = \frac{1}{T} \sum_{t=1}^T x_t$. Then
$$
\E[\ell(\bar{x})] \le \min_{x \in \X} \ell(x) + \frac{3GD}{2\sqrt{T}}
$$
Proof:
$$
\begin{align*}
\E[f(\bar{x})] - f(x^*) &\le \E\left[\frac{1}{T} \sum_{t=1}^T f(x_t) \right] - f(x^*) \tag{convexity of $f$} \\
&\le \frac{1}{T} \E\left[\sum_t \nabla f(x_t)\T(x_t - x^*)\right] \\
&= \frac{1}{T} \E\left[\sum_t \hat\nabla_t \T(x_t - x^*)\right] \\
&= \frac{1}{T} \E\left[\sum_t f_t(x_t) - f_t(x^*)\right] \\
&= \frac{R_T^{\text{OGD}}}{T}
\end{align*}
$$

## Exponentiated gradient

Exponentiated gradient is an algorithm for solving problems on the simplex $\X = \Delta_n$ (e.g. expert problem). At each step $t$, we compute a new unnormalized vector $y_t$ according to
$$
y_{t+1}(i) = y_t(i) \exp(-\eta_t \nabla_t(i))
$$
and then project
$$
x_{t+1} = \frac{y_{t+1}}{\|y_{t+1\|_1}}
$$
Let $G_\infty \ge \|\nabla_t\|_\infty$ upper bound the gradient $\infty$-norm. Taking $\eta = \sqrt{\frac{\log n}{2TG_\infty^2}}$, exponentiated gradient has regret bounded by
$$
R_T \le G_{\infty} \sqrt{2T \log n}
$$

## Multi-armed bandits

The bandit setting is a specialization of OCO where we only observe $\ell_t(x_t)$, rather than all of $\ell_t$. 

### Non-stochastic setting

In the non-stochastic multi-armed bandit (MAB) problem, the "action" is a probability vector, and the loss is computed by averaging according to the provided weighting:
$$
\ell_t(x) = l_t \T x
$$
where $l_t$ is a vector of losses provided by the adversary.

### UCB

Upper confidence bound (UCB) algorithms are algorithms for stochastic MAB that follow the principle of "optimism in the face of uncertainty". UCB proceeds by generating a range of values that contains the true arms' means with high probability, then choosing the arm with the highest upper bound. For losses bounded in $[0,1]$, Hoeffding's inequality gives
$$
\P(\hat\mu_t(a) \ge \mu(a) + \varepsilon) \leq \exp(-2n_t(a)\varepsilon^2)
$$
where $n_t(a)$ is the number of times $a$ has been played at round $t$,  $\mu(a)$ is the mean reward for playing arm $a$ (which we assume is fixed across $t$ here), and $\hat\mu_t(a)$ is our empirical estimate of $\mu(a)$ at time $t$. Hence, for a given probability threshold $\delta = \exp(-2n_t(a)\varepsilon^2)$, the upper confidence bound is
$$
\hat\mu_a + \sqrt{\frac{\log\frac{1}{\delta}}{2n_t(a)}}
$$
UCB achieves logarithmic asymptotic regret bounded by $\log T \sum_{a | \Delta_a > 0} \Delta_a$.

### EXP3

EXP3 is similar to exponentiated gradient. At each step $t$,

* Sample $i_t \sim x_t$ and play $i_t$
* Update $y_{t+1}(i_t) = y_t(i_t) \exp(-\varepsilon x_t(i_t))$
* Normalize $x_{t+1} = \frac{y_{t+1}}{\|y_{t+1}\|_1}$

Taking $\varepsilon = \sqrt{\frac{\log n}{Tn}}$ guarantees regret bounded by $\O(\sqrt{T n\log n})$, which is uptimal up to logarithmic factors.

### Sphere sampling estimator

Consider a $\delta$-smoothed version of $f$:
$$
\hat{f}_\delta(x) = \E_{v \sim \Unif(B)} [f(x + \delta v)]
$$
where $B$ is the unit ball. Note that when $f$ is linear, $\hat{f}_\delta(x) = f(x)$.
$$
\E_{u \sim \Unif(B)}[f(x + \delta u)u] = \frac{\delta}{n} \nabla \hat{f}_\delta(x)
$$
where $S$ is the unit sphere. This suggests a simple estimator for $\nabla f$ where $f$ is linear: draw a random unit vector $u$ and take $\frac{n}{\delta} f(x + \delta u)u$ as the gradient estimate.

The FKM algorithm uses the spherical estimator:

* Sample $u_t$ from unit sphere, set $y_t = x_t + \delta u_t$
* Play $y_t$, observe loss $\ell_t(y_t)$. Let $g_t = \frac{n}{\delta} f_t(y_t)u_t$
* Update $x_{t+1} = x_t - \eta g_t$

For a certain choice of $\eta, \delta$, regret is bounded like $O(T^{3/4})$.
