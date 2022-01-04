$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\E}{\mathbb{E}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\given}{\,|\,}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\KL}[2]{\operatorname{KL}(#1 \,\|\, #2)}
\newcommand{\cN}{\mathcal{N}}
$$

# Probability and Statistics

## Distributions

### Gaussian

The single-variable Gaussian distribution has the density
$$
p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

### Laplace

The density of a Laplace random variable with scale parameter $b$ is
$$
p(x; b) = \frac{1}{2b} \exp(-\frac{|x|}{b})
$$


### Poisson

The Poisson is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area or volume.

The probability mass function is
$$
p(x; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
If $X \sim \text{Poisson}(\lambda)$, then $\lambda = \E[X] = \Var(X)$.

### Gamma

The density for a Gamma distribution with shape parameter $\alpha$ and rate parameter $\beta$ is
$$
p(x; \alpha, \beta) = \frac{x^{\alpha-1} e^{-\beta x} \beta^\alpha}{\Gamma(\alpha)}
$$
If $X_1, \dots, X_n$ are iid exponential r.v. with rate parameter $\lambda$, then $\sum_i X_i \sim \operatorname{Gamma}(n, \lambda)$ and $\bar{X} \sim \operatorname{Gamma}(n, n\lambda)$.

### Dirichlet

The Dirichlet distribution is a multivariate generalization of the beta distribution. Parameterized by shape parameters $\alpha_1, \dots, \alpha_K$ has density
$$
p(x_1, \dots, x_K; \alpha_1, \dots, \alpha_K) = \frac{1}{B(\alpha_1, \dots, \alpha_K)} \prod_{k=1}^K x^{\alpha_k - 1}
$$
where $\{x_k\}$ are restricted to the simplex. This form makes it a useful prior in Bayesian statistics, as it gives a distribution over (discrete) distributions.

### Multinomial

The multinomial distribution is a multivariate generalization of the binomial distribution. It models the number of various outcomes given $n$ (independent) trials where each trial picks outcome $k$ with probability $p_k$. The pmf is
$$
p(x_1, \dots, x_K; p_1, \dots, p_K) = \frac{n!}{\prod_{k} x_k!} \prod_{k} p_k^{x_k}
$$
where $n = \sum_k x_k$.

Note

* $\E[X_i] = np_k$
* $\Var(X_i) = np_i(1-p_i)$
* $\Cov(X_i, X_j) = -np_ip_j$
* If $K = 2$, we have binomial distribution

## Exponential family

Let $\X$ be a (discrete) set and $\phi : \X \to \R^d$ a feature function. The exponential family parameterized by $\theta$ has pmf
$$
p_\theta(x) = \exp(\theta \cdot \phi(x) - A(\theta))
$$
where $A(\theta)$ is the log-partition function
$$
A(\theta) = \log \sum_x \exp(\theta \cdot \phi(x))
$$
Properties:

* $\nabla A(\theta) = \E_{x \sim p_\theta}[\phi(x)]$
* $\nabla^2 A(\theta) = \Cov_{x \sim p_\theta}(\phi(x))$
  * Since covariance matrices are always PSD, it follows that $A$ is a convex function of $\theta$.

The family is said to be **minimal** if $\nabla^2 A$ is strictly positive definite everywhere. For a minimal exponential family, the mapping $\mu = \nabla A(\theta)$ can be inverted to obtain $\theta = (\nabla A)^{-1}(\mu)$. In this context, $\theta$ are the **canonical parameters** and $\mu$ are the **mean parameters**.

When performing maximum likelihood, the critical points must satisfy
$$
0 = \hat\mu - \nabla A(\hat\theta)
$$
so have sufficient statistics $\hat\mu = \frac{1}{n} \phi(x_i)$, and $\hat\theta = (\nabla A)^{-1} \hat\mu$.

## Sufficient statistics

A statistic $T(X)$ is **sufficient** for parameter $\theta$ if $p(X \given T) = p(X \given T, \theta)$, or equivalently if $I(\theta, T(X)) = I(\theta, X)$.

## Maximum likelihood

## Asymptotic normality of M estimators

Suppose the estimator $\hat\theta$ is consistent (converges to $\theta^*$ w.p. 1 as $n \to \infty$) and regularity conditions such as $\nabla^2 L(\theta^*)$ being full rank. Then
$$
\sqrt{n}(\hat\theta - \theta^*) \overset{d}{\to} \cN(0, (\nabla^2 L(\theta^*))^{-1} \Cov(\nabla \ell(z, \theta ^*)) (\nabla^2 L(\theta^*))^{-1}
$$
Proof sketch: We know $\nabla \hat{L}(\hat\theta) = 0$. Taylor expand $\nabla\hat{L}(\theta)$ around $\theta^*$:
$$
0 = \nabla \hat{L}(\hat{\theta}) = \nabla \hat{L}(\theta^*) + \nabla^2 \hat{L}(\theta^*)(\hat\theta - \theta^*) + O(\|\hat\theta - \theta^*\|^2)
$$
Rearranging and multiplying by $\sqrt{n}$,
$$
\sqrt{n}(\hat\theta - \theta^*) = -(\nabla^2 \hat{L}(\theta^*)) \, \sqrt{n} \, \nabla \hat{L}(\theta^*) + h.o.t.
$$
Now apply CLT where $X_i = \nabla \ell(z; \theta^*)$ and $\bar{X} = \nabla \hat{L}(\theta^*)$, and we get that
$$
\sqrt{n} (\nabla \hat{L}(\theta^*) - \underbrace{\nabla L(\theta^*)}_0) \overset{d}{\to} \cN(0, \Cov(\nabla \ell(z, \theta^*)))
$$
We also have $\nabla^2 \hat{L}(\theta^*)$ converging to $\nabla^2 L(\theta^*)$ by LLN. Thus, by Slutsky's theorem,
$$
\sqrt{n} (\hat\theta - \theta^*) \overset{d}{\to} (\nabla^2 L(\theta^*))^{-1} \cN(0, \Cov(\nabla\ell(z, \theta^*)))
$$
from which the statement above follows.

In the well-specified case (i.e. the true distribution $p^*$ lies within the family $\{p_\theta : \theta \in \Theta\}$), $\Cov(\nabla\ell(z, \theta^*)) = \nabla^2 L(\theta^*)$, so the expression for asymptotic covariance simplifies to $(\nabla^2 L(\theta^*))^{-1}$.

## Fisher information

The Fisher information matrix (FIM) is
$$
I(\theta) = \E_{x \sim p_\theta}\left[ \nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^\top \right]
$$
or, under certain regularity conditions(?),
$$
I(\theta) = - \E_{x \sim p_\theta}\left[ \nabla^2_\theta \log p_\theta(x) \right]
$$
The FIM is the Hessian of the KL divergence.

## Information theory

### Entropy, conditional entropy

Entropy of a distribution with pmf $p$ is
$$
H(p) = - \sum_x p(x) \log p(x)
$$


### KL divergence

KL divergence between $p$ and $q$ is
$$
\KL{p}{q} = \sum_x p(x) \log\frac{p(x)}{q(x)}
$$
KL divergence is always $\ge 0$, with equality iff $p = q$.

### Mutual information

Mutual information between $X$ and $Y$ is the KL divergence between their joint distribution $p(x,y)$ and the product of the conditional distributions $p(x)$, $p(y)$:
$$
I(X, Y) = \KL{p_{X,Y}}{p_X \times p_Y} = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$
As a consequence of properties of KL divergence, $I(X,Y) \ge 0$, with equality iff $X$ and $Y$ are independent.

### Relationships

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Figchannel2017ab.svg/2880px-Figchannel2017ab.svg.png)

## Bias-variance tradeoff (for squared loss)

Under squared loss, the population risk decomposes as
$$
\begin{align*}
L(h) &= \E_{(x,y) \sim P}[(h(x) - y)^2] \\
&= \E_x \E_{y|x} [h(x)^2 - 2yh(x) + y^2] \\
&= \E_x\big[h(x)^2 - 2h(x)\E_{y|x}[y] + \E_{y|x}[y^2]\big] \\
&= \E_x\big[h(x)^2 - 2h(x)\E_{y|x}[y] + \E_{y|x}[y]^2 + \E_{y|x}[y^2] - \E_{y|x}[y]^2\big] \\
&= \E_x\big[\underbrace{(h(x) - \E_{y|x}[y])^2}_{\text{bias}^2} + \underbrace{\Var(y \given x)}_{\text{variance}}\big]
\end{align*}
$$
Since the variance does not depend on $h$, it is clearly best to pick $h$ that approximates $h(x) \approx \E_{y|x}[y]$.

## Jensen's inequality

For a convex function $f$, the secant line lies above the graph of $f$:
$$
f(tx_1 + (1-t)x_2) \le t f(x_1) + (1-t)f(x_2)
$$
Thus
$$
f(\E[X]) \le \E[f(X)]
$$

## Hypothesis testing

### $p$-value, power

The $p$-value is the probability that the probability of observing a result "at least as extreme" as what has been observed, assuming the null hypothesis is true. Exactly what is meant by "at least as extreme" depends on the type of test:

* One-sided test: $\P(T \ge t \given H_0)$ or $\P(T \le t \given H_0)$
* Two-sided test (symmetric about 0): $\P(|T| \ge |t| \given H_0)$

There are different kinds of outcomes in a test:

![Screen Shot 2021-12-15 at 11.48.38 PM](/Users/garrett/Desktop/Screen Shot 2021-12-15 at 11.48.38 PM.png)

The **power** of a hypothesis test against an alternative hypothesis $H_1$ is $\P(\text{reject $H_0$} \given \text{$H_1$ is true})$.

### $t$-test

A $t$-test is any hypothesis test in which the test statistic follows a Student's $t$ distribution under the null hypothesis.

A one-sample $t$-test for testing whether or not population mean equals $\mu$: construct statistic
$$
t = \frac{\bar{x} - \mu}{s/\sqrt{n}}
$$
where $\bar{x}$ is the sample mean, $s$ is the sampled standard deviation, and $n$ is the sample size. This statistic follows a $t$ distribution with $n-1$ degrees of freedom. This can also be extended to paired two-sample tests, by performing a one-sample test on the differences of pairs.

For a two-sample test with equal samples sizes and variances, use
$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2 + s_2^2}{n}}}
$$

### Bootstrap

To test whether a parameter equals a certain value, construct a bootstrap confidence interval and take quantiles.

### Multiple hypothesis testing

If performing $m$ simultaneous tests at level $\alpha$, the number of errors increases with $m$. The *Bonferroni correction* (based on a union bound) suggests testing each hypothesis at level $\alpha/m$. Then the overall level of errors is still at level $\alpha$.
