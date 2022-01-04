$$
\newcommand{\indic}[1]{1_{#1}}
$$



# Causality

## Potential outcomes framework

Suppose we want to measure the causal effect of different treatments. Assume that the outcome for an individual $i$ will be $Y_i(t)$ at treatment condition $t$.  Commonly, we are interested in comparing the effect of a treatment vs. a control (no treatment), in which case we will identify the treatment with $t = 1$ and control with $t = 0$. Assume that for each individual we observe covariates $X_i$, treatment $T_i \in \{0,1\}$, and outcome $Y_i = Y_i(T_i)$. However, crucially, we cannot in general observe both $Y_i(0)$ and $Y_i(1)$ for any individual $i$.

## Average treatment effect

The **average treatment effect** (ATE) is the expected difference in outcome between the treatment and control, averaged over some population:
$$
ATE = \E[Y(1) - Y(0)] = \E[Y(1)] - \E[Y(0)]
$$
Computing ATE would be easy if you knew both $Y_i(1)$ and $Y_i(0)$ for every individual $i$. However, as mentioned previously, we do not expect to know both of these.

### Estimation

With a randomized experiment in which each individual is assigned a random treatment $T_i$ which is independent of $(Y_i(0), Y_i(1))$, one can obtain an unbiased estimate for the ATE. Write
$$
\begin{align*}
ATE &= \E[Y(1) - Y_(0) \given T = 1] \\
&= \E[Y(1) \given T = 1] - \E[Y_i(0) \given T = 1] \\
&= \E[Y(1) \given T = 1] - \E[Y_i(0) \given T = 0] \\
&= \E[Y \given T = 1] - \E[Y \given T = 0]
\end{align*}
$$
The quantities $\E[Y \given T =1]$ and $\E[Y \given T = 0]$ can be replaced by sample averages.

## Propensity score matching

In practice, randomized treatment is inappropriate for some settings due to ethical reasons. For example, we want to estimate the effect of smoking on people's health, but it is unethical to force people to smoke. Therefore we turn to observational studies.

The problem with observational studies is that the treatment may be highly correlated with an individual's attributes, so the outcome $Y_i$ may be more indicative of the individual's attributes $X_i$ than the actual efficacy of the treatment.

The **propensity score**, defined as
$$
s(x) = \P(T = 1 \given X = x)
$$
can be used to correct for this bias by reweighting *inversely* to the probability of treatment. Assuming **conditional exchangeability**, i.e.
$$
Y(0), Y(1) \perp T \given X
$$
it holds that
$$
\E[Y(t)] = \E\left[\frac{Y\indic{T = t}}{\P(T = t \given X)}\right]
$$
Note that this is a form of importance sampling. (More on IS below.) Derivation:
$$
\begin{align*}
\E\left[\frac{Y\indic{T=t}}{\P(T=t \given X)}\right] &= \E\left[ \E\left[\frac{Y(t)\indic{T=t}}{\P(T=t \given X)} \given X\right] \tag{LIE} \right] \\
&= \E\left[ \frac{\E[Y(t) \given X] \E[1_{T=t} \given X]}{\P(T=t \given X)} \right] \tag{conditional exchangeability} \\
&= \E\big[ \E[Y(t) \given X] \big] \tag{$\P(\cdot)$ and $\E[\indic{\cdot}]$ cancel} \\
&= \E[Y(t)] \tag{LIE}
\end{align*}
$$
Thus ATE can be computed as the difference between two terms:
$$
\begin{align*}
\E[Y(1)] &= \E\left[\frac{YT}{s(X)}\right] \\
\E[Y(0)] &= \E\left[\frac{Y(1-T)}{1-s(X)}\right]
\end{align*}
$$
and these can be replaced by sample averages. If $s(x)$ is unknown, as is typically the case in practice, is can be fit to the data $\{(X_i, T_i)\}_{i=1}^n$ using a logistic regression model.

## Off-policy batch RL



## Importance sampling

