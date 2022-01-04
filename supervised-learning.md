$$
\newcommand{\sign}{\operatorname{sign}}
$$



# Supervised Learning

## Logistic regression

## Support vector machines

## Boosting (AdaBoost)

A generic boosting algorithm proceeds as follows:

* For $t = 1, \dots, T$,
  * Construct a distribution $D_t$ over the indices
  * Run a weak learning algorithm on $D_t$, producing $h_t : \X \to \{-1, 1\}$
* Output $H$, a combination of $\{h_t\}$

AdaBoost is an instantiation of this generic algorithm with the following design choices:

* $D_t$ is given recursively by $D_{t+1}(i) \propto D_t(i) \exp(-y_i \alpha_th_t(x_i))$
  * That is, if $h_t(x_i) = y_i$, we downweight this example, and if $h_t(x_i) \ne y_i$, we upweight it.
* $H(x) = \sign(\sum_t \alpha_t h_t(x))$

Letting $\epsilon_t = \operatorname{err}_{D_t}(h_t) = \frac{1}{2} - \gamma_t$ (where $\gamma_t \ge \gamma$ by weak learning assumption), we have
$$
\operatorname{err}(H) \le \exp(-2\gamma^2T)
$$


## Decision trees

### Random forests

