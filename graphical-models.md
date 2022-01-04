$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\iid}{\overset{\text{iid}}{\sim}}
\newcommand{\given}{\,|\,}
\newcommand{\d}{\,\mathrm{d}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\D}{\mathcal{D}}
\newcommand{\cC}{\mathcal{C}}
\newcommand{\pye}{\tilde{p}}
\newcommand{\KL}[2]{\mathrm{KL}(#1 \,\|\, #2)}
\newcommand{\xye}{\tilde{x}}
$$



# Graphical Models

## Bayesian networks

A **Bayesian network** is a probabilistic graphical model where the associated graph $G = (V,E)$ is directed and acyclic (that is, a DAG). Each node $v \in V$ has an associated random variable $x_v$. Recall that DAGs can always be linearized (i.e. ordered), so we identify $V$ with $\{1, \dots, d\}$, where $d = |V|$.

Similarly, every joint distribution $p(x_1, \dots, x_d)$ can be factorized as
$$
p(x_1, \dots, x_d) = \prod_{v=1}^d p(x_v \given x_1,\dots,x_{v-1})
$$
In a Bayesian network, the distributions are typically dependent on only a subset of their ancestors:
$$
p(x_v \given x_1, \dots, x_{v-1}) = p(x_v \given A_v)
$$
where $A_v \subseteq \{x_1, \dots, x_{v-1}\}$ are the *ancestors* of vertex $v$ in the DAG.

### Dependency structures

Consider these simple DAG structures:

![Bayesian networks over three variables, encoding different types of dependencies: cascade (a,b), common parent (c), and v-structure (d).](https://ermongroup.github.io/cs228-notes/assets/img/3node-bayesnets.png)

* Common parent, $A \leftarrow B \rightarrow C$:
  * $A \perp C \given B$, because $B$ contains all the information that links $A$ and $C$.
  * $A \not\perp C$ if $B$ is not observed, because $B$ contains information that partially determines $A$ and $C$.
*  Cascade, $A \rightarrow B \rightarrow C$:
  * $A \perp C \given B$, because $B$ contains all the information from $A$, as far as $C$ is concerned.
  * $A \not\perp C$ if $B$ is not observed, because $A$ contains information that partially determines $B$, which partially determines $C$.
* $V$-structure, i.e. **explaining away**, $A \rightarrow C \leftarrow B$:
  * $A \not\perp B \given C$, because $C$ can be "explained by" either $A$ or $B$, so $p(a \given b) \neq p(a \given b, c)$ in general. For example, suppose $A$ is "it rained", $B$ is "sprinkler turned on", and $C$ is "grass is wet". If we know that the grass is wet and the sprinkler did not turn on, it must have rained, i.e. $p(a = 1 \given b = 0, c = 1) = 1$. However, $p(a = 1 \given b = 0) \ne 1$ in a reasonable model; clearly, we do not think it rains constantly when the sprinkler is off.
  * $A \perp B$ if $C$ is unobserved

### $d$-separation

Let $Q$, $W$, $O$ be three sets of nodes in a Bayesian network. We say that $Q$ and $W$ are **$d$-connected** given $O$ if $Q$ and $W$ are connected by an **active path**, which is an undirected path such that for every consecutive triplet $X$, $Y$, $Z$, one of the following holds:

* $X \leftarrow Y \leftarrow Z$ and $Y \not\in O$
* $X \rightarrow Y \rightarrow Z$ and $Y \not\in O$
* $X \leftarrow Y \rightarrow Z$ and $Y \not\in O$
* $X \rightarrow Y \leftarrow Z$ and $Y$ (or any of its descents) is observed

$Q$ and $W$ are **$d$-separated** given $O$ if they are not $d$-connected.

### Parameter estimation

Suppose we have a probabilistic model parameterized some unknown parameters $\theta$, and we want to estimate $\theta$ from data $\D = \{\x^{j}\}_{j=1}^n$. Maximum likelihood estimation (MLE) is a common and general approach. We solve
$$
\max_\theta L(\theta; \D) \quad \text{where} \quad L(\theta; \D) = \prod_{j=1}^n p_\theta(\x^j)
$$
Equivalently, one take maximize
$$
\log L(\theta; \D) = \sum_{j=1}^n \log p_\theta(\x^j)
$$
Using the factorization of Bayesian networks, we have
$$
\log L(\theta; \D) = \sum_{j=1}^n \log \prod_{v \in V} p_\theta(x_v^j \given A_v^j) = \sum_j \sum_v \log p_\theta(x_v^j \given A_v^j)
$$
If $x_1, \dots, x_d$ are discrete variables, and $\theta$ is a lookup table $\theta = \{\theta_{x_v|A_v}\}$, this becomes
$$
\log L(\theta; \D) = \sum_v \sum_{x_v} \sum_{A_v} \#(x_v, A_v) \log \theta_{x_v|A_v}
$$
where $x_v$ ranges over the values that $x_v$ can take, $A_v$ ranges over values that $A_v$ can take, and $\#(x_v, A_v)$ counts the number of occurrences of $(x_v, A_v)$ in the dataset $\D$.


## Markov networks

A **Markov network**, aka **Markov random field** (MRF), is a graphical model where the graph is undirected. Markov networks may contain cycles, unlike Bayesian networks.

According to the Hammersley-Clifford theorem, any strictly positive density $p(x_1, \dots, x_n)$ can always be factorized as
$$
p(x_1, \dots, x_n) = \frac{1}{Z} \prod_{C \in \cC} \phi_C(x_C)
$$
 where

* $\cC$ is the set of *cliques* (fully connected subgraphs) of the graph. $C \in \cC$ refers to a single clique.
* $x_C = \{x_i : i \in C\}$
* $\phi_C$ is the **factor** associated with the clique $C$. Factors are required to be positive.
* $Z = \int \prod_{C \in \cC} \phi_C(x_C) \d \mathbf{x}$ is the **partition function**, a normalizing constant which ensures the distribution sums to $1$.

### Parameter estimation

We can write a Markov network as an exponential family:
$$
\prod_{C \in \cC} \phi_C(x_C) = \exp\left(\sum_{C \in \cC} \log \phi_C(x_C)\right) = \exp(\theta^\top f(x))
$$

Then use gradient ascent on the (concave!) log-likelihood
$$
\log p(\theta; \D) = \frac{1}{|\D|} \sum_{x \in \D} \theta^\top f(x) - \log Z(\theta)
$$
The gradient of the first term is easy. The gradient of the log-partition function is given by
$$
\nabla_\theta \log Z(\theta) = \E_{x \sim p}[f(x)]
$$


## Inference

### Exact inference in trees via dynamic programming

We consider inference on a factor graph. Bayesian networks can be represented as factor graphs where each factor corresponds to a node with its parents, and Markov networks can be represented as factor graphs where each factor corresponds to a node with its neighborhood.

The variable elimination algorithm proceeds by eliminating variables one at a time. For each variable $v = 1, \dots, |V|$,

1. Multiply all factors which contain $v$ to obtain a factor product. For example, two factors $\phi_1(a,b)$ and $\phi_2(b,c)$ can be combined to form $\phi_3(a,b,c) = \phi_1(a,b)\phi_2(b,c)$.
2. Marginalize out $x_v$ to obtain a new factor $\tau$. For example, a factor $\phi(x,y)$ is marginalized to obtain $\tau(x) = \sum_y \phi(x,y)$
3. Replace the factors with $\tau$

To answers conditional questions such as $\P(Y \given E = e)$, we can simply run variable elimination twice to compute $\P(Y, E = e)$ and $\P(E = e)$.

For a general graph, it is NP-hard to determine the optimal ordering, but for a tree, for computing $p(x_v)$ it is best to root the tree at $v$ and iterate in post-order (i.e. a node is always visited after its children).

#### Message-passing algorithms

Variable elimination can be interpreted as message-passing. How do we determine when to send a message? A node $x_i$ sends a message to its neighbor $x_j$ when it has received messages from all its neighbors besides $x_j$. All messages will have been sent after $2|E|$ steps, since each edge receives a message only twice (once in each direction).

The **sum-product message-passing** algorithm, also known as **belief propagation**, is used for computing marginals, $p(x_v)$. Whenever node $i$ is ready to transmit to node $j$, send the message
$$
m_{i \to j}(x_j) = \sum_{x_i} \phi(x_i) \phi(x_i,x_j) \prod_{\ell \in N(i) \setminus j} m_{\ell \to i}(x_i)
$$
where $N(i)$ is the neighbors of $i$. Once we have computed all the messages, marginal's can be computed easily via
$$
p(x_i) \propto \phi(x_i) \prod_{\ell \in N(i)} m_{\ell \to i}(x_i)
$$
The **max-product message-passing** algorithm is used for MAP estimation, i.e. (arg)$\max_{\x} p(\x)$. It operates in the same way, but rather than summing  over the possible values of $x_i$, you take the max.

### Loopy belief propagation, Bethe free energy

Loopy belief propagation is essentially the same algorithm as belief propagation, but applied to cyclic graphs rather than acyclic graphs. There is no longer a guarantee of convergence, or of obtaining the correct probabilities, but empirically it still works reasonably well.

### Variational inference

Suppose we are given a probability distribution $p(\x) \propto \pye(\x) := \prod_k \phi_k(x_k)$ with factors $\phi_k$. That is $p(\x) = \frac{\pye(\x)}{Z}$, where $Z = \sum_{\x} \prod_k \phi_k(x_k)$ is the normalizing constant. Performing inference is hard because we must compute $Z$.

In **variational inference** (VI), we attempt to find the distribution $q$ from some set $Q$ of distributions, such that the divergence $\KL{q}{p}$ is minimized. Note that
$$
\begin{align*}
\KL{q}{p} &= \sum_{\x} q(\x) \log \frac{q(\x)}{p(\x)} \\
&= \sum_{\x} q(\x) \log \frac{q(\x)Z}{\pye(\x)} \\
&= \underbrace{\sum_{\x} q(\x) \log \frac{q(\x)}{\pye(\x)}}_{J(q)} + \log Z
\end{align*}
$$

Note that, since KL divergence is non-negative,
$$
\log Z = \KL{q}{p} - J(q) \ge -J(q) 
$$
That is, $-J(q)$ is a lower bound on the log-partition function. Therefore it is called the **variational lower bound** or **evidence lower bound** (ELBO).

#### Mean-field VI

The **mean-field approximation** factorizes $q$ across its variables:
$$
q(\x) = \prod_j q_j(x_j)
$$

Under a mean-field approximation, the variational optimization problem $\min_q J(q)$ can be efficiently solved by coordinate descent, as it has a closed-form solution:
$$
\log q_j(x_j) \leftarrow \E_{q_{-j}}[\log\pye(\x)] + \text{const}
$$


### Sampling

#### Monte Carlo integration

To approximate the integral $\int p(x) f(x) \d x$ where $p(x)$ is a density, we may take a sample average
$$
\int p(x) f(x) \d{x} = \E_{x \sim p}[f(x)] \approx \frac{1}{n} \sum_{i=1}^n f(x_i) \quad \text{where} \quad x_i \iid p
$$
The law of large numbers guarantees that the sample average converges a.s. to the integral in the limit $n \to \infty$.

If the integral is not weighted by a density $p$, we can sample from another distribution and reweight accordingly. For a bounded domain $A$, we may use a uniform distribution with constant density $u_A(x) = \frac{1}{\operatorname{vol}(A)}$.
$$
\int_A f(x) \d{x} = \int_A u_A(x) \frac{f(x)}{u_A(x)} \d{x} = \operatorname{vol}(A) \E_{x \sim u_A} \left[f(x)\right]
$$

#### Importance sampling

Importance sampling (IS) is based on the following identity:
$$
\E_{x \sim p}[f(x)] = \int p(x) f(x) \d x = \int q(x) \frac{p(x)}{q(x)} f(x) \d x = \E_
{x \sim q}\left[\frac{p(x)f(x)}{q(x)}\right]
$$
The variance of the IS estimator is $\sigma_q^2/n$ where
$$
\begin{align*}
\sigma^2_q &:= \Var\left(\frac{p(x)f(x)}{q(x)}\right) \\
&= \E\left[\left(\frac{p(x)f(x)}{q(x)}\right)^2\right] - \mu^2 \tag{$\mu = \E_p[f(x)]$} \\
&= \int \frac{(p(x)f(x))^2}{q(x)} \d{x} - \mu^2
\end{align*}
$$

The proposal distribution that minimizes the variance is
$$
q^*(x) \propto p(x) \, |f(x)|
$$
Proof: Let $q$ be any density which is positive when $pf \ne 0$. Then
$$
\begin{align*}
\mu^2 + \sigma^2_{q^*} &= \int \frac{(p(x)f(x))^2}{q^*(x)} \d{x} \\
&= \E_{x \sim p}[|f(x)|] \int \frac{(p(x)f(x))^2}{p(x) |f(x)|} \d{x} \\
&= \E_{x \sim p}[|f(x)|]^2 \\
&= \E_{x \sim q}\left[\frac{p(x) |f(x)|}{q(x)}\right]^2 \\
&\le \E_{x \sim q}\left[\frac{p(x)^2 f(x)^2}{q(x)^2}\right] = \mu^2 + \sigma^2_q
\end{align*}
$$

#### Markov chain Monte Carlo

Markov chain Monte Carlo (MCMC) is a general class of algorithms based on constructing Markov chains whose stationary distribution is the distribution from which we wish to sample. A Markov chain on a finite state space has associated transition probabilities $T_{ij} = T(j \given i) = p(s_{t+1} = j \given s_t = i)$ which can be stacked into a row-stochastic matrix $T = [T_{ij}]$. The stationary distribution of a Markov chain with transition matrix $T$ can be represented as a row vector satisfying
$$
\pi T = \pi \quad \text{i.e.} \quad \pi_j = \sum_i \pi_i T_{ij}
$$
A sufficient condition for $\pi$ to be a stationary distribution for a Markov chain with transition probabilities $T$ is the **detailed balance** condition:
$$
\pi_i T_{ij} = \pi_j T_{ji} \quad \text{i.e.} \quad \pi(i) T(j \given i) = \pi(j) T(i \given j)
$$
Intuitively, this states that at equilibrium, the forward process is at equilibrium with the reverse process.

##### Metropolis-Hastings

The Metropolis-Hastings (MH) algorithm is a general procedure for constructing Markov chains that satisfy detailed balance, given a (potentially unnormalized) target density $p$ and a proposal distribution $q(x' \given x)$.

At each step, if the current state of the chain is $x^t = x$, the algorithm does the following:

1. Sample a candidate $x' \sim q(x' \given x)$

2. Compute acceptance probability
   $$
   A(x' \given x) = \min\left\{1, \frac{p(x') q(x \given x')}{p(x) q(x' \given x)}\right\}
   $$

​		Since we take the ratio $p(x')/p(x)$, we can use unnormalized versions.

3. Accept $x^{t+1} = x'$ with probability $A(x' \given x)$, otherwise $x^{t+1} = x$

Why does this satisfy detailed balance? Assume w.l.o.g. (the other case is symmetric) that $A(x' \given x) < 1$, then $A(x' \given x) = \frac{p(x') q(x \given x')}{p(x) q(x' \given x)}$, and $A(x \given x') = 1$, so
$$
p(x')q(x \given x')A(x \given x') = p(x)q(x' \given x)A(x' \given x)
$$

#### Gibbs sampling

Gibbs sampling is an algorithm which is appropriate when it is easy to sample from the conditional distirbution of each variable given the other variables. At each step, if the current state of the chain is $x_t = x$, the algorithm does the following:

1. Sample $x'_i \sim p(x_i \given x_{-i})$. Note that this distribution only depends on the Markov blanket of variable $i$.
2. Set $x^{t+1} = (x_1, \dots, x'_{i}, \dots, x_d)$

Gibbs sampling can be viewed as a special case of MH where the proposal distribution changes only a single coordinate:
$$
q(x'_i, x_{-i} \given x_i, x_{-i}) = p(x_i \given x_{-i})
$$
The acceptance probability is always 1.

### Sequential Monte Carlo

Consider a Markov process with transition probabilities $p(x' \given x)$ and emission probabilities $p(y \given x)$. We are interested in (approximately) computing a few distributions:

* The posterior distribution $p(x_{1:t} \given y_{1:t})$, given by Bayes rule
  $$
  p(x_{1:t} \given y_{1:t}) \propto p(x_{1:t}) p(y_{1:t} \given x_{1:t}) = \prod_{\tau=1}^t p(x_\tau \given x_{\tau-1}) p(y_\tau \given x_\tau)
  $$
  where $p(x_1 \given x_0) = p(x_1)$.

* The marginal distribution $p(x_t \given y_{1:t})$ can be obtained by marginalizing over the posterior. It satisfies the following recursions:

  * Prediction:
    $$
    p(x_t \given y_{1:t-1}) = \int p(x_t \given x_{t-1}) p(x_{t-1} \given y_{1:t-1}) \d{x_{t-1}}
    $$

  * Updating:
    $$
    p(x_t \given y_{1:t}) \propto p(y_t \given x_t) p(x_t \given y_{1:t-1})
    $$

#### Particle filtering

Particle filtering combines sequential importance sampling with bootstrap resampling. Initially, sample $K$ particles $x_1^k \sim p(x_1)$ for $k \in [K]$, then repeat:

1. Sample $\xye_t^k \sim p(x' \given x_{t-1}^i)$ 
1. Evaluate importance weights $w_t^k = p(y_t \given \xye_t^k)$
1. Normalize importance weights across $k$
1. Resample (with replacement) $K$ particles $x_{1:t}^k$ from the set $\{(x_{1:t-1}^k, \xye_t^k)\}_{k=1}^K$ according to normalized importance weights

## Structure learning

### Chow-Liu trees

The Chow-Liu algorithm aims to find the maximum-likelihood tree-structured graph. It operates as follows:

1. Form a complete undirected graph in which the weights on edge $\{X,Y\}$ is $MI(X,Y)$
2. Find a maximum-weight spanning tree (using e.g. Kruskal or Prim algorithm)
3. Pick any node to be the root variable, and assign directions radiating outward

Why does this work? The log-likelihood decomposes into mutual information terms and entropy terms:
$$
\log p(\theta; \D) \propto \sum_i MI(X_i, X_{\operatorname{pa}(i)}) - \sum_i H(X_i)
$$
The entropy terms don't depend on the structure of the graph, so they can be ignored. So it suffices to maximize the mutual informations. But since we are assuming a tree structure,
$$
\sum_i MI(X_i, X_{\operatorname{pa}(i)}) = \sum_{(i,j) \in E} MI(X_i, X_j)
$$

