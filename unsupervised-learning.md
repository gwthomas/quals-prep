# Unsupervised Learning

## K-means clustering

The goal is to find a set of cluster centers and assignments which minimize the within-cluster sum of squares:
$$
\min_{S} \sum_{k=1}^K \sum_{i \in S_k} \|x_i - \mu_k\|_2^2
$$
The usual algorithm consists of two alternating steps, after randomly initializing cluster centers:

1. Assign each point to its closest cluster center
2. Recompute each cluster center by averaging the observations assigned to that cluster

## Expectation maximization

Expectation maximization algorithms alternate between two steps:

1. E-step: Construct objective $Q(\theta \given \theta_t) = \E_{Z|X,\theta_t}[\log L(\theta; X, Z)]$ where $L(\theta; X, Z) = p(X, Z \given \theta)$ is the likelihood function
2. M-step: $\theta_{t+1} = \arg\max Q(\theta \given \theta_t)$ 

Alternatively, EM can be viewed as coordinate ascent on the free energy
$$
F(q, \theta) = \E_q[\log L(\theta; X, Z)] + H(q) = -\KL{q}{p_{Z|X}} + \log L(\theta; X)
$$
As an example, consider clustering using a mixture of Gaussians. The latent variables $z_i$ represent which cluster the $i$th example came from.

## PCA

In PCA, one aims to find the (orthogonal) set of unit vectors such that projecting onto each gives the maximal variance. Intuitively, these are the directions that capture the most information about the data.

Assume we are given a data matrix $X$. We first center the matrix by subtracting the mean value from each column, so that now the column means are zero.

The first weight vector is given by maximizing the variance of the projected points:
$$
u_1 = \arg\max_{\|u\|_2 = 1} \sum_i (u\T x_i)^2 = \arg\max_{\|u\|_2 \le 1} \|Xu\|_2^2 = \arg\max_{\|u\|_2 \le 1} u\T X\T Xu
$$
Knowledge of linear algebra tells us that the maximizer is the unit eigenvector corresponding to the largest eigenvalue of $X\T X$.

Subsequent weight vectors follow a similar calculation, with the additional constraint of orthogonality to all previous vectors. Thus we end up with an orthonormal basis of eigenvectors. (Such a basis always exists because $X\T X$ is a symmetric matrix.)

Typically we only keep a relatively small number of weight vectors. Given a new point $x$, it can be projected to a lower-dimensional space:
$$
\tilde{x} = (u_1\T x, u_2\T x, \dots, u_k\T x)
$$

## Matrix factorization

Matrix factorization seeks an approximate decomposition of a given matrix $X$ into constituent matrices $X \approx UV$, such that $\|X - UV\|$ is minimized. In some cases, $U$ and $V$ are restricted to have positive entries.

This can be used for collaborative filtering tasks where each user has an associated vector $u_i$, and similarly each product has an associated vector $v_j$, and the model's score for how much user $i$ "likes" $v_j$ is modeled as $u_i\T v_j$. However, $X$ is sparse because we cannot observe every user's response to every product. 

## Variational autoencoders

A VAE is a latent variable model with latent $z_i$ and observed $x_i$ for each example $i$. We assume knowledge of a prior $p(z)$, and specify the conditional distribution $p(x \given z)$ using a parametric model $p_\theta(x \given z)$. 

The marginal likelihood
$$
p_\theta(x) = \int p(z)p_\theta(x \given z) \d{z}
$$
is intractable to compute in general because it requires integrating over the latent variable.

Instead, by introducing a variational approximation to the posterior, $q_\phi(z \given x)$, a tractable lower bound on the marginal log likelihood can be obtained:
$$
\begin{align*}
\log p_\theta(x) &= \log \int q_\phi(z \given x) \frac{p(z) p_\theta(x \given z)}{q_\phi(z \given x)} \d{z} \\
&\ge \int q_\phi(z \given x) \log \frac{p(z)p_\theta(x \given z)}{q_\phi(z \given x)} \d{z} \tag{Jensen} \\
&= \underbrace{\E_{q_\phi(z \given x)}[\log p_\theta(x \given z)]}_{\text{reconstruction}} - \underbrace{\KL{q_\phi(z \given x)}{p(z)}}_{\text{regularization}}
\end{align*}
$$
Typically in a VAE, the prior and posterior are chosen to be Gaussian so that 

* samples from $q_\phi$ are differentiable w.r.t. $\phi$ via the reparameterization trick: $\cN(\mu, \sigma^2) = \mu + \sigma \cN(0,1)$
* the KL divergence can be computed analytically
