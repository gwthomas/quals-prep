$$
\newcommand{\d}{\,\mathrm{d}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\given}{\,|\,}
\newcommand{\D}{\mathcal{D}}
\newcommand{\Dir}{\operatorname{Dir}}
\newcommand{\Multi}{\operatorname{Multi}}
$$



# Bayesian Methods

## Prior, likelihood, posterior

Bayes rule:
$$
p(\theta \given x) = \frac{p(\theta)p(x \given \theta)}{p(x)}
$$
Or, omitting the normalizing factor,
$$
\underbrace{p(\theta \given x)}_{\text{posterior}} \propto \underbrace{p(\theta)}_{\text{prior}} \underbrace{p(x \given \theta)}_{\text{likelihood}}
$$

## Conjugate priors

Conjugate priors are priors which (for a given likelihood) give closed-form expressions for the posterior.

### Gaussian



### Dirichlet

Let $p \sim \operatorname{Dir}(\alpha_1, \dots, \alpha_K)$ prior and $X \sim \operatorname{Multinomial}(p, n)$. The posterior over $\{p_i\}$ takes the form
$$
\begin{align*}
f(\alpha_1, \dots, \alpha_K \given x_1, \dots, x_K) &\propto f(p_1, \dots, p_K \given \alpha_1, \dots, \alpha_K) p(x_1, \dots, x_k \given p_1, \dots, p_K) \\
&\propto \prod_k p_k^{\alpha_k-1} \prod_k p_k^{x_k} \\
&= \prod_k p_k^{\alpha_k + x_k - 1}
\end{align*}
$$
which, by inspection, is a $\operatorname{Dir}(\alpha_1 + x_1, \dots, \alpha_K + x_K)$ distribution.

## Marginal likelihood, model selection

Consider a *model* $M$ which corresponds to a family of probability distributions indexed by $\theta$. The marginal likelihood of the data under $M$ is 
$$
p(\D \given M) = \int p(\D \given \theta, M) \, p(\theta \given M) \d\theta
$$

## Hypothesis testing

Bayesian hypothesis testing, like frequentist hypothesis testing, aims to evaluate the "reasonable"ness of a hypothesis in some sense. The null hypothesis $H_0$ is compared against the alternative hypothesis, $H_1 = \lnot H_0$.

Compare posterior probabilities conditioned on observed data:
$$
p(H_i \given \D) = \frac{p(H_i) p(\D \given H_i)}{p(\D)}
$$
We are interested in the ratio
$$
\underbrace{\frac{p(H_0 \given \D)}{p(H_1 \given \D)}}_{\text{posterior odds}} = \underbrace{\frac{p(H_0)}{p(H_1)}}_{\text{prior odds}} \underbrace{\frac{p(\D \given H_0)}{p(\D \given H_1)}}_{\text{Bayes factor}}
$$
where the numerator and denominator can be computed as above ("marginal likelihood"), integrating over the parameters of the model.

The prior probabilities are typically chosen to be $p(H_0) = p(H_1) = \frac{1}{2}$ unless the tester has prior knowledge.

## Bayesian mixture model

A general mixture of $K$ distributions $\{p_k\}$ with coefficients $\phi_k$ (which must be nonnegative and sum to one):
$$
p(x) = \sum_k \phi_k p_k(x)
$$
A Bayesian mixture model is a mixture model in which the mixture weights $\phi_k$ are themselves random variables.

## Latent Dirichlet allocation

**Latent Dirichlet allocation** (LDA) is a topic model that consists of the following:

* $M$ denotes the number of documents, with document $i$ having $N_i$ words
* $\alpha$ is the parameter of the Dirichlet prior on the per-document topic distributions. Typically $\alpha < 1$ to induce sparsity.
* $\beta$ is the parameter of the Dirichlet prior on the per-topic word distribution. Typically $\beta$ is also sparse.
* $\theta_i \sim \Dir(\alpha)$ is the topic distribution for document $i$
* $\phi_k \sim \Dir(\beta)$ is the word distribution for topic $k$
* $z_{ij} \sim \Multi(\theta_i)$ is the topic for the $j$th word in document $i$
* $w_{ij} \sim \Multi(\phi_{z_{ij}})$ is the specific $j$th word in document $i$
