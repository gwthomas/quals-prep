$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\H}{\mathcal{H}}
\newcommand{\inner}[1]{\langle #1 \rangle_{\H}}
\newcommand{\biginner}[1]{\left\langle #1 \right\rangle_{\H}}
$$



# Kernels

![Screen Shot 2021-12-15 at 1.16.14 PM](/Users/garrett/Desktop/Screen Shot 2021-12-15 at 1.16.14 PM.png)

## Definition

A kernel is a function $K : \X \times \X \to \R$ which is

* Symmetric: $K(x,y) = K(y,x)$
* Positive definite: $\sum_{i,j} c_i c_j K(x_i, x_j) \ge 0$ for all choices of $\{c_i\}$ and $\{x_i\}$

A sufficient condition for $K$ to be a kernel is if you can write $K(x, y) = \inner{\phi(x), \phi(y)}$ for some feature map $\phi$.

## Types of kernels

### Polynomial

$$
K(x,y) = (x^\top y + c)^d
$$

Parameters are the offset $c$ and degree $d$.

### Radial basis function (RBF)

$$
K(x,y) = \exp(-\frac{\|x-y\|^2}{2\sigma^2})
$$

Parameter is the scale $\sigma$.

## Reproducing kernel Hilbert space

A Hilbert space $\H$ of real-valued functions on a set $\X$ is said to be a **reproducing kernel Hilbert space** (RKHS) if the evaluation functional
$$
L_x : f \mapsto f(x)
$$
is continuous at any $f \in \H$, or equivalently, if $L_x$ is a bounded operator on $\H$, i.e. there exists $M_x > $ such that
$$
|L_x(f)| = |f(x)| \leq M_x \|f\|_\H
$$
where $\|f\|_\H = \sqrt{\langle f, f \rangle_\H}$ is the norm induced by the inner product of $\H$.

The *Riesz representation theorem* guarantees (for each $x \in \X$) the existence of a function $\Phi_x \in \H$ with the property that (for all $f : \X \to \R$)
$$
f(x) = L_x(f) = \inner{f, \Phi_x}
$$
Furthermore, we see that
$$
\Phi_x(y) = L_y(\Phi_x) = \inner{\Phi_x, \Phi_y}
$$
Hence we define $K : \X \times \X \to \R$ by
$$
K(x,y) = \inner{\Phi_x, \Phi_y}
$$
and we observe that, as a result of all this,
$$
f(x) = \inner{f, \Phi_x} = \inner{f, K(x, \cdot)}
$$
Note that $K$ is symmetric by definition, and positive definite as
$$
\sum_{i,j} c_i c_j K(x_i, x_j) = \sum_{i,j} c_i c_j \inner{\Phi_{x_i}, \Phi_{x_j}} = \biginner{\sum_i c_i \Phi_{x_i}, \sum_j c_j \Phi_{x_j}} = \left\|\sum_i c_i \Phi_{x_i}\right\|_\H^2 \ge 0
$$
