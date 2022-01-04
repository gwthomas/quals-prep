$$
\newcommand{\bigO}[1]{\mathcal{O}\left(#1\right)}
\newcommand{\T}{^{\top}}
\newcommand{\inv}{^{-1}}
\newcommand{\inner}[1]{\langle #1 \rangle}
\newcommand{\subjectto}{\quad\text{subject to}\quad}
\newcommand{\cL}{\mathcal{L}}
$$

# Optimization

## Convex analysis for differentiable functions

### Convexity

Recall that a differentiable function $f$ is convex if
$$
f(y) \ge f(x) + \nabla f(x)\T(y-x)
$$
This says that $f$ lies above every tangent line approximation.

### Strong convexity

A differentiable function $f$ is $\alpha$-strongly convex if
$$
f(y) \ge f(x) + \nabla f(x)\T(y-x) + \frac{\alpha}{2} \|x-y\|_2^2
$$
This says that $f$ lies above the tangent line approximation plus a quadratic term that depends on how far you moved from the point of approximation. That is, $f$ is "at least as curved" as a quadratic.

In particular, one consequence is
$$
f(x) - f(x^*) \ge \frac{\alpha}{2} \|x-x^*\|_2^2
$$

### Smoothness

Conversely, $f$ is $\beta$-smooth if
$$
f(y) \leq f(x) + \nabla f(x)\T(y-x) + \frac{\beta}{2}\|x-y\|_2^2
$$
That says that $f$ is bounded above by the tangent line approximation plus a quadratic.

## Algorithms and convergence rates

### Gradient descent

Gradient descent on a differentiable function $f : \R^n \to \R$ proceeds according to
$$
x_{t+1} = x_t - \alpha_t \nabla f(x_t)
$$
Summary of rates for gradient descent:

* General: error decreases as $\bigO{\frac{1}{\sqrt{T}}}$ 
* $\alpha$-strongly convex: $\bigO{\frac{1}{\alpha T}}$
* $\beta$-smooth: $\bigO{\frac{\beta}{T}}$
* Strongly convex and smooth: $\bigO{c^k}$ for some $0 < c < 1$, so only $\bigO{\log(1/\epsilon)}$ iterations needed

#### Sketches

##### GD on strongly convex functions



##### GD on smooth functions

On a $\beta$-smooth function,
$$
\begin{align*}
f(x_{t+1}) &\le f(x_t) + \nabla f(x_t)\T (x_{t+1} - x_t) + \frac{\beta}{2}\|x_{t+1} - x_t\|_2^2 \\
&= f(x_t) - \alpha_t \|\nabla f(x_t)\|_2^2 + \frac{\beta\alpha_t^2}{2}\|\nabla f(x_t)\|_2^2 \\
&= f(x_t) + \left(\frac{\beta\alpha_t^2}{2} - \alpha_t\right) \|\nabla f(x_t)\|_2^2
\end{align*}
$$
To minimize this, we take $\alpha_t$ to minimize the function $\alpha \mapsto \frac{1}{2}\beta\alpha^2 - \alpha$, which is minimized at $\alpha_t = \frac{1}{\beta}$, yielding
$$
f(x_{t+1}) \le f(x_t) - \frac{1}{2\beta} \|\nabla f(x_t)\|_2^2
$$
That is, the function value decreases by at least this quantity at every iteration.

Ultimately (this requires more work to show),
$$
f(x_T) - f(x^*) \le \frac{\beta \|x_0 - x^*\|_2^2}{2T} = \bigO{\frac{\beta}{T}}
$$

### Newton's method

Newton's method (as applied to optimization problems) attempts to find zeros of the function $x \mapsto \nabla f(x)$, i.e., critical points where $\nabla f(x) = 0$.

It operates by finding the minimum of a quadratic approximation to the objective, centered at the current iterate:
$$
x_{t+1} = \arg\min_x \left\{ f(x_t) + \nabla f(x_t)\T(x - x_t) + \frac{1}{2} (x - x_t)\T \nabla^2 f(x_t)(x-x_t) \right\}
$$
The gradient of the above expression must vanish:
$$
0 = \nabla f(x_t) + \nabla^2 f(x_t)(x_{t+1} - x_t)
$$
Solving for $x_{t+1}$ yields the update
$$
x_{t+1} = x_t - (\nabla^2 f(x_t))\inv \nabla f(x_t)
$$

Newton's method takes $T = C \frac{f(x_0) - f^*}{\alpha/\beta^2} + \log_2\log_2(\frac{2\alpha^3/L^2}{\epsilon})$ iterations to obtain $\epsilon$ error, where $L$ is Lipschitz constant of Hessian.

## Stochastic gradient descent

Convergence is like
$$
\E[f(x_T)] - f^*  = \bigO{\frac{1}{\sqrt{T}}}
$$

## Types of programs

### Linear programming

A linear program has a linear objective and linear constraints:
$$
\min_x c\T x \subjectto Ax \le b, \quad x \ge 0
$$

### Quadratic programming

A quadratic program has a quadratic objective and linear constraints:
$$
\min_x \frac{1}{2} x\T Qx + c\T x \subjectto Ax \le b
$$


### Semidefinite programming

A semidefinite program has the form
$$
\min_X \inner{C, X} \subjectto  \inner{A_k, X} \le b_k ~ (\forall k), \quad X \succeq 0
$$

where all these variables are symmetric matrices. The inner product is given by $\inner{A, B} = \tr(A\T B) = \sum_{i,j} A_{ij}B_{ij}$. Note that this is very similar to the linear programming formulation above, except the nonnegativity constraint becomse a positive semidefinite constraint.

## Lagrangian duality

We wish to solve constrained problems of the form
$$
\begin{align*}
	\min_{x} ~&~ f(x) \\\text{s.t.} ~&~ f_i(x) \le 0 & i &= 1, \dots, m \\&~ h_j(x) = 0 & j &= 1, \dots, n
\end{align*}
$$
Recall the Lagrangian
$$
\cL(x; \lambda, \nu) = f(x) + \sum_{i} \lambda_i f_i(x) + \sum_j \nu_j h_j(x)
$$
where $\lambda_i \ge 0$ and $\nu_j \in \R$. This is a way to move the constraints into the objective. Notice that the original constrained problem can be represented as
$$
p^\star = \min_x \max_{\lambda \ge 0,\, \nu} \cL(x; \lambda, \nu)
$$
known as the *primal problem*. The *dual problem* is the result of swapping the min and max:
$$
d^\star = \max_{\lambda \ge 0,\,\nu} \underbrace{\inf_x \cL(x, \lambda, \nu)}_{g(\lambda, \nu)}
$$
Note that the dual function $g(\lambda, \nu)$ is concave: for each fixed $x$, the Lagrangian is an affine (and therefore concave) function of $(\lambda, \nu)$, and the pointwise minimum of concave functions is concave.
Another crucial fact about the dual function is that it provides a lower bound on the optimal value of the primal problem.

> **Proposition.** For all $\lambda \ge 0$ and all $\nu$, it holds that $g(\lambda,\nu) \le p^\star$.

*Proof.* Let $x^\star$ be any optimal solution of the primal problem. Then $f_i(x^\star) \le 0$ for all $i$, $h_j(x^\star) = 0$ for all $j$, and $f(x^\star) = p^\star$, so assuming $\lambda \ge 0$, we have
$$
g(\lambda, \nu) = \inf_x \cL(x, \lambda, \nu) \le \cL(x^\star, \lambda, \nu) = f(x^\star) + \sum_{i=1}^m \underbrace{\lambda_i f_i(x^\star)}_{\le 0} + \sum_{j=1}^n \underbrace{\nu_j h_j(x^\star)}_0 \le f(x^\star) = p^\star
$$
as claimed.

We denote the optimal value of the dual problem by $d^\star$, observing that $d^\star \leq p^\star$ as a consequence of the previous proposition. Another way to say this is that the *duality gap*, $p^\star - d^\star$, is always nonnegative. If the duality gap is zero (i.e. $p^\star = d^\star$), it is said that *strong duality* holds; otherwise, *weak duality* holds.

#### Slater's condition

Slater's condition is a sufficient condition for strong duality in convex programs where the equality constraints are affine. It essentially requires that the interior of the feasible set is non-empty. That is, there exists $\tilde{x}$ such that $f_i(\tilde{x}) < 0$ strictly for all $i$, and $A\tilde{x} = b$.

## KKT conditions

In unconstrained optimization, a necessary and sufficient condition for global optimality is that $0 \in \partial f(x)$. In constrained optimization, this is no longer the case: points $x$ which satisfy $0 \in \partial f(x)$ may not be feasible! Therefore, we need new necessary and sufficient conditions for the constrained case.

Let $x^\star$ be primal optimal and $(\lambda^\star, \nu^\star)$ be dual optimal, with strong duality holding. The *Karush-Kuhn-Tucker (KKT) conditions* are necessary conditions, as follows:

* *Primal feasibility*: $f_i(x^\star) \le 0$ for all $i$, and $h_j(x^\star) = 0$ for all $j$

* *Dual feasibility*: $\lambda^\star_i \ge 0$ for all $i$

* *Stationarity*: $0 \in \partial_x \cL(x^\star, \lambda^\star, \nu^\star)$. If $f$, $f_i$, and $h_j$ are all differentiable, this can be written
  $$
  0 = \nabla f(x^\star) + \sum_i \lambda^\star_i \nabla f_i(x^\star) + \sum_j \nu^\star_j \nabla h_j(x^\star)
  $$
  This is because $x^\star$ must minimize $\cL(x, \lambda^\star, \nu^\star)$ over $x$.

* *Complementary slackness*: $\lambda^\star_i f_i(x^\star) = 0$ for all $i$.  Roughly this means that the optimal Lagrange multiplier is zero unless its associated constraint is active at the optimum. The reason this must hold is
  $$
  \begin{align*}
  f(x^\star) &= g(\lambda^\star, \nu^\star) \\
  &= \inf_x \cL(x, \lambda^\star, \nu^\star) \\
  &\le \cL(x^\star, \lambda^\star, \nu^\star) \\
  &= f(x^\star) + \sum_i \underbrace{\lambda^\star_i}_{\ge 0} \underbrace{f_i(x^\star)}_{\le 0} + \sum_j \nu^\star_j \underbrace{h_j(x^\star)}_0 \\
  &\leq f(x^\star)
  \end{align*}
  $$
  Indeed, by sandwiching, we see that all the inequalities must hold with equality. Thus
  $$
  \sum_i \lambda^\star_i f_i(x^\star) = 0
  $$
  Since each term in the sum is non-positive, every term must equal zero.

