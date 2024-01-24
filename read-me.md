<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

## Forming a convex envelope $\hat{f_i}$ of $f_i$

We aim to find a point $w$ along the sigmoid such that

$$r'(w)=f_i'(w)$$

for a line $r$ starting at $l$ and intersecting $f_i$ at $w$.

Finding such a line is an easy procedure. If we assume $w \geq z$, then $f_i'$ is strictly decreasing on the interval $[w,u]$. This means we can create a simple rule for adjusting $w$ at successive iterations:

$$w_{i+1} = w_i(1\pm\epsilon_i)$$
$$\epsilon_{i+1} = \mu \epsilon_i$$

After finding $w$, the envelope $r$ can be easily constructed.

## Barrier Method for Solving Convex Approximations

For a search region $\mathcal{Q}$, we have 

$$\underset{x}{max} \sum_i^n c_i s_i(x_i)$$
$$s.t. \; x \in S \cap \mathcal{Q} $$

Specifically,

$$\underset{x}{max} \sum_i^n c_i s_i(x_i)$$
$$s.t. \; d^Tx = r$$
$$x \geq 0$$
$$l_i \leq x_i \leq u_i \; \forall i$$

Where $l_i$ and $u_i$ represent the 'lower' and 'upper' coordinates of the hyperrectangle $\mathcal{Q}$ along the ith dimension.

The convex approximation, with envelopes $\hat{s}_i$,

$$\underset{x}{max} \sum_i^n c_i \hat{s}_i(x_i)$$

must then be solved via the following barrier problem:

$$\underset{x}{max} \sum_i^n c_i \hat{s}_i(x_i)- \left( \sum_i^n \frac{ln(-x_i+u_i)}{t}+\sum_i^n \frac{ln(x_i-l_i)}{t}+\sum_i^n \frac{ln(x_i)}{t} \right)$$
$$s.t. \; d^Tx=r$$

If the piecewise linear bound methods is used, we must only solve a linear programming problem, which has $m_i$ tangent lines defining the bound for function $i$:

$$\underset{y}{max} \sum_i^n \;y_i$$
$$s.t. \; d^Tx=r$$
$$\; x_i\leq u_i \; \forall i$$
$$\; x_i\geq l_i \; \forall i$$
$$a_{ij} x_i \leq b_{ij} \; \forall i, \; \forall j \in \{1,...,m_i\}$$

