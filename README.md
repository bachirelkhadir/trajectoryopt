# Optimal control for obstacle avoidance

Data: radius $r > 0$, center $c \in \mathbb R^2$, starting point $a \in \mathbb R^2$, end point $b \in \mathbb R^2$

$$\min \int_0^1 \|u(t)\|^2 $$
s.t.
$$\dot x(t) = u(t)$$

$$x(0) = a, x(1) = b$$

$$x(t) \in [0, 1]^2 \setminus B(c, r)$$

## Moment method:
- Admissible space for $x$, $X := [0, 1]^1 \setminus B(c, r)$.
- Admissible space for control $u$, $U = \{ u \in \mathbb R^2 \; | \; \|u\| \le M\}$.
- Overall space $S = X \times U \times [0, 1]$


Morally, $x(t) \in A, u(t) \in B, t \in D$, so that:

$$\mu(A, C, D) = \int_{[0, 1] \cap D} 1_{A\times C}(x(t), u(t)) {\rm d}t$$

Measure defined on $X \times [0, 1]$:

$$\mu_0 = \delta_{x = a} \delta_{t= 0}, \mu_1 = \delta_{x = b} \delta_{t = 1},$$
so that, e.g.,
$\langle \mu_0, \phi(t, x) \rangle = \phi(0, a)$.


For any test function $\phi: [-1, 1]^2 \times  [0,1] \rightarrow \mathbb R, (x, t) \mapsto \
\phi(x, t)$, then

$$\phi(x(1), 1) - \phi(x(0), 0) = \int_0^1 [\partial_t \phi(t, x(t)) + \nabla_x \phi(t, x(t)) \cdot u(t)] {\rm d}t$$
$$= \int_S [\partial_t \phi(t, x) + \nabla_x \phi(t, x) \cdot u] {\rm d}\mu(x, u, t)$$

i.e.
$$\langle \mu_1, \phi \rangle - \langle \mu_0, \phi \rangle =  \int_S [\partial_t \phi(x, t) + \nabla_x \phi(x, t) \cdot u] {\rm d}\mu(x, u, t) := F_\mu(\phi) $$

For $\phi = x^\alpha t^k$, we get

$$b^\alpha 1^k  - a^\alpha 0^k =  \int_S [\partial_t \phi(t, x) + \nabla_x \phi(t, x) \cdot u] {\rm d}\mu(x, u, t)$$
