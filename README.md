# Conformal_PB
## Implementation Details

### System and Base Controller Parameters
We consider a sampling time $T_s=0.05 \text{ s}$. The mobile robot has a mass $M=1.0$ and is subject to nonlinear drag with coefficients $b_1 = 1.0$ and $b_2 = 0.2$. We set the gains of the pre-stabilizing PD controller to $k_p=1.0$ and $k_d=1.0$.

### Experiment Details
We sample the initial position of the robot from a Gaussian Mixture Model (GMM) with means $\{(-2.0, -2.0), (1.0, -1.0), (0.0, 1.5)\}$, standard deviations $\{0.3, 0.1, 0.1\}$, and weights $\{0.5, 0.3, 0.2\}$. 

The environment contains two elliptical obstacles: the first is centered at $\mu_1=(0.0, 0.6)$ with radii $r_{1,x} = 0.8, r_{1,y}=0.1$; the second is centered at $\mu_2=(-1.0, -1.0)$ with radii $r_{2,x}=0.3, r_{2,y}=0.3$. 

The tracking cost is parameterized with weight matrices $Q = 4 I_4$ and $R = 10^{-6} I_2$. The experiments are conducted over a finite time horizon of $T=500$ steps.

### Training Hyperparameters
We trained a contractive REN (Revay et al., 2023) with internal linear and nonlinear state dimensions set to 4, and initialized its parameters from a zero-mean Gaussian with standard deviation 0.5. Training was conducted for 3000 episodes with independent batches of 200 trajectories.

We optimized the primal parameters $\{\theta, \tau\}$ and the dual variable $\lambda$ using the Adam optimizer. The REN parameters $\theta$ used a learning rate of $10^{-3}$, while the CVaR quantile parameter $\tau$ used a faster learning rate of $10^{-2}$ to promote timescale separation (Borkar, 2008).

Let us define the empirical safety constraint violation $v$ as the difference between the empirical Conditional Value-at-Risk ($\widehat{\text{CVaR}}$) of the collision cost and the safety threshold $\bar{\tau}_{\text{safe}}$ over a set of 200 trajectories (sampled independently from the training and validation datasets):

$$
v = \left( \tau + \frac{1}{200 \alpha} \sum_{i=1}^{200} \max(c_i - \tau, 0) \right) - \bar{\tau}_{\text{safe}}.
$$

For the dual ascent step, we used a base dual learning rate $\eta_\lambda^{\text{base}} = 10^{-2}$ and adapted it based on $v$ as:

$$
\eta_\lambda = 
\begin{cases} 
0.2 \eta_\lambda^{\text{base}}, & |v| \le 0.02 \quad \text{(near feasibility)} \\
2.0 \eta_\lambda^{\text{base}}, & v > 0.1 \quad \text{(clear violation)} \\
\eta_\lambda^{\text{base}}, & \text{otherwise.}
\end{cases}
$$

Thus, $\lambda$ is updated aggressively when constraints are strongly violated and conservatively near the feasible boundary. In addition, the dual is only updated every $5$ steps of the inner minimization over $\{\theta, \tau\}$ to promote timescale separation.
