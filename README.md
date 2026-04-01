# Risk-Aware Stability-Preserving Design of Neural Controllers with Conformal Verification

## Overview

This repository contains the implementation of a framework for designing neural network controllers that guarantee **closed-loop stability by design**, promote **probabilistic safety** using risk-aware training, and provide **distribution-free finite-sample safety certificates** through conformal prediction. The approach combines the Performance Boosting (PB) framework with CVaR-based training and conformal verification to enable safe deployment of learning-based controllers for nonlinear systems.

## Key Contributions

- **Stability-preserving neural network architecture**: Uses Internal Model Control (IMC) parameterization to guarantee closed-loop ℓₚ-stability for any controller parameters, eliminating the need for stability constraints during optimization.
- **Risk-aware CVaR training**: Replaces intractable chance constraints with a smooth Conditional Value-at-Risk (CVaR) surrogate that promotes safe behavior for worst-case disturbances.
- **Distribution-free conformal verification**: Provides two-sided finite-sample probabilistic bounds on safety certificates using the DKWM inequality, with no assumptions on the underlying disturbance distribution.
- **Iterative design feedback**: Leverages conformal bounds to guide controller redesign, distinguishing between poor design and insufficient data collection.

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DecodEPFL/Conformal_PB.git
   cd Conformal_PB
   ```

2. **Create a virtual environment** (Python 3.8+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch numpy matplotlib jupyter
   ```

### Running the Experiments

The main reproducibility notebook is `main.ipynb`, which contains:
- CVaR-Lagrangian training phase
- ERM-RBF baseline comparison with hyperparameter search
- Conformal verification and safety certification
- Pareto frontier visualization
- Trajectory plots and analysis

To run:
```bash
jupyter notebook main.ipynb
```

The notebook will train models, generate figures in a `figures/` directory, and save trained checkpoints.

## Project Structure

```
conformal_pb/
├── main.ipynb                      # Main reproducibility notebook with full pipeline
├── ren.py                          # REN (Recurrent Equilibrium Networks) controller implementation
├── robot.py                        # Robot dynamics and PD base controller
├── dataset.py                      # Data generation utilities
├── losses_and_wrappers.py          # Loss functions and CVaR/Lagrangian wrappers
├── training_function.py            # Training loop and optimization
├── performance_boosting.py         # Performance Boosting closed-loop architecture
├── plot_functions.py               # Visualization utilities for trajectories
├── *_checkpoints/                  # Pre-trained model checkpoints
└── README.md                       # This file
```

### Key Files

- **`ren.py`**: Implements the contractive REN, a stable-by-design neural network that parameterizes the controller in the IMC framework.
- **`robot.py`**: Defines the nonlinear robot dynamics (point-mass with drag) and base PD controller.
- **`losses_and_wrappers.py`**: Contains `PBLoss` (tracking + collision avoidance), `ERMWrapper`, `CVaRLossWrapper`, and `LagrangianCVaRLossWrapper` for different training objectives.
- **`training_function.py`**: Implements the min-max optimization loop for the Lagrangian-CVaR formulation.
- **`performance_boosting.py`**: Implements the IMC-based closed-loop system combining the stable controller with the plant.

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

## Results

The approach demonstrates that risk-aware CVaR-based training effectively promotes conformal safety verification. On the robot obstacle avoidance task:

- **Safety vs. Performance Tradeoff**: Our method achieves 1% collision frequency with mean tracking cost of 1.05, while the baseline suffers a 16% collision frequency at equivalent cost.
- **Conformal Certification**: The learned controller satisfies the 95% safety requirement with 95% confidence, verified on independent calibration data.
- **Finite-Sample Confidence**: Conformal bounds account for finite-sample uncertainty and guide iterative redesign decisions.

See `main.ipynb` for detailed results, Pareto curves, and trajectory visualizations.

## Authors & Acknowledgments

**Authors**: Laura Meroi, Sabri El Amrani, Danilo Saccani, Dario Piga, Giancarlo Ferrari-Trecate

**Affiliations**: 
- Institute of Mechanical Engineering, EPFL, Switzerland
- Dalle Molle Institute for Artificial Intelligence (IDSIA), SUPSI, Switzerland

This work was supported as part of NCCR Automation, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40 225155) and the NECON project (grant number 200021219431).
