# Reinforcement Learning Algorithms: From Tabular Methods to Advanced Actor–Critic

**A curated, experiment-driven collection of foundational and modern Reinforcement Learning (RL) algorithms implemented from scratch in PyTorch and NumPy, benchmarked across classic control and discrete environments.**

---
## 1. Overview
This repository aggregates implementations spanning value-based, policy-gradient, and deterministic continuous-control paradigms:

Algorithms included:
- Tabular: Q-Learning, SARSA
- Value-Based Deep RL: DQN, Double DQN (DDQN)
- Policy Gradient / Stochastic Actor–Critic: Vanilla Actor-Critic, Advantage Actor-Critic (A2C), Proximal Policy Optimization (PPO)
- Deterministic Actor–Critic for Continuous Control: DDPG, Twin Delayed DDPG (TD3)

Environments used (Gymnasium): FrozenLake-v1, Taxi-v3, CartPole-v1, Pendulum-v1, BipedalWalker-v3 (fallback to Pendulum if Box2D unavailable).

---
## 2. Mathematical Foundations
Below are the core update rules and objectives distinguishing the algorithms.

### 2.1 Tabular Methods
- Q-Learning (off-policy, bootstrapped):  
  Update: \(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\)
- SARSA (on-policy):  
  Update: \(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]\)

### 2.2 Deep Value-Based
- DQN: Minimize TD error \(L = (r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q_{\text{online}}(s,a))^2\) with Experience Replay + Target Network.
- Double DQN: Action selection via online network, evaluation via target:  
  \(y = r + \gamma Q_{\text{target}}(s', \arg\max_a Q_{\text{online}}(s',a))\)

### 2.3 Policy Gradient & Advantage Actor–Critic
- Policy Objective: \(J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]\)
- REINFORCE Gradient: \(\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]\)
- Actor–Critic Advantage: \(A_t = G_t - V(s_t)\); Actor Loss: \(-\log \pi_\theta(a_t|s_t) A_t\); Critic Loss: MSE \((V(s_t) - G_t)^2\)
- A2C (synchronous multi-step): n-step return:  
  \(R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})\) (bootstrapped if not terminal).

### 2.4 PPO (Clipped Surrogate)
Ratio: \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\)  
Objective: \(L^{\text{CLIP}} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]\) with entropy bonus and value loss.

### 2.5 Deterministic Actor–Critic (DDPG / TD3)
- Deterministic Policy: \(a = \mu_\theta(s)\)  
- Critic Target: \(y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))\)  
- Actor Gradient: \(\nabla_\theta J \approx \mathbb{E}[ \nabla_a Q_\phi(s,a) |_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)]\)
- TD3 Enhancements: (1) Twin Critics: take \(\min(Q_1, Q_2)\) to mitigate positive bias; (2) Target Policy Smoothing: add clipped noise to target actions; (3) Delayed Policy Updates: update actor less frequently than critic.

---
## 3. Why One Algorithm Over Another?
| Algorithm | Paradigm | Strengths | Limitations | When to Use |
|-----------|----------|-----------|-------------|-------------|
| Q-Learning | Tabular Off-Policy | Simple, convergence guarantees in finite MDPs | Not scalable to large/continuous spaces | Small discrete state spaces |
| SARSA | Tabular On-Policy | Safer (accounts for exploration policy) | Can under-explore optimal risky paths | When conservative estimates preferred |
| DQN | Value-Based Deep | Scales to large state spaces; replay + target stabilize | Overestimation bias | Discrete high-dimensional observations |
| Double DQN | Value-Based Deep | Reduces overestimation | Slightly more compute (two forward passes) | Stable discrete action problems |
| Actor-Critic (Vanilla) | Policy Gradient + Value Baseline | Lower variance than pure REINFORCE | High variance advantage estimates | Educational baselines |
| A2C | Synchronous Advantage Actor-Critic | Multi-step returns improve bias/variance tradeoff | Less sample efficient than PPO | Fast prototyping on small tasks |
| PPO | Trust-Region Inspired | Stable updates, robust to hyperparameters | More computation (multiple epochs) | General-purpose baseline |
| DDPG | Deterministic Continuous Control | Handles continuous actions | Sensitive to noise & hyperparameters | Simpler continuous tasks |
| TD3 | Improved DDPG | Mitigates Q overestimation; more stable | More components to tune | Continuous control with noisy rewards |


---
## 4. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium[box2d] matplotlib numpy imageio
```
(Optional) macOS prerequisites for Box2D: `brew install swig` then reinstall gymnasium extras.

---
## 5. Per-Algorithm Insights
### Q-Learning vs SARSA
Q-Learning’s maximization introduces optimistic estimates aiding faster exploitation; SARSA’s on-policy target incorporates exploratory actions leading to safer convergence in stochastic settings.

### DQN / Double DQN
Double DQN reduces positive bias by decoupling action selection and evaluation—improving stability and preventing premature saturation of Q-values.

### Actor-Critic Family
Vanilla Actor-Critic suffers from high variance; A2C’s synchronous n-step returns lower variance while retaining a tractable bias–variance tradeoff. PPO’s clipped surrogate prevents destructive policy updates (implicit trust region) increasing reproducibility.

### Deterministic Methods (DDPG / TD3)
DDPG is sensitive to noise and overestimation; TD3 addresses these via critic redundancy, target smoothing (regularization of sharp Q surfaces), and delayed policy updates (reducing actor drift).

---
