# first.py
"""
QRL Simulation for Lattice-Based Post-Quantum Cryptography
Author: R. Palanivel
Description: Quantum Reinforcement Learning-based parameter optimization
for lattice-based PQC with Ramanujan noise modeling.
"""

import numpy as np

# ----------------------------
# Ramanujan Noise Modeling
# ----------------------------
def ramanujan_noise(size, tau=0.5):
    """
    Generates Ramanujan-theoretic noise vector.
    Uses simplified approximation with discrete theta function.
    """
    k = np.arange(-10, 11)  # limited sum approximation
    theta_tau = np.sum(np.exp(-np.pi * k**2 * tau))
    probs = np.exp(-np.pi * k**2 * tau) / theta_tau
    return np.random.choice(k, size=size, p=probs/np.sum(probs))

# ----------------------------
# Lattice Parameter Class
# ----------------------------
class LatticeParams:
    def __init__(self, n=512, q=3329, sigma=3.2, m=512):
        self.n = n        # Dimension
        self.q = q        # Modulus
        self.sigma = sigma # Noise width
        self.m = m        # LWE samples
    
    def update(self, action):
        """
        Updates lattice parameters based on QRL agent action
        """
        if action == "increase_security":
            self.n += 32
        elif action == "reduce_failure":
            self.sigma += 0.2
        elif action == "latency_constraint":
            self.q = max(2, self.q - 32)
        elif action == "memory_constraint":
            self.m = max(64, self.m - 32)
        elif action == "energy_minimize":
            self.sigma = max(1.0, self.sigma - 0.5)

# ----------------------------
# QRL Agent
# ----------------------------
class QRLAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {}  # simple state-action table
    
    def select_action(self, state):
        # epsilon-greedy placeholder
        epsilon = 0.1
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        # deterministic policy placeholder
        return self.actions[0]
    
    def update(self, state, action, reward):
        # placeholder: Q-value update
        key = (state, action)
        self.q_table[key] = reward

# ----------------------------
# Reward Function
# ----------------------------
def compute_reward(params):
    """
    Reward function combining security, runtime, memory, and energy proxy.
    Higher is better.
    """
    security_bits = params.n / 4  # simplified security metric
    runtime = 1.0 / params.q       # simplified runtime metric
    memory = 1.0 / params.m
    energy_proxy = params.sigma * params.m * 0.01
    reward = security_bits + runtime + memory - energy_proxy
    return reward

# ----------------------------
# Simulation Loop
# ----------------------------
def run_qrl_simulation(episodes=20):
    actions = ["increase_security", "reduce_failure", "latency_constraint",
               "memory_constraint", "energy_minimize"]
    
    agent = QRLAgent(actions)
    params = LatticeParams()
    
    for ep in range(episodes):
        state = (params.n, params.q, params.sigma, params.m)
        action = agent.select_action(state)
        params.update(action)
        reward = compute_reward(params)
        agent.update(state, action, reward)
        
        noise_vec = ramanujan_noise(size=params.m)
        
        print(f"Episode {ep+1}: Action={action}, "
              f"Params=n:{params.n}, q:{params.q}, sigma:{params.sigma:.2f}, m:{params.m}, "
              f"Reward={reward:.2f}, NoiseNorm={np.linalg.norm(noise_vec):.2f}")

if __name__ == "__main__":
    run_qrl_simulation()
