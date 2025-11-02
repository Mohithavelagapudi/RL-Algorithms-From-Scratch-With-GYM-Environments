import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import imageio
import os, subprocess

try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Please install with: brew install ffmpeg")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)



class PPOAgent:
    def __init__(self, state_dim, action_dim, clip=0.2, gamma=0.99, lr=3e-4, k_epochs=4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optim = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip = clip
        self.gamma = gamma
        self.k_epochs = k_epochs

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, log_probs_old, returns):
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        for _ in range(self.k_epochs):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


# ------------------------------------
# Train PPO
# ------------------------------------
def train_ppo(env_name="CartPole-v1", episodes=2000, rollout_len=2000, update_timestep=4000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    rewards_log = []
    avg_rewards = deque(maxlen=100)
    timestep = 0

    state, _ = env.reset()

    memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}

    for ep in range(episodes):
        ep_reward = 0

        for t in range(rollout_len):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = agent.actor.act(state_t)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in memory
            memory["states"].append(state)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(reward)
            memory["dones"].append(done)

            state = next_state
            ep_reward += reward
            timestep += 1

            if done:
                state, _ = env.reset()

            # Update PPO
            if timestep % update_timestep == 0:
                with torch.no_grad():
                    next_val = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()

                returns = agent.compute_returns(memory["rewards"], memory["dones"],
                                                agent.critic(torch.FloatTensor(memory["states"])).detach().squeeze(),
                                                next_val)

                agent.update(
                    torch.FloatTensor(memory["states"]),
                    torch.tensor(memory["actions"]),
                    torch.stack(memory["log_probs"]),
                    returns
                )

                memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}

        rewards_log.append(ep_reward)
        avg_rewards.append(ep_reward)
        print(f"Ep {ep+1}/{episodes} | Reward: {ep_reward:.1f} | Avg(100): {np.mean(avg_rewards):.1f}")

        if np.mean(avg_rewards) >= 475:
            print("🏆 Solved the environment!")
            break

    env.close()

    # Plot reward curve
    plt.figure(figsize=(10,6))
    plt.plot(rewards_log, color='cyan', label='Episode Reward')
    plt.plot(np.convolve(rewards_log, np.ones(10)/10, mode='valid'), color='orange', linewidth=2, label='Smoothed')
    plt.title('PPO Training on CartPole-v1', fontsize=15)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    return agent


def record_video(agent, env_name="CartPole-v1", filename="ppo_cartpole.mp4"):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    total_reward = 0

    for _ in range(500):
        frame = env.render()
        frames.append(frame)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = agent.actor(state_t)
        action = torch.argmax(probs, dim=-1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        if done:
            break

    env.close()
    imageio.mimsave(filename, frames, fps=30, codec='libx264')
    print(f"🎥 Video saved: {filename} | Total Reward: {total_reward}")


if __name__ == "__main__":
    agent = train_ppo()
    record_video(agent)
