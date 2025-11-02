import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import imageio
import os, subprocess

try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Please install with: brew install ffmpeg")

class ReplayBuffer:
    def __init__(self, maxlen=1000000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r).reshape(-1, 1),
                np.array(s_), np.array(d).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        # Critic 2
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

    def forward(self, x, a):
        xu = torch.cat([x, a], 1)

        # Q1
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)

        # Q2
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)
        return q1, q2

    def Q1(self, x, a):
        xu = torch.cat([x, a], 1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        return q1


class TD3:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return None, None

        self.total_it += 1

        # Sample batch
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(s)
        action = torch.FloatTensor(a)
        reward = torch.FloatTensor(r)
        next_state = torch.FloatTensor(s_)
        done = torch.FloatTensor(d)

        # Select next action with noise
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Target Q
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # Current Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed Policy Update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update Targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item() if actor_loss else None, critic_loss.item()


def plot_metrics(rewards, actor_loss, critic_loss):
    plt.style.use("dark_background")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(rewards, color='lime', label='Reward')
    axs[0].set_title("Episode Rewards")
    axs[0].legend()

    axs[1].plot(actor_loss, color='orange', label='Actor Loss')
    axs[1].set_title("Actor Loss (Delayed Updates)")
    axs[1].legend()

    axs[2].plot(critic_loss, color='red', label='Critic Loss')
    axs[2].set_title("Critic Loss")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    # Try BipedalWalker, fallback to Pendulum if Box2D not installed
    try:
        env = gym.make("BipedalWalker-v3")
        print("✓ Using BipedalWalker-v3")
        episodes = 400
        max_steps = 1500
    except:
        print("⚠️ BipedalWalker not available. Install with: brew install swig && pip install 'gymnasium[box2d]'")
        print("✓ Falling back to Pendulum-v1")
        env = gym.make("Pendulum-v1")
        episodes = 200
        max_steps = 200
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    buffer = ReplayBuffer()

    batch_size = 256

    rewards, actor_losses, critic_losses = [], [], []
    exploration_noise = 0.1

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(np.array(state))
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            actor_loss, critic_loss = agent.train(buffer, batch_size)
            if actor_loss:
                actor_losses.append(actor_loss)
            if critic_loss:
                critic_losses.append(critic_loss)

            if done:
                break

        rewards.append(episode_reward)
        print(f"Ep {ep+1}/{episodes} | Reward: {episode_reward:.1f}")

    env.close()
    plot_metrics(rewards, actor_losses, critic_losses)


if __name__ == "__main__":
    main()
