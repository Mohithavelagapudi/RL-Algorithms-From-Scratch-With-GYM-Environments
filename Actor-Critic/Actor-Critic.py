import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import os, subprocess


try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Install with: brew install ffmpeg")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def train_actor_critic(env_name="CartPole-v1", episodes=300, gamma=0.99, lr=3e-4):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    reward_history = []
    actor_loss_list = []
    critic_loss_list = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []

        done = False
        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

        # Compute returns
        Qvals = []
        Qval = 0
        for r in reversed(rewards):
            Qval = r + gamma * Qval
            Qvals.insert(0, Qval)

        values = torch.cat(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        advantage = Qvals - values.squeeze()

        # Loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        total_loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))
        actor_loss_list.append(actor_loss.item())
        critic_loss_list.append(critic_loss.item())

        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1}/{episodes} | Reward: {sum(rewards):.2f}")

    env.close()
    plot_metrics(reward_history, actor_loss_list, critic_loss_list)
    return agent


def plot_metrics(rewards, actor_loss, critic_loss):
    plt.style.use("dark_background")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(rewards, color="lime")
    axs[0].set_title("Episode Rewards")

    axs[1].plot(actor_loss, color="orange")
    axs[1].set_title("Actor Loss")

    axs[2].plot(critic_loss, color="red")
    axs[2].set_title("Critic Loss")

    plt.tight_layout()
    plt.show()


def record_video(agent, env_name="CartPole-v1", filename="cartpole_actor_critic.mp4", max_steps=500):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        action, _, _ = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if done:
            state, _ = env.reset()

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"🎥 Video saved as {filename}")


if __name__ == "__main__":
    trained_agent = train_actor_critic(episodes=200)
    record_video(trained_agent, max_steps=500)
