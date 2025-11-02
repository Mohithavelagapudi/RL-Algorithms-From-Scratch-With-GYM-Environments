import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import matplotlib.pyplot as plt

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
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def train_a2c(env_name="CartPole-v1", episodes=500, gamma=0.99, n_steps=5, lr=3e-4):
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
        states, actions, log_probs, rewards, values = [], [], [], [], []

        done = False
        ep_reward = 0
        t = 0

        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store step
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state
            ep_reward += reward
            t += 1

            # Multi-step update
            if (t % n_steps == 0) or done:
                if done:
                    next_value = 0
                else:
                    _, next_value = agent.forward(torch.FloatTensor(next_state))
                    next_value = next_value.detach().item()

                returns = []
                R = next_value
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)

                returns = torch.FloatTensor(returns)
                values_tensor = torch.cat(values)
                log_probs_tensor = torch.stack(log_probs)
                advantages = returns - values_tensor.squeeze()

                actor_loss = -(log_probs_tensor * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())

                # Clear buffers
                states, actions, log_probs, rewards, values = [], [], [], [], []

        reward_history.append(ep_reward)

        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1}/{episodes} | Reward: {ep_reward:.2f}")

    env.close()
    plot_metrics(reward_history, actor_loss_list, critic_loss_list)
    return agent


def record_video(agent, env_name="CartPole-v1", filename="cartpole_a2c.mp4", max_steps=500):
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


if __name__ == "__main__":
    trained_agent = train_a2c(episodes=300, n_steps=5)
    record_video(trained_agent, max_steps=500)
