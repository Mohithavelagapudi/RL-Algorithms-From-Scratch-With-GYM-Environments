import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os
import subprocess

try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Please install with: brew install ffmpeg")

class DiscretizedPendulum(gym.Env):
    def __init__(self, bins=11, render_mode=None):
        super().__init__()
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(bins)
        self.bins = bins
        self.low = self.env.action_space.low[0]
        self.high = self.env.action_space.high[0]
        self.action_values = np.linspace(self.low, self.high, bins)
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        return reset_out

    def step(self, action):
        action_cont = np.array([self.action_values[action]])
        step_out = self.env.step(action_cont)
        return step_out

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.s = deque(maxlen=maxlen)
        self.a = deque(maxlen=maxlen)
        self.r = deque(maxlen=maxlen)
        self.ns = deque(maxlen=maxlen)
        self.d = deque(maxlen=maxlen)

    def add(self, s, a, r, ns, d):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.ns.append(ns)
        self.d.append(d)

    def sample(self, batch_size=64):
        idxs = np.random.choice(len(self.a), size=min(batch_size, len(self.a)), replace=False)
        s = torch.FloatTensor([self.s[i] for i in idxs])
        a = torch.LongTensor([self.a[i] for i in idxs]).unsqueeze(1)
        r = torch.FloatTensor([self.r[i] for i in idxs]).unsqueeze(1)
        ns = torch.FloatTensor([self.ns[i] for i in idxs])
        d = torch.FloatTensor([self.d[i] for i in idxs]).unsqueeze(1)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.a)


def draw(values, name):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    smooth = np.convolve(values, np.ones(5) / 5, mode='valid')
    plt.title(f'{name} DQN on Pendulum-v1', fontsize=16)
    plt.xlabel('Episode')
    plt.ylabel(name)
    plt.plot(values, color='gray', alpha=0.5, label='Raw')
    plt.plot(smooth, color='lime', linewidth=2, label='Smoothed')
    plt.legend()
    plt.show()


def record_video(policy, filename="pendulum_dqn.mp4", steps=300):
    print("🎥 Recording video...")
    env = DiscretizedPendulum(render_mode='rgb_array')
    reset_out = env.reset()
    s = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    frames = []

    for t in range(steps):
        with torch.no_grad():
            q_vals = policy(torch.FloatTensor(s))
            action = torch.argmax(q_vals).item()

        frame = env.render()
        frames.append(frame)

        step_out = env.step(action)
        if len(step_out) == 5:
            s, r, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            s, r, done, _ = step_out

        if done:
            reset_out = env.reset()
            s = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    env.close()
    imageio.mimsave(filename, frames, fps=30, codec='libx264')
    print(f"✅ Saved video as {filename}")


def select_action(q_net, s, eps, action_dim):
    if np.random.rand() < eps:
        return np.random.randint(0, action_dim)
    else:
        q_vals = q_net(torch.FloatTensor(s))
        return torch.argmax(q_vals).item()

def main():
    env = DiscretizedPendulum(bins=11, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = 'cpu'
    gamma = 0.99
    lr = 1e-3
    tau = 0.005
    buffer = ReplayBuffer()
    q_net = QNetwork(state_dim, action_dim).to(device)
    target_q_net = QNetwork(state_dim, action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    episodes = 50
    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    rewards_hist = []
    q_loss_hist = []

    eps = eps_start
    for ep in range(episodes):
        reset_out = env.reset()
        s = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_r = 0
        for t in range(300):
            a = select_action(q_net, s, eps, action_dim)
            step_out = env.step(a)
            if len(step_out) == 5:
                ns, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                ns, r, done, _ = step_out

            buffer.add(s, a, r, ns, done)
            s = ns
            total_r += r

            if len(buffer) > 128:
                bs, ba, br, bns, bd = buffer.sample(64)
                q_vals = q_net(bs).gather(1, ba)
                with torch.no_grad():
                    target_q = target_q_net(bns).max(1)[0].unsqueeze(1)
                    y = br + gamma * target_q * (1 - bd)
                loss = F.mse_loss(q_vals, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                q_loss_hist.append(loss.item())

                # Soft update
                for tp, p in zip(target_q_net.parameters(), q_net.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            if done:
                break

        eps = max(eps * eps_decay, eps_end)
        rewards_hist.append(total_r)
        print(f"Ep {ep+1}/{episodes} | Reward: {total_r:.2f} | Eps: {eps:.3f}")

    draw(rewards_hist, "Rewards")
    draw(q_loss_hist, "Q Loss")
    record_video(q_net, "pendulum_dqn.mp4", steps=300)
    env.close()


if __name__ == "__main__":
    main()
