import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
from collections import deque
import os, subprocess


try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Please install with: brew install ffmpeg")


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
        self.s, self.a, self.r, self.ns, self.d = deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen)

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
    plt.title(f'{name} Double DQN on CartPole-v1', fontsize=16)
    plt.xlabel('Episode')
    plt.ylabel(name)
    plt.plot(values, color='gray', alpha=0.5, label='Raw')
    plt.plot(smooth, color='lime', linewidth=2, label='Smoothed')
    plt.legend()
    plt.show()


def record_video(policy, filename="cartpole_double_dqn.mp4", steps=500):
    print("🎥 Recording video...")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    s, _ = env.reset()
    frames = []

    for t in range(steps):
        with torch.no_grad():
            q_vals = policy(torch.FloatTensor(s))
            action = torch.argmax(q_vals).item()

        frame = env.render()
        frames.append(frame)

        ns, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        s = ns

        if done:
            s, _ = env.reset()

    env.close()
    imageio.mimsave(filename, frames, fps=30, codec='libx264')
    print(f"✅ Video saved as {filename}")


def select_action(q_net, s, eps, action_dim):
    if np.random.rand() < eps:
        return np.random.randint(0, action_dim)
    else:
        q_vals = q_net(torch.FloatTensor(s))
        return torch.argmax(q_vals).item()


def ddqn_update(q_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    s, a, r, ns, d = replay_buffer.sample(batch_size)
    s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

    # Current Q
    curr_Q = q_net(s).gather(1, a)

    # Double DQN: use q_net for argmax, target_net for Q value
    with torch.no_grad():
        next_actions = torch.argmax(q_net(ns), dim=1, keepdim=True)
        next_Q = target_net(ns).gather(1, next_actions)
        target_Q = r + gamma * next_Q * (1 - d)

    # Loss
    loss = F.mse_loss(curr_Q, target_Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = 'cpu'
    gamma = 0.99
    lr = 1e-3
    tau = 0.005
    batch_size = 64

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    episodes = 100

    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    eps = eps_start

    rewards_hist = []
    loss_hist = []

    for ep in range(episodes):
        s, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            a = select_action(q_net, s, eps, action_dim)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            buffer.add(s, a, r, ns, done)
            s = ns
            total_r += r

            loss = ddqn_update(q_net, target_net, optimizer, buffer, batch_size, gamma, device)
            if loss is not None:
                loss_hist.append(loss)

            # Soft update target net
            for tp, p in zip(target_net.parameters(), q_net.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        eps = max(eps * eps_decay, eps_end)
        rewards_hist.append(total_r)
        print(f"Ep {ep+1}/{episodes} | Reward: {total_r:.1f} | Eps: {eps:.3f}")

    env.close()

    draw(rewards_hist, "Rewards")
    draw(loss_hist, "Q Loss")
    record_video(q_net, "cartpole_double_dqn.mp4", steps=500)


if __name__ == "__main__":
    main()
