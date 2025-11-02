import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os, subprocess

try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
except:
    print("⚠️ Could not find ffmpeg automatically. Please install with: brew install ffmpeg")

def draw(values, name):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    smooth = np.convolve(values, np.ones(10) / 10, mode='valid')
    plt.title(f'{name} - Q-learning on FrozenLake-v1', fontsize=16)
    plt.xlabel('Episode')
    plt.ylabel(name)
    plt.plot(values, color='gray', alpha=0.5, label='Raw')
    plt.plot(smooth, color='cyan', linewidth=2, label='Smoothed')
    plt.legend()
    plt.show()


def record_video(Q, env_name="FrozenLake-v1", filename="frozenlake_qlearn.mp4", steps=100):
    print("🎥 Recording video...")
    env = gym.make(env_name, render_mode='rgb_array')
    s, _ = env.reset()
    frames = []

    for t in range(steps):
        frame = env.render()
        frames.append(frame)

        a = np.argmax(Q[s, :])
        ns, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s = ns
        if done:
            s, _ = env.reset()

    env.close()
    imageio.mimsave(filename, frames, fps=4, codec='libx264')
    print(f"✅ Video saved as {filename}")


def main():
    # Initialize environment
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))

    # Hyperparameters
    alpha = 0.8         # learning rate
    gamma = 0.95        # discount factor
    eps_start = 1.0     # initial epsilon
    eps_end = 0.05
    eps_decay = 0.995
    eps = eps_start
    episodes = 3000

    # Logs
    rewards_hist = []
    success_rate = []

    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s, :])

            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Update Q-value (Bellman Equation)
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[ns, :]) - Q[s, a])
            s = ns
            total_reward += r

        eps = max(eps * eps_decay, eps_end)
        rewards_hist.append(total_reward)

        # Track success rate every 100 episodes
        if (ep + 1) % 100 == 0:
            success = np.mean(rewards_hist[-100:])
            success_rate.append(success)
            print(f"Ep {ep+1}/{episodes} | Avg Reward: {success:.3f} | Eps: {eps:.3f}")

    env.close()

    # Results
    draw(rewards_hist, "Episode Reward")
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(success_rate)) * 100, success_rate, '-o', color='orange')
    plt.title("Success Rate (mean over last 100 episodes)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Success")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Record policy behavior
    record_video(Q, "FrozenLake-v1", "frozenlake_qlearn.mp4", steps=150)


if __name__ == "__main__":
    main()
