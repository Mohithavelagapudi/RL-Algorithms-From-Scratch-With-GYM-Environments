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


def plot_metrics(rewards, td_errors, epsilons, name="SARSA - Taxi-v3"):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    smooth = np.convolve(rewards, np.ones(10) / 10, mode='valid')
    axs[0].plot(rewards, color="gray", alpha=0.4, label="Raw")
    axs[0].plot(smooth, color="lime", linewidth=2, label="Smoothed")
    axs[0].set_title(f"{name} - Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()


    td_smooth = np.convolve(td_errors, np.ones(50) / 50, mode='valid')
    axs[1].plot(td_errors, color="orange", alpha=0.4, label="TD Error (raw)")
    axs[1].plot(td_smooth, color="red", linewidth=2, label="TD Error (smoothed)")
    axs[1].set_title("Temporal Difference Error")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("TD Error Magnitude")
    axs[1].legend()


    axs[2].plot(epsilons, color="cyan", linewidth=2)
    axs[2].set_title("Exploration Rate (Epsilon Decay)")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Epsilon")

    plt.tight_layout()
    plt.show()


def record_video(Q, env_name="Taxi-v3", filename="taxi_sarsa.mp4", steps=300):
    print("🎥 Recording video...")
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []

    for _ in range(steps):
        frame = env.render()
        frames.append(frame)
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if done:
            state, _ = env.reset()

    env.close()
    imageio.mimsave(filename, frames, fps=4, codec="libx264")
    print(f"✅ Video saved as {filename}")

def sarsa_train(env_name="Taxi-v3", episodes=2000, alpha=0.7, gamma=0.95,
                eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    epsilon = eps_start
    epsilons = []
    td_errors = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        # Initial action (ε-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        total_reward = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Choose next action
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            # Compute TD target and error
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            td_errors.append(abs(td_error))

            # Update rule
            Q[state, action] += alpha * td_error

            # Move to next
            state, action = next_state, next_action

        rewards_per_episode.append(total_reward)
        epsilons.append(epsilon)
        epsilon = max(epsilon * eps_decay, eps_end)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_per_episode[-100:])
            print(f"Ep {ep+1}/{episodes} | Avg(100): {avg_r:.2f} | ε={epsilon:.3f}")

    env.close()
    return Q, rewards_per_episode, td_errors, epsilons


if __name__ == "__main__":
    Q, rewards, td_errors, epsilons = sarsa_train()
    plot_metrics(rewards, td_errors, epsilons, "SARSA on Taxi-v3")
    record_video(Q, "Taxi-v3", "taxi_sarsa.mp4", steps=400)
