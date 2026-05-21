"""
Training loop for SpdrBot using the PPO agent.
"""

import os
import numpy as np
from spider_env_new import SpiderEnv
from ppo import PPO

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 1_000_000   # how many env steps to train for in total
UPDATE_INTERVAL = 2048       # Update PPO every N steps (NOT episodes!)
SAVE_INTERVAL = 10           # save model weights every N episodes
LOG_FILE      = "reward_log.txt"
MODEL_PATH    = "./models/spider_ppo.pth"

RESUME_TRAINING = True 
train = True

# ── Setup ─────────────────────────────────────────────────────────────────────
env   = SpiderEnv(render_mode="human")
obs, _= env.reset()

agent = PPO()

# Load existing model if resuming
if RESUME_TRAINING and os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    print("Model loaded - resuming training")
elif not train and os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)

# ── Training loop ─────────────────────────────────────────────────────────────
episode = 0
episode_reward = 0.0
episode_length = 0
reward_history = []
length_history = []
collision_history = []
steps_since_update = 0  # Track steps for PPO updates

for step in range(TOTAL_STEPS):
    action, log_prob, value = agent.choose_action(obs)
    current_obs = obs
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_length += 1
    steps_since_update += 1
    done = terminated or truncated

    agent.save_memory(current_obs, action, log_prob, value, reward, done)

    # UPDATE EVERY N STEPS (not every episode)
    if steps_since_update >= UPDATE_INTERVAL:
        print(f"\n{'='*60}")
        print(f"UPDATING PPO at step {step} ({steps_since_update} steps collected)")
        print(f"{'='*60}")
        agent.update()
        steps_since_update = 0

    if done:
        episode += 1
        end_reason = info.get("end_reason", "?")
        per_step = episode_reward / episode_length

        reward_history.append(episode_reward)
        length_history.append(episode_length)
        collision_history.append(1 if end_reason == "collision" else 0)

        mean_reward = np.mean(reward_history[-100:])
        mean_length = np.mean(length_history[-100:])
        collision_rate = np.mean(collision_history[-100:]) * 100.0

        with open(LOG_FILE, "a") as f:
            f.write(
                f"ep={episode}"
                f"  total={episode_reward:.2f}"
                f"  per_step={per_step:.3f}"
                f"  steps={episode_length}"
                f"  end={end_reason}"
                f"  mean100={mean_reward:.2f}"
                f"  mean_len={mean_length:.0f}"
                f"  coll%={collision_rate:.1f}"
                f"\n"
            )

        if episode % SAVE_INTERVAL == 0:
            print(
                f"ep {episode:>5} | step {step:>7} | "
                f"total {episode_reward:>8.2f} | steps {episode_length:>5} | "
                f"end={end_reason:<9} | mean100 {mean_reward:>8.2f} | "
                f"coll% {collision_rate:>5.1f}"
            )
            agent.save_model(MODEL_PATH)

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

env.close()