# a script to evaluate the model because i don't trust my own brain
from spider_env_new import SpiderEnv
from ppo import Agent
import numpy as np

def evaluate_model(model_path, num_episodes=100):
    env = SpiderEnv(render_mode=None)  # No rendering for speed
    agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0])
    agent.load_models()
    
    rewards = []
    lengths = []
    outcomes = {'goal': 0, 'collision': 0, 'timeout': 0}
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(2000):
            action, _, _ = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                outcome = info.get('end_reason', 'unknown')
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                break
        
        print(f"reward: {episode_reward} | length: {episode_length}")
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Length:    {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print(f"Success Rate:      {outcomes.get('goal', 0)/num_episodes*100:.1f}%")
    print(f"Collision Rate:    {outcomes.get('collision', 0)/num_episodes*100:.1f}%")
    print(f"Timeout Rate:      {outcomes.get('timeout', 0)/num_episodes*100:.1f}%")
    print(f"{'='*60}\n")
    
    env.close()

if __name__ == "__main__":
    evaluate_model("./models/spider_ppo.pth", num_episodes=100)