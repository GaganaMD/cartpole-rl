import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import os

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=5e-3)

def run_episode():
    env = gym.make('CartPole-v1')
    states, rewards, log_probs = [], [], []
    obs, _ = env.reset()
    done = False
    while not done:
        obs_t = torch.FloatTensor([obs])
        logits = policy(obs_t)
        probs = torch.softmax(logits, dim=1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        obs, reward, done, trunc, _ = env.step(action.item())
        done = done or trunc
        log_prob = m.log_prob(action)
        states.append(obs)
        rewards.append(reward)
        log_probs.append(log_prob.squeeze())
    return states, rewards, log_probs

recent_rewards = deque(maxlen=100)

for episode in range(800):
    states, rewards, log_probs = run_episode()
    
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + 0.97 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    loss = 0
    for lp, R in zip(log_probs, returns):
        loss += -lp * R
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    episode_reward = sum(rewards)
    recent_rewards.append(episode_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {episode}, avg reward: {avg_reward:.0f}")

os.makedirs("backend/app", exist_ok=True)
torch.save(policy.state_dict(), "backend/app/ppo.pth")
print("✅ Model saved!")
