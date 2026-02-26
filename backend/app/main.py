from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import torch
import gymnasium as gym
import time

from .model import model, Policy

app = FastAPI(title="CartPole RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = gym.make("CartPole-v1", render_mode="rgb_array")

# --- simple metrics ---
start_time = time.time()
request_count = 0


def _inc():
    global request_count
    request_count += 1


@app.get("/")
async def root():
    _inc()
    return {"msg": "CartPole PPO API ready!"}


@app.post("/act")
async def act(state: list = Body(...)):
    """
    Given a CartPole state [x, x_dot, theta, theta_dot],
    return greedy action and action probabilities.
    """
    _inc()
    state_t = torch.FloatTensor([state])
    logits = model(state_t)
    probs = torch.softmax(logits, dim=1)
    action = torch.argmax(probs, dim=1).item()
    return {"action": int(action), "probs": probs[0].tolist()}


@app.post("/simulate")
async def simulate(steps: int = 500):
    """
    Run one episode with the current policy, return rewards per step
    and episode summary.
    """
    _inc()
    obs, info = env.reset()
    rewards = []
    states = []

    for _ in range(steps):
        state_t = torch.FloatTensor([obs])
        logits = model(state_t)
        probs = torch.softmax(logits, dim=1)
        action = torch.argmax(probs, dim=1).item()

        obs, rew, done, trunc, info = env.step(action)
        rewards.append(float(rew))
        states.append(obs.tolist())

        if done or trunc:
            break

    total_reward = float(sum(rewards))
    return {
        "total_reward": total_reward,
        "rewards": rewards,
        "steps": len(rewards),
        "last_state": states[-1] if states else None,
    }


@app.get("/health")
async def health():
    uptime = time.time() - start_time
    return {
        "status": "ok",
        "uptime_seconds": int(uptime),
        "requests": request_count,
    }


@app.get("/metrics")
async def metrics():
    uptime = int(time.time() - start_time)
    body = (
        f"cartpole_requests_total {request_count}\n"
        f"cartpole_uptime_seconds {uptime}\n"
    )
    return body
