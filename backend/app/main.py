from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import io
import numpy as np
from PIL import Image
import gymnasium as gym
from .model import model, Policy

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

env = gym.make("CartPole-v1", render_mode="rgb_array")

@app.get("/")
async def root():
    return {"msg": "CartPole PPO API ready!"}

@app.post("/act")
async def act(state: list = Body()):
    state_t = torch.FloatTensor([state])
    logits = model(state_t)
    probs = torch.softmax(logits, dim=1)
    action = torch.argmax(probs, dim=1).item()
    return {"action": action, "probs": probs[0].tolist()}

@app.post("/simulate")
async def simulate(steps: int = 200):
    obs, info = env.reset()
    frames = []
    trajectory = []
    
    for _ in range(steps):
        frame = env.render()
        frames.append(Image.fromarray(frame))
        
        state_t = torch.FloatTensor([obs])
        logits = model(state_t)
        probs = torch.softmax(logits, dim=1)
        action = torch.argmax(probs, dim=1).item()
        
        obs, rew, done, trunc, info = env.step(action)
        trajectory.append({"obs": obs.tolist(), "action": action, "rew": float(rew)})
        
        if done or trunc:
            break
    
    # First frame PNG bytes
    img_bytes = io.BytesIO()
    frames[0].save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return {
        "trajectory": trajectory,
        "final_reward": sum(t["rew"] for t in trajectory),
        "frame": img_bytes.getvalue()
    }
