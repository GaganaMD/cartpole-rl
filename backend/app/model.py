import torch
import torch.nn as nn
import os

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppo.pth")

model = Policy()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
