import random
import numpy as np
import os
import torch
from .train import transform_state


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.eval()
        
    def act(self, state):
        state = transform_state(state)
        return self.model(torch.tensor(state).float()).argmax().item()

    def reset(self):
        pass

