from gym import make
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import copy
from collections import deque
import random

N_STEP = 3
GAMMA = 0.9


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    # p = state[0]
    # v = state[1]
    # return np.array([p, v, p ** 2, p * v, v ** 2])
    #
    intervals = np.linspace(0, 1, 30)
    s_p = np.exp(-(state[0] - intervals) ** 2)
    s_v = np.exp(-(state[1] - intervals) ** 2)
    return (s_p * s_v.reshape((-1, 1))).flatten()
    #
    # result = []
    # result.extend(state)
    # return np.array(result)


class AQL:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.q = nn.Linear(state_dim, action_dim)
        self.q.weight.data.fill_(0.0)
        self.q.bias.data.fill_(0.0)
        self.loss = nn.MSELoss()
        self.optim = SGD(self.q.parameters(), lr=1e-4)
        self.gamma = GAMMA ** N_STEP

    def update(self, transition):
        self.optim.zero_grad()
        state, action, next_state, reward, done = transition
        target = torch.tensor(reward)
        if not done:
            with torch.no_grad():
                target += self.gamma * self.q(torch.tensor(next_state).float()).max()
        loss = self.loss(self.q(torch.tensor(state).float())[action], target)
        loss.backward()
        self.optim.step()

    def act(self, state, target=False):
        return self.q(torch.tensor(state).float()).argmax().item()

    def save(self):
        weight = np.array(self.q.weight.detach().numpy())
        bias = np.array(self.q.bias.detach().numpy())
        np.savez("agent.npz", weight, bias)


if __name__ == "__main__":
    env = make("MountainCar-v0")
    eps = 0.1
    episodes = 1000
    np.random.seed(42)
    torch.manual_seed(42)
    env.seed(42)
    random.seed(42)
    aql = AQL(state_dim=30 ** 2, action_dim=3)

    for i in range(episodes):
        state_p = env.reset()
        state = transform_state(state_p)
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state_p, reward, done, _ = env.step(action)
            next_state = transform_state(next_state_p)
            shaped_reward = reward + 20 * (GAMMA * abs(next_state_p[1]) - abs(state_p[1]))
            total_reward += reward
            steps += 1
            reward_buffer.append(shaped_reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        print((i, total_reward))
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))
        if i % 20 == 0:
            aql.save()
    env.close()
