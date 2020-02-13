from gym import make
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

N_STEP = 1
GAMMA = 0.99
TAU = 1e-3


def transform_state(state):
    return np.array(state)


class Memory:
    def __init__(self, size, sample_size):
        self.current = 0
        self.state_replay = []
        self.action_replay = []
        self.next_state_replay = []
        self.reward_replay = []
        self.done_replay = []
        self.size = size
        self.sample_size = sample_size

    def add(self, state, action, next_state, reward, done):
        if len(self.state_replay) < self.size:
            self.state_replay.append(state)
            self.action_replay.append(action)
            self.next_state_replay.append(next_state)
            self.reward_replay.append(reward)
            self.done_replay.append(done)
        else:
            self.state_replay[self.current] = state
            self.action_replay[self.current] = action
            self.next_state_replay[self.current] = next_state
            self.reward_replay[self.current] = reward
            self.done_replay[self.current] = done
            self.current = (self.current + 1) % self.size

    def sample(self):
        ind = np.random.choice(len(self.state_replay), self.sample_size, replace=False)
        state = [torch.tensor(self.state_replay[i]).float().view(-1) for i in ind]
        action = [torch.tensor(self.action_replay[i]).view(-1) for i in ind]
        next_state = [torch.tensor(self.next_state_replay[i]).float().view(-1) for i in ind]
        reward = [torch.tensor(self.reward_replay[i]).float().view(-1) for i in ind]
        done = [torch.tensor(self.done_replay[i]).float().view(-1) for i in ind]
        return torch.cat(state).view(len(ind), -1), \
               torch.cat(action).view(len(ind), -1), \
               torch.cat(next_state).view(len(ind), -1), \
               torch.cat(reward).view(len(ind), -1), \
               torch.cat(done).view(len(ind), -1)

    def __len__(self):
        return len(self.state_replay)

    def ready(self):
        return len(self.state_replay) >= self.sample_size


class DQN:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.tau = TAU
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.target = copy.deepcopy(self.model)
        self.optim = optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss = nn.MSELoss()

    def update_target_(self):
        for curr_layer, target_layer in zip(self.model.parameters(), self.target.parameters()):
            target_layer.data.copy_(self.tau * curr_layer + (1 - self.tau) * target_layer)

    def update(self, state, action, next_state, reward, done):
        target_q = torch.zeros(done.size()).float()
        with torch.no_grad():
            target_q = reward + self.gamma * self.target(next_state).max(1)[0].unsqueeze(1).detach() * (1 - done)
        self.optim.zero_grad()
        loss = self.loss(self.model(state).gather(1, action), target_q)
        loss.backward()
        self.optim.step()
        self.update_target_()

    def act(self, state, target=False):
        with torch.no_grad():
            return self.model(torch.tensor(state).float()).argmax().item()

    def save(self):
        torch.save(self.model, "agent.pkl")


if __name__ == "__main__":
    env = make('LunarLander-v2')
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    env.seed(42)
    dqn = DQN(state_dim=8, action_dim=4)
    eps_max = 1
    eps_min = 0.01
    eps_decay = 0.995
    eps = eps_max
    episodes = 340
    memory = Memory(int(1e5), 64)

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            shaped_reward = reward + 10 * (- GAMMA * abs(next_state[5]) + abs(state[5]))
            total_reward += reward
            # if reward == 100:
            #     print("eagle has landed")
            steps += 1
            memory.add(state, action, next_state, shaped_reward, done)
            state = next_state
            if memory.ready():
                dqn.update(*memory.sample())
            if steps >= 1000:
                break
        eps = max(eps_min, eps * eps_decay)
        print((i, total_reward))

        if i % 20 == 0:
            dqn.save()
