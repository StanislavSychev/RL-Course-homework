from gym import make
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from collections import deque

N_STEP = 200
GAMMA = 0.9
BETA = 1e-1


def transform_state(state):
    return np.array(state)


class ActorA2C(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorA2C, self).__init__()
        self.activation = nn.ReLU()
        self.input = nn.Linear(state_dim, hidden_dim)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mu_output = nn.Linear(hidden_dim, action_dim)
        self.var_output = nn.Linear(hidden_dim, action_dim)
        self.var_activation = nn.Softplus()

    def forward(self, state):
        base = self.activation(self.hidden(self.activation(self.input(state))))
        mu = self.mu_output(base)
        var = self.var_activation(self.var_output(base))
        return mu, var


class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.gamma = GAMMA ** N_STEP
        self.actor = ActorA2C(state_dim, action_dim, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=5e-3)
        self.actor_optim = Adam(self.actor.parameters(), lr=5e-4)
        self.critic_loss = nn.MSELoss()
        self.loss = torch.tensor(0.0)

    def update(self, transition):
        state, action, next_state, reward, done = transition
        reward = torch.tensor(reward).view(1).float()
        if not done:
            with torch.no_grad():
                reward += self.gamma * self.critic(torch.tensor(next_state).float())
        state = torch.tensor(state).float()
        mu, var = self.actor(state)
        val = self.critic(state)
        critic_loss = self.critic_loss(val, reward)
        advantage = reward - val.detach()
        action = torch.tensor(action).float()
        p1 = - ((mu - action) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * np.pi * var))
        log_pi = p1 + p2
        actor_loss = - (advantage * log_pi).mean()
        entropy_loss = BETA * (-(torch.log(2 * np.pi * var) + 1) / 2).mean()
        self.loss += critic_loss + actor_loss + entropy_loss

    def step(self):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.loss.backward()
        self.critic_optim.step()
        self.actor_optim.step()
        self.loss = torch.tensor(0.0)

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        with torch.no_grad():
            state = torch.tensor(state).float()
            mu, var = self.actor(state)
            mu = mu.data.numpy()
            sigma = torch.sqrt(var).data.numpy()
            return np.random.normal(mu, sigma)

    def act_agent(self, state):
        with torch.no_grad():
            state = transform_state(state)
            mu, _ = self.actor(torch.tensor(state).float())
            return mu.detach().numpy()

    def save(self):

        torch.save(self.actor, "agent.pkl")


def test(agent, env):
    episodes = 75
    total_reward = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act_agent(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    average_reward = total_reward / episodes
    env.close()
    return average_reward


if __name__ == "__main__":
    env = make("Pendulum-v0")
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    env.seed(42)
    algo = A2C(state_dim=3, action_dim=1)
    episodes = 10000
    best_score = -2000

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            action = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                rb = list(reward_buffer)
                for k in range(1, N_STEP):
                    algo.update((state_buffer[k], action_buffer[k], next_state,
                                 sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))
                action_buffer.clear()
                reward_buffer.clear()
                state_buffer.clear()
                algo.step()
            state = next_state
        print((i, total_reward))

        if (i % 20) == 0 and i != 0:
            score = test(algo, env)
            if score >= best_score:
                best_score = score
                print("new best score")
                print((i, score))
                algo.save()
