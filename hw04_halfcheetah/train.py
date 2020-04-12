from gym import make
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from collections import deque
from copy import deepcopy

GAMMA = 0.9
CLIP = 1e-1
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 1024
K_EPOCHS = 1
LAMBDA = 0.8


def transform_state(state):
    return np.array(state)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorPPO, self).__init__()
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

    def to_save(self):
        return nn.Sequential(
            self.input,
            self.activation,
            self.hidden,
            self.activation,
            self.mu_output,
        )


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.gamma = GAMMA ** TRAJECTORY_SIZE
        self.actor = ActorPPO(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic_loss = nn.MSELoss()
        self.critic_optim = Adam(self.critic.parameters(), lr=5e-3)
        self.actor_optim = Adam(self.actor.parameters(), lr=5e-4)

    def update(self, trajectory):
        state, action, advantage, rollouted_reward = zip(*trajectory)
        state = torch.tensor(state).float()
        action = torch.tensor(action).float()
        advantage = torch.tensor(advantage).view(-1, 1).float()
        rollouted_reward = torch.tensor(rollouted_reward).view(-1, 1).float()
        mu, var = self.actor(state)
        p1 = - ((mu - action) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * np.pi * var))
        old_log_pi = (p1 + p2).detach()
        for _ in range(K_EPOCHS):
            val = self.critic(state)
            mu, var = self.actor(state)
            p1 = - ((mu - action) ** 2) / (2 * var.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * np.pi * var))
            log_pi = p1 + p2
            entropy = ((torch.log(2 * np.pi * var) + 1) / 2).mean()
            ratio = torch.exp(log_pi - old_log_pi.detach())
            s1 = ratio * advantage
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss = - torch.min(s1, s2).mean() + self.critic_loss(rollouted_reward, val) - ENTROPY_COEF * entropy.mean()
            loss.backward()
            self.critic_optim.step()
            self.actor_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float()
            return self.critic(state).item()

    def act(self, state):
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
        torch.save(self.actor.to_save(), "agent.pkl")


def test(agent, env):
    episodes = 50
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
    return average_reward


if __name__ == "__main__":
    env = make("HalfCheetahBulletEnv-v0")
    # env = make("LunarLanderContinuous-v2")
    # env = make("Pendulum-v0")
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    env.seed(42)
    # algo = PPO(state_dim=26, action_dim=6)
    # algo = PPO(state_dim=8, action_dim=2)
    algo = PPO(state_dim=3, action_dim=1)
    episodes = 500
    best_score = -2000

    reward_buffer = deque()
    state_buffer = deque()
    action_buffer = deque()
    done_buffer = deque()
    next_value_buffer = deque()
    value_buffer = deque()
    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            next_value_buffer.append(algo.get_value(next_state))
            value_buffer.append(algo.get_value(state))
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            done_buffer.append(done)
            state = next_state
            if len(action_buffer) == TRAJECTORY_SIZE:
                # rollouted_reward = [algo.get_value(state) if not done else 0]
                # for r, d in zip(reversed(reward_buffer), reversed(done_buffer)):
                #     rollouted_reward.append(r + GAMMA * d * rollouted_reward[-1])
                # rollouted_reward = list(reversed(rollouted_reward))
                v = np.array(value_buffer)
                d = 1 - np.array(done_buffer)
                r = np.array(reward_buffer)
                nv = np.array(next_value_buffer) * GAMMA * d + r
                rollouted_reward = np.copy(nv)
                delta = nv - v
                d[-1] = 0
                c = GAMMA * LAMBDA
                for _ in range(TRAJECTORY_SIZE):
                    nv = np.roll(nv, -1) * GAMMA + r
                    delta += c * (nv - v) * d
                    d *= np.roll(d, -1)
                    c *= GAMMA * LAMBDA
                trajectory = []
                for k in range(0, len(state_buffer)):
                    trajectory.append((state_buffer[k], action_buffer[k], delta[k].item(), rollouted_reward[k].item()))
                algo.update(trajectory)
                action_buffer.clear()
                reward_buffer.clear()
                state_buffer.clear()
                done_buffer.clear()
                next_value_buffer.clear()
                value_buffer.clear()
        print((i, total_reward))

        if i % 50 == 0 and i != 0:
            algo.save()
            # score = test(algo, env)
            # if score >= best_score:
            #     best_score = score
            #     print("new best score ===========================================")
            #     print((i, score))
            #     algo.save()
