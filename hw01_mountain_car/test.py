from hw01_mountain_car.agent import Agent
from gym import make
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = make("MountainCar-v0")
    agent = Agent()

    pp = np.linspace(-1.2, 0.6, 101)
    vv = np.linspace(-0.07, 0.07, 101)
    a = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            a[i, j] = agent.act((np.array([(pp[i] + pp[i + 1]) / 2, (vv[j] + vv[j + 1]) / 2])))
    fig, ax = plt.subplots()
    c = ax.pcolormesh(vv, pp, a)
    ax.set_ylabel('position')
    ax.set_xlabel('velocity')
    fig.colorbar(c, ax=ax, ticks=[0, 1, 2])
    fig.show()

    episodes = 20
    total_reward = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            next_state, reward, done, _ = env.step(agent.act(state))
            # env.render()
            total_reward += reward
            state = next_state
    average_reward = total_reward / episodes
    env.close()
    print(average_reward)
