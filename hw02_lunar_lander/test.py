from hw02_lunar_lander.agent import Agent
from gym import make

if __name__ == '__main__':
    env = make('LunarLander-v2')
    # env = make("MountainCar-v0")
    agent = Agent()
    episodes = 50
    total_reward = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            # print((reward, reward + 100 * (- 0.99 * abs(next_state[5]) + abs(state[5]))))
            # print(reward)
            total_reward += reward
            state = next_state
    average_reward = total_reward / episodes
    env.close()
    print(average_reward)
