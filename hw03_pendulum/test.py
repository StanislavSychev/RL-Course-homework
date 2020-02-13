from hw03_pendulum.agent import Agent
from gym import make
# from .train import ActorA2C

if __name__ == '__main__':
    env = make('Pendulum-v0')
    agent = Agent()
    episodes = 75
    total_reward = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward
            state = next_state
    average_reward = total_reward / episodes
    env.close()
    print(average_reward)
