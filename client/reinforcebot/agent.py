import argparse

import gym
import numpy as np


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, done):
        return self.action_space.sample()


class ReplayBuffer:
    def __init__(self, observation_space, action_space, max_size=1024):
        self.d = np.empty((max_size,))
        self.s = np.empty((max_size, *observation_space.shape))
        self.a = np.empty((max_size, *action_space.shape))
        self.r = np.empty((max_size,))
        self.sp = np.empty((max_size, *observation_space.shape))
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def write(self, done, s, a, r, sp):
        self.d[self.index] = done
        self.s[self.index] = s
        self.a[self.index] = a
        self.r[self.index] = r
        self.sp[self.index] = sp
        self.index = (self.index + 1) % self.size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size=64):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.d[indices],
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.sp[indices],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', nargs='?', default='CartPole-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.seed(0)
    agent = Agent(env.action_space)
    episode_count = 100
    done = False
    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, done)
            ob, reward, done, _ = env.step(action)
            print(reward)
            if done:
                # train
                # s,a,r,s+t
                # Q(s,a)=Q(s,a)-a*(r+max_a(Q(st+1,a))-Q(s,a))
                break
    env.close()


if __name__ == '__main__':
    main()
