import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def main():
    problem = "Pendulum-v0"
    env = gym.make(problem)

    num_state = env.observation_space.shape[0]
    print("Size of State Space: %d" % num_state)
    num_action = env.action_space.shape[0]
    print("Size of Action Space: %d" % num_action)
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    print("Upper bound: %d" % upper_bound)
    print("Lower bound: %d" % lower_bound)

    class QUActionNoise:
        def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
            self.theta = theta
            self.mean = mean
            self.std_deviation = std_deviation
            self.dt = dt
            self.x_initial = x_initial
            # self.reset()
            if self.x_initial is not None:
                self.x_pre = self.x_initial
            else:
                self.x_pre = np.zeros_like(self.mean)

        def __call__(self):
            x = (self.x_pre +
                 self.theta * (self.mean - self.x_pre) * self.dt +
                 self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
            self.x_pre = x
            return x

        def reset(self):
            if self.x_initial is not None:
                self.x_pre = self.x_initial
            else:
                self.x_pre = np.zeros_like(self.mean)

    class Buffer:
        def __init__(self, buffer_capacity=10000, batch_size=64):
            self.buffer_capacity = buffer_capacity
            self.batch_size = batch_size
            self.buffer_counter = 0

            self.state_buffer = np.zeros((self.buffer_capacity, num_state))
            self.action_buffer = np.zeros((self.buffer_capacity, num_action))
            self.reward_buffer = np.zeros((buffer_capacity, 1))
            self.next_state_buffer = np.zeros((buffer_capacity, num_state))

        def record(self, obs_tuple):
            index = self.buffer_counter % self.buffer_capacity

            self.state_buffer[index] = obs_tuple[0]
            self.action_buffer[index] = obs_tuple[1]
            self.reward_buffer[index] = obs_tuple[2]
            self.next_state_buffer[index] = obs_tuple[3]

            self.buffer_counter += 1

        @tf.function
        def update(self, state_batch, action_batch, reward_batch, next_state_batch):
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_state_batch, training=True)
                y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)

if __name__ == '__main__':
    main()
