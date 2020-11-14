from simulator_lib_v1_1 import CarTrackSimulator
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

env = CarTrackSimulator(visualize_enable=True)

num_states = env.observation_space
num_actions = env.action_space

upper_bound = env.action_space_high
lower_bound = env.action_space_low
# lower_bound = 0

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


class QUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.x_prev = np.zeros_like(self.mean)

    def __call__(self):
        x = (
            self.x_pre
            + self.theta * (self.mean - self.x_pre) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_pre = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_pre = self.x_initial
        else:
            self.x_pre = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau_ldu):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau_ldu + a * (1 - tau_ldu))


def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state_ldu, noise_object):
    sampled_actions = tf.squeeze(actor_model(state_ldu))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    # sampled_actions = sampled_actions.numpy()
    # legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    legal_action = np.clip(sampled_actions, [-1, -1], [1, 1])
    # legal_action = sampled_actions
    # print(legal_action)
    return [np.squeeze(legal_action)]


std_dev = 0.1
ou_noise = QUActionNoise(mean=np.zeros(2), std_deviation=float(std_dev) * np.ones(2))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

critic_lr = 0.00002
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10000

gamma = 0.9999

tau = 0.005

buffer = Buffer(50000, 5000)

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):

    env.car_1.__init__(visualize_enable=env.visualize_enable)
    prev_state, reward, done = env.step(0, 0)
    episodic_reward = 0
    live_counter = 20
    pre_reward = 0
    while True:
        live_counter -= 1
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        env.action = np.array([action[0][0], action[0][1]])  # For display purpose
        if env.visualize_enable == 1:
            _, _ = env.get_keyboard_input()
        state, reward, done = env.step(action[0][0]*upper_bound[0]+2*upper_bound[0], action[0][1]*upper_bound[1])
        reward += action[0][0]*upper_bound[0]/2+upper_bound[0]
        if live_counter == 0:
            if reward - pre_reward < 1:
                reward -= 15*pre_reward
                done = 1
            else:
                pre_reward = reward
                live_counter = 20
        else:
            pass
        buffer.record((prev_state, [action[0][0], action[0][1]], reward, state))

        episodic_reward += reward

        buffer.learn()

        env.buffer_counter = buffer.buffer_counter  # For display purpose
        env.episode = ep  # For display purpose

        update_target(target_actor.variables, actor_model.variables,  tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        if done == 1:
            break
        prev_state = state
    env.episodic_reward = episodic_reward  # For display purpose
    print("Episodic reward is: %0.5f" % episodic_reward)
    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    print(action)
    avg_reward_list.append(avg_reward)
    print(buffer.buffer_counter)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
