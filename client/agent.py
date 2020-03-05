from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import copy

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx('float64')


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            tf.constant(self.state[ind]),
            tf.constant(self.action[ind]),
            tf.constant(self.next_state[ind]),
            tf.constant(self.reward[ind]),
            tf.constant(self.not_done[ind]),
        )


class Actor(Model):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = Dense(256, input_dim=state_dim, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(action_dim, activation='tanh')

        self.max_action = max_action

    def call(self, inputs, **kwargs):
        state = inputs
        a = self.l1(state)
        a = self.l2(a)
        a = self.l3(a)
        return self.max_action * a


class Critic(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = Dense(256, input_dim=(state_dim + action_dim), activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(1)

        # Q2 architecture
        self.l4 = Dense(256, input_dim=(state_dim + action_dim), activation='relu')
        self.l5 = Dense(256, activation='relu')
        self.l6 = Dense(1)

    def call(self, inputs, **kwargs):
        state, action = inputs
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(sa)
        q2 = self.l5(q2)
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = tf.concat([state, action], 1)
        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        return q1


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,

            # how far into the future rewards should be accounted for
            # 0 = only care about immediate rewards
            # 1 = get highest cumulative reward for entire run
            discount=0.99,

            # rate at which frozen target models are updated
            tau=0.005,

            # noise added to actions for effective exploration
            # may be helped with HER + curiosity later...
            policy_noise=0.2,

            # limit at which noise may take on the action
            noise_clip=0.5,

            # frequency at which the policy should be updated
            # 2 = update policy every 2 critic updates
            policy_freq=2,
    ):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)

        # # todo: determine whether to copy weights
        # self.actor_target.set_weights(self.actor.weights)

        self.actor_optimiser = Adam(3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # # todo: determine whether to copy weights
        # self.critic_target.set_weights(self.critic.weights)
        self.critic_optimiser = Adam(3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.loss_object = tf.keras.losses.MeanSquaredError()

    def select_action(self, state):
        state = tf.reshape(state, (1, -1))
        return tf.reshape(self.actor(state), [-1])

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = tf.random.normal(action.shape, dtype=tf.dtypes.float64) * self.policy_noise
        noise = tf.clip_by_value(noise, -self.max_action, self.max_action)

        next_action = self.actor_target(next_state) + noise
        next_action = tf.clip_by_value(next_action, -self.max_action, self.max_action)

        # Compute the target Q value
        # double Q learning - take the lowest of 2 Q function approximations
        target_Q1, target_Q2 = self.critic_target((next_state, next_action))
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        with tf.GradientTape() as tape:
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic((state, action))

            # Compute critic loss
            loss = self.loss_object(target_Q, current_Q1) + self.loss_object(target_Q, current_Q2)

        # Optimise the critic
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimiser.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            with tf.GradientTape() as tape:
                # original took mean to reduce size... the nerds...
                # its actually only expected to be one value here, but if needed
                # for tensorflow i'll take the mean too :-)
                actor_loss = -self.critic.Q1(state, self.actor(state))

            # Optimise the actor
            gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimiser.apply_gradients(zip(gradients, self.actor.trainable_variables))

            # Update the frozen target models
            self.critic_target.set_weights([
                self.tau * param + (1 - self.tau) * target_param
                for param, target_param in zip(self.critic.weights, self.critic_target.weights)
            ])

            self.actor_target.set_weights([
                self.tau * param + (1 - self.tau) * target_param
                for param, target_param in zip(self.actor.weights, self.actor_target.weights)
            ])
