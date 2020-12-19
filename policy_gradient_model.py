from action import Action
from constants import (NUMBER_OF_CIRCLES, NUMBER_OF_COLORS, NUMBER_OF_ROWS,
                       PER_GAME)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
    
class PolicyGradientModel:
    def __init__(self, random_or_override,hyper_parameters,name="Bilbo",human=False):
        self.gamma = 0.99  # Discount factor for past rewards
        self._create_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.episode_count = 0
        self.train_count = 0
        self.random_or_override = random_or_override
        self.save_interval = hyper_parameters.save_interval
        self.name = name
        if human:  # Add check for weights file exisiting
            self.model.load_weights("PG_complete_weights_{0}.h5".format(self.name))
    
    def _create_model(self):
        num_inputs = 185
        self.num_actions = 180
        num_hidden = 128

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(self.num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

    def simulated_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state = np.array([state])
        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = self.model(state)
        self.critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        pruned_actions = [0] * self.num_actions
        action_probs_list = np.squeeze(action_probs)
        for possible_action in possible_actions:
            encoded_action = self.encode_action(possible_action)
            pruned_actions[encoded_action] = action_probs_list[encoded_action]
        pruned_actions = np.array(pruned_actions) / np.sum(np.array(pruned_actions))
        action = self.random_or_override.weighted_random_choice(self.num_actions, np.squeeze(pruned_actions))
        pruned_action_tensor = tf.convert_to_tensor([pruned_actions],dtype=tf.float32)
        self.action_probs_history.append(tf.math.log(pruned_action_tensor[0, action]))
        return self._convert_action_num(action)


    def greedy_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state = np.array([state])
        action_probs, critic_value = self.model(state)
        self.critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        pruned_actions = [0] * self.num_actions
        action_probs_list = np.squeeze(action_probs)
        for possible_action in possible_actions:
            encoded_action = self.encode_action(possible_action)
            pruned_actions[encoded_action] = action_probs_list[encoded_action]
        return self._convert_action_num(np.argmax(pruned_actions))


    def train(self, reward, tape):
        self.train_count += 1
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = self._calculate_returns(reward, len(self.action_probs_history))

        # Calculating loss values to update our network
        history = zip(self.action_probs_history, self.critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # print(loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients((grad, var)
            for (grad, var) in zip(grads, self.model.trainable_variables) if grad is not None)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear the loss and reward history
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        if self.train_count % self.save_interval == 0:
            self.model.save_weights("PG_weights_{0}.h5".format(self.name))    
    def _calculate_returns(self, reward, size):
        returns = []
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        rewards_history = [0] * (size - 1) + [reward]

        # Smallest number such that 1.0 + eps != 1.0
        eps = np.finfo(np.float32).eps.item()
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        return returns

    def encode_action(self, action):
        action_num = action.circle * 30
        action_num += (action.color-1) * 6
        action_num += action.row
        return action_num

    def _convert_action_num(self, action_number):
        circle = action_number // (NUMBER_OF_COLORS * NUMBER_OF_ROWS)
        action_number = action_number % (NUMBER_OF_COLORS * NUMBER_OF_ROWS)
        color = action_number // (NUMBER_OF_ROWS) + 1
        row = action_number % NUMBER_OF_ROWS
        return Action(circle, color, row)
        

    
