from action import Action
from constants import (NUMBER_OF_CIRCLES, NUMBER_OF_COLORS, NUMBER_OF_ROWS,
                       PER_GAME)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
    
class PolicyGradientModel:
    def __init__(self, random_or_override,hyper_parameters,name="Bilbo",human=False):
        self.gamma = 0.9  # Discount factor for past rewards
        self.hyper_parameters = hyper_parameters
        self._create_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=hyper_parameters.learning_rate)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.episode_count = 0
        self.train_count = 0
        self.random_or_override = random_or_override
        self.name = name
        self.legal_moves = 0
        self.illegal_moves = 0
        if human:  # Add check for weights file exisiting
            print("Loading Weights...")
            try:
                self.model.load_weights("PG_weights_{0}_complete.h5".format(self.name))
            except:
                print("No Model :(")
    
    def _create_model(self):
        num_inputs = 185
        self.num_actions = 180

        state_inputs = layers.Input(shape=(num_inputs,))
        initializer = tf.keras.initializers.GlorotUniform()

        # "Hidden" layers of the model are configured via hyperparameters
        prev_layer = state_inputs
        for n in self.hyper_parameters.model_size:
            dense_layer = layers.Dense(n, activation="relu", kernel_initializer=initializer)(prev_layer)
            prev_layer = dense_layer

        last_layer_before_mask = layers.Dense(self.num_actions, activation="relu", kernel_initializer=initializer)(prev_layer)
        possible_action_inputs = layers.Input(shape=(self.num_actions,))
        possible_action_masked = layers.Add()([last_layer_before_mask, possible_action_inputs])
        action = layers.Activation(activation="softmax")(possible_action_masked)
        critic = layers.Dense(1)(last_layer_before_mask)

        self.model = keras.Model(inputs=[state_inputs,possible_action_inputs], outputs=[action, critic, last_layer_before_mask])

    def simulated_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state = np.array([state])
        possible_actions_encoded = self.encode_possible_actions(possible_actions)
        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value, action_scores_before_pruning = self.model([state,possible_actions_encoded])
        best_guess = np.argmax(action_scores_before_pruning)
        legal_move = self._convert_action_num(best_guess) in set(possible_actions)
        if (legal_move):
            self.legal_moves += 1
        else:
            self.illegal_moves += 1

        if self.train_count % self.hyper_parameters.save_interval == 0 and self.hyper_parameters.print_model_nn:
            print("before pruning: ", best_guess, " Is it in possible_actions? ", legal_move)
            print("so far, ", self.legal_moves, " legal moves and ", self.illegal_moves, " illegal moves.")

        self.critic_value_history.append(critic_value[0, 0])
        action = self.random_or_override.weighted_random_choice(self.num_actions, np.squeeze(action_probs))
        self.action_probs_history.append(tf.math.log(action_probs[0, action]))

        self.action_probs = action_probs
        self.state = state

        return self._convert_action_num(action)


    def greedy_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state = np.array([state])
        possible_actions_encoded = self.encode_possible_actions(possible_actions)
        action_probs, critic_value, _ = self.model([state,possible_actions_encoded])
        self.critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        return self._convert_action_num(np.argmax(action_probs))

    def encode_possible_actions(self,possible_actions):
        action_mask = [-10000000000000000000000000000000000]*self.num_actions
        for action in possible_actions:
          action_num = self.encode_action(action)
          action_mask[action_num] = 0.0

        return  tf.convert_to_tensor(np.array([action_mask]),dtype=tf.float32)
    


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
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients((grad, var)
            for (grad, var) in zip(grads, self.model.trainable_variables) if grad is not None)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear the loss and reward history
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        if self.train_count % self.hyper_parameters.save_interval == 0:
            self.model.save_weights("PG_weights_{0}.h5".format(self.name))
            if self.hyper_parameters.print_model_nn:
                print("printing layers.")
                for layer in self.model.layers:
                    print("layer: ")
                    print(layer.get_config())
                    print(layer.get_weights()) # list of numpy arrays
                print("action_probs", self.action_probs)
                print("state", self.state)

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
        

    
