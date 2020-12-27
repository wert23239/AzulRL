import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from util import human_print

from action import Action
from constants import (NUMBER_OF_CIRCLES, NUMBER_OF_COLORS, NUMBER_OF_ROWS,
                       PER_GAME, WIN_LOSS)
from example import Example
from modified_tensorboard import ModifiedTensorBoard


class PolicyGradientModel:
    def __init__(self, random_or_override,hyper_parameters,settings,name="Bilbo"):
        self.gamma = hyper_parameters.gamma  # Discount factor for past rewards
        self.hyper_parameters = hyper_parameters
        self._create_model()
       # Custom tensorboard object
        file_name = str(hyper_parameters)
        file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit()]).strip()
        self.file_name = "logs/{}-{}-{}".format(name,file_name,int(time.time()))
        print(self.file_name)

        if settings.tb_log:
            self.tensorboard = ModifiedTensorBoard(name,log_dir=log_name)
        self.use_ucb = settings.use_ucb
        self.episode_count = 0
        self.train_count = 0
        self.random_or_override = random_or_override
        self.name = name
        self.legal_moves = 0
        self.illegal_moves = 0
        self._reset_state_and_action_counts()
        self.examples = []
        if settings.load:  # Add check for weights file exisiting
            print("Loading Weights...")
            try:
                self.model.load_weights("PG_complete.h5".format(self.name))
            except:
                print("No Model :(")

    def _reset_state_and_action_counts(self):
        self.state_counts = defaultdict(int)
        self.action_counts = defaultdict(int)
        self.action_totals = defaultdict(int)
    
    def _create_model(self):
        num_inputs = 1240
        self.num_actions = 180

        state_inputs = layers.Input(shape=(num_inputs,))
        initializer = tf.keras.initializers.GlorotUniform()

        # "Hidden" layers of the model are configured via hyperparameters
        prev_layer = state_inputs
        for _ in range(self.hyper_parameters.num_hl):
            dense_layer = layers.Dense(self.hyper_parameters.hl_size, activation="relu", kernel_initializer=initializer)(prev_layer)
            prev_layer = dense_layer

        last_layer_before_mask = layers.Dense(self.num_actions, activation="relu", kernel_initializer=initializer)(prev_layer)
        possible_action_inputs = layers.Input(shape=(self.num_actions,))
        possible_action_masked = layers.Add()([last_layer_before_mask, possible_action_inputs])
        action = layers.Activation(activation="softmax")(possible_action_masked)
        self.model = keras.Model(inputs=[state_inputs,possible_action_inputs], outputs=action)
        self.model.compile(optimizer="Adam", loss="categorical_crossentropy")

    def simulated_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state_string = state.tostring()
        state = np.array([state])
        possible_actions_encoded = self.encode_possible_actions(possible_actions,True)
        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs = self.model.predict([state,possible_actions_encoded])
        self.legal_moves += 1

        action = None
        if self.use_ucb:
            action = self._upper_confidence_bound(state_string, action_probs, set(possible_actions))
            self.state_counts[state_string] += 1
            self.action_counts[(state_string, action)] += 1
        else:
            action = self.random_or_override.weighted_random_choice(self.num_actions, np.squeeze(action_probs))

        return self._convert_action_num(action)

    def record_action_reward(self, state, action, score):
        # The key is a pair of encoded state and encoded action.
        self.action_totals[(state, self.encode_action(action))] += score
    
    def real_action(self,state,possible_actions,turn):
        state = state.to_observable_state(turn)
        state_string = state.tostring()
        total_count = self.state_counts[state_string]
        policy_vector = [0]*self.num_actions
        for action in possible_actions:
            policy_vector[self.encode_action(action)]=self.action_counts[(state_string, self.encode_action(action))]/total_count
        action_choosen = self.random_or_override.weighted_random_choice(len(policy_vector), policy_vector)
        example = Example(policy_vector,possible_actions,state)
        self.examples.append(example)
        return self._convert_action_num(action_choosen)
    
    def train_and_clear(self):
        states = np.array([example.state for example in self.examples])
        possible_actions = [self.encode_possible_actions(example.possible_actions,False) for example in self.examples]
        possible_actions_tensor = tf.convert_to_tensor(np.array(possible_actions),dtype=tf.float32)
        policy_vector = np.array([example.policy_vector for example in self.examples])
        self.model.fit([states,possible_actions_tensor],policy_vector,verbose = 0)
        self.examples.clear()
        self._reset_state_and_action_counts()
        self.model.save_weights(self.file_name)



    def no_op_action(self, state, possible_actions, turn):
        state = state.to_observable_state(turn)
        state = np.array([state])
        possible_actions_encoded = self.encode_possible_actions(possible_actions,True)
        action_probs = self.model.predict([state,possible_actions_encoded],callbacks=[self.tensorboard])
        return self._convert_action_num(np.argmax(action_probs))

    def _upper_confidence_bound(self, state, action_probs, possible_actions):
        best_action = (-1, -np.inf)
        for a_tuple, p in np.ndenumerate(action_probs):
            a = a_tuple[1]
            if self._convert_action_num(a) in possible_actions:
                q = 0
                if self.action_counts[(state, a)] != 0:
                    q = self.action_totals[(state, a)] / self.action_counts[(state, a)]
                action_score = q + p * np.sqrt(self.state_counts[state]) / (1.0 + self.action_counts[(state, a)])
                if action_score > best_action[1]:
                    best_action = (a, action_score)
        return best_action[0]

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
    
    def encode_possible_actions(self,possible_actions,to_tensor):
        action_mask = [-10000000000000000000000000000000000]*self.num_actions
        for action in possible_actions:
          action_num = self.encode_action(action)
          action_mask[action_num] = 0.0
        if to_tensor:
            return tf.convert_to_tensor(np.array([action_mask]),dtype=tf.float32)
        return action_mask
        

    
