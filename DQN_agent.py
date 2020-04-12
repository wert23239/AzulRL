import random
from collections import defaultdict, deque

from keras import Model, Sequential
from keras import backend as K
from keras.layers import Dense, Input, Lambda, Multiply, merge, Subtract
from keras.optimizers import Adam
from numpy import argmax, array

import action
from action import Action
from constants import NUMBER_OF_CIRCLES, NUMBER_OF_COLORS, NUMBER_OF_ROWS
from random_or_override import RandomOrOverride

STATE_SPACE =  185
ACTION_SPACE = 180

class DQNAgent():
    def __init__(self, random_or_override,hyper_parameters, human=False):
        self.batch_size =  hyper_parameters.batch_size
        self.random_or_override = random_or_override
        self.memory = deque(maxlen=hyper_parameters.memory_length)
        self.discount_factor = hyper_parameters.discount_factor
        # exploration vs. exploitation  params
        self.epsilon = 1.0
        self.epsilon_min = hyper_parameters.epsilon_min
        self.epsilon_decay = hyper_parameters.epsilon_decay
        self.learning_rate = hyper_parameters.learning_rate
        self.target_train_interal = hyper_parameters.target_train_interval
        self.train_count = 0

        self.tau = 0.125
        self.first_choices = defaultdict(int)

        self.model = self.__create_model()
        if human:  # Add check for weights file exisiting
            self.model.load_weights('DQN_complete_weights.h5')
        self.target_model = self.__create_model()

    def action(self, state, possible_actions, turn, train):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if train and self.random_or_override.random_range_cont() < self.epsilon:
            l = list(possible_actions)
            result = self.random_or_override.random_sample(l, 1)[0]
            if result not in possible_actions:
              raise Exception
            return (result, self.encode_action(result), -1)
        observable_state = state.to_observable_state(turn)
        state_as_example = array([observable_state])
        action_mask, final_mask = self.convert_action_space_to_bit_mask(possible_actions)
        predictions =  self.model.predict([state_as_example,action_mask, final_mask])[0]
        choosen_action = argmax(predictions)
        if self.convert_action_num(choosen_action) not in possible_actions:
          raise Exception
        return (self.convert_action_num(choosen_action),choosen_action,0) # Figure out how to get wrong guess

    def convert_action_space_to_bit_mask(self,possible_actions):
        action_mask = [0]*ACTION_SPACE
        output_mask = [1]*ACTION_SPACE
        for action in possible_actions:
          action_num = self.encode_action(action)
          action_mask[action_num] = 1
          output_mask[action_num] = 0

        return array([action_mask]), array([output_mask])


    def get_best_possible_action(self, prediction, possible_actions):
        prediction_list = prediction.argsort().tolist()
        wrong_guesses = 0
        first_choice = True
        while len(prediction_list):
            action_number = prediction_list.pop()
            action = self.convert_action_num(action_number)
            if first_choice:
                self.first_choices[action] += 1
                first_choice = False
            if action in possible_actions:
                return (action, action_number, wrong_guesses)
            wrong_guesses += 1
        raise Exception

    def encode_action(self, action):
        action_num = action.circle * 30
        action_num += (action.color-1) * 6
        action_num += action.row
        return action_num


    def convert_action_num(self, action_number):
        circle = action_number // (NUMBER_OF_COLORS * NUMBER_OF_ROWS)
        action_number = action_number % (NUMBER_OF_COLORS * NUMBER_OF_ROWS)
        color = action_number // (NUMBER_OF_ROWS) + 1
        row = action_number % NUMBER_OF_ROWS
        return Action(circle, color, row)

    def save(self, example):
        self.memory.append(example)

    def train(self):
        self.epsilon *= self.epsilon_decay
        self.train_count += 1
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            example = sample
            action_mask, final_mask = self.convert_action_space_to_bit_mask(example.possible_actions)
            next_action_mask, next_final_mask = self.convert_action_space_to_bit_mask(example.next_possible_actions)
            state_as_example = array([example.state])
            next_state_as_example = array([example.next_state])
            action_mask_as_example = array([action_mask])
            next_action_mask_as_example = array([next_action_mask])
            target = self.target_model.predict([state_as_example,action_mask,final_mask])
            if self.convert_action_num(example.action) not in example.possible_actions:
              raise Exception
            if sample.done:
                target[0][example.action] = example.reward
            else:
                Q_future = max(self.target_model.predict([next_state_as_example,next_action_mask,next_final_mask])[0])
                reward = example.reward + Q_future * self.discount_factor
                target[0][example.action] = reward
            self.model.fit([state_as_example,action_mask,final_mask],
                           target, epochs=1, verbose=0)
        if self.train_count % self.target_train_interal == 0:
            self.__target_train()
            # self.model.save_weights("DQN_weights.h5")
            # self.model.save_weights("DQN_target_weights.h5")

    def __target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def __create_model(self):
        model = Sequential()
        main_input = Input(shape = (STATE_SPACE,))
        dense1 = Dense(24, activation='relu')(main_input)
        dense2 = Dense(48, activation='relu')(dense1)
        dense3 = Dense(24, activation='relu')(dense2)
        dense_advantage = Dense(24, activation='relu')(dense3)
        advantage = Dense(ACTION_SPACE)(dense_advantage)
        dense_value = Dense(24, activation='relu')(dense3)
        value = Dense(1)(dense_value)
        action_mask = Input(shape = (ACTION_SPACE,))
        dense_action_mask = Dense(ACTION_SPACE, activation='relu')(action_mask)
        advantage_masked = Multiply()([advantage, action_mask]) # [.24,.8,.63] * [1,0,1]=
        final_mask = Input(shape = (ACTION_SPACE,))
        policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1], (ACTION_SPACE,))([advantage_masked, value])
        policy_masked = Subtract()([policy, final_mask])
        model = Model(inputs=[main_input, action_mask, final_mask], outputs=[policy_masked])
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def updateFinalReward(self,reward):
        example = self.memory.pop()
        example.done = True
        example.reward = reward
        self.memory.append(example)