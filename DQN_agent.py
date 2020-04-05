import random

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from random_or_override import RandomOrOverride


STATE_SPACE = 164
ACTION_SPACE = 180
BATCH_SIZE = 32


class DQNAgent():
    def __init__(self, random_or_override):
        self.memory = deque(maxlen=2000)
        self.discount_factor = .95
        # exploration vs. exploitation  params
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

        self.model = self.__create_model()
        self.target_model = self.__create_model()

    def action(self, state, possible_actions, turn):
        return possible_actions.pop()

    def __convert_action_num(self, action_number):
        pass

    def save(self, example):
        memory_discounted = self.__caculate_discount_reward()
        self.memory.extend(memory_discounted)

    def __caculate_discount_reward(self):
        return self.memory

    def __create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=STATE_SPACE, activation='relu'))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(ACTION_SPACE))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model
