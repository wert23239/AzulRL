import random
from collections import deque

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from numpy import argmax,array

from action import Action
from constants import NUMBER_OF_CIRCLES, NUMBER_OF_COLORS, NUMBER_OF_ROWS
from random_or_override import RandomOrOverride

STATE_SPACE = 185
ACTION_SPACE = 180
BATCH_SIZE = 64
TRAIN_AMOUNT = 5
EPOCHS = 3


class DQNAgent():
    def __init__(self, random_or_override):
        self.random_or_override = random_or_override
        self.memory = deque(maxlen=2000)
        self.discount_factor = .95
        # exploration vs. exploitation  params
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.train_count = 0


        self.model = self.__create_model()
        self.target_model = self.__create_model()

    def action(self,state,possible_actions,_,train):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if train and self.random_or_override.random_range_cont() < self.epsilon:
            l = list(possible_actions)
            return (self.random_or_override.random_sample(l, 1)[0],0,-1)
        observable_state = state.to_observable_state()
        return self.__get_best_possible_action(self.model.predict(array([observable_state]))[0],possible_actions)

    def __get_best_possible_action(self,prediction,possible_actions):
        prediction_list = prediction.argsort().tolist()
        wrong_guesses = 0
        while(len(prediction_list)):
            action_number = prediction_list.pop()
            action = self.__convert_action_num(action_number)
            if action in possible_actions:
                return (action,action_number,wrong_guesses)
            wrong_guesses += 1
        raise Exception

    def __convert_action_num(self,action_number): #TEST
        circle = action_number // (NUMBER_OF_COLORS*NUMBER_OF_ROWS)
        action_number = action_number % (NUMBER_OF_COLORS*NUMBER_OF_ROWS)
        color = action_number // (NUMBER_OF_ROWS) + 1
        row = action_number % NUMBER_OF_ROWS
        return Action(circle,color,row)

    def save(self,example):
        self.memory.append(example)


    def train(self):
        self.epsilon *= self.epsilon_decay
        self.train_count += 1
        batch_size = BATCH_SIZE
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            example = sample
            target = self.target_model.predict(array([example.state]))
            if sample.done:
                target[0][example.action] = example.reward
            else:
                prediction = self.target_model.predict(array([example.next_state]))[0]
                action = self.__get_best_possible_action(prediction,example.possible_actions)[1]
                Q_future = prediction[action]
                reward = example.reward + Q_future * self.discount_factor
                target[0][example.action] = reward
            self.model.fit(array([example.state]),target, epochs=EPOCHS, verbose=0)
        if self.train_count % TRAIN_AMOUNT == 0:
            self.__target_train()
            print("epilson",self.epsilon)
            self.model.save_weights('DQN_weights.h5')
            self.model.save_weights('DQN_target_weights.h5')

    def __target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def __create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=STATE_SPACE, activation='relu'))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(ACTION_SPACE))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model
