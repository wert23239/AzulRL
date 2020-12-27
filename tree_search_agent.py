import copy
from collections import defaultdict
import operator
from policy_gradient_model import PolicyGradientModel
from random_model_with_scored_actions import RandomModelWithScoredActions
from random_or_override import RandomOrOverride
from constants import  WIN_LOSS
from statistics import mean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TreeSearchAgent:
    def __init__(self, random, hyper_parameters,name="Bilbo"):
        self.hyper_parameters = hyper_parameters
        self.r = random
        self.model = PolicyGradientModel(random,hyper_parameters,name)
        self.mc_max_depth = hyper_parameters.max_depth
        self.visited = set()

    def action(self, environment):
        possible_actions_list = list(environment.possible_moves)
        for i in range(self.hyper_parameters.num_simulations):
            e = copy.deepcopy(environment)
            action = self.model.simulated_action(environment.state, possible_actions_list, environment.turn)
            state_action_turns, reward = self._find_action_value(action, e)
            for s, a, t in state_action_turns:
                self.model.record_action_reward(s, a, t, reward)
        return self.model.real_action(environment.state,possible_actions_list,environment.turn)


    def _find_action_value(self,action, environment):
        turn = environment.turn
        state_action_turns = [(environment.state, action, turn)]
        state, temp_turn, possible_actions, _, total_rewards, done = environment.move(action)
        while not done:
            hashable_state = state.to_hashable_state(temp_turn)
            if hashable_state not in self.visited:
                self.visited.add(hashable_state)
                value = self.calculate_value(total_rewards[turn] - total_rewards[(turn + 1) % 2])
                return state_action_turns, value
            possible_actions_list = list(possible_actions)
            a = self.model.simulated_action(state, possible_actions_list, temp_turn)
            state_action_turns.append((state, a, temp_turn))
            state, temp_turn, possible_actions, _, total_rewards, done = environment.move(a)
        value = self.calculate_value(total_rewards[turn] - total_rewards[(turn + 1) % 2])
        return state_action_turns, value

    def calculate_value(self, reward):
        if(self.hyper_parameters.pgr == WIN_LOSS):
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
            return reward
        else:
            reward=min(reward,10)
            reward=max(-10,reward)
            reward=float(reward/10)
            return reward

    def train(self):
        self.model.train()


