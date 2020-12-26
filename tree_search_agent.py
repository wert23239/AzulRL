import copy
from collections import defaultdict
import operator
from policy_gradient_model import PolicyGradientModel
from random_model_with_scored_actions import RandomModelWithScoredActions
from random_or_override import RandomOrOverride
from statistics import mean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TreeSearchAgent:
    def __init__(self, random, hyper_parameters,name="Bilbo"):
        self.num_simulations = hyper_parameters.num_simulations
        self.r = random
        self.model = PolicyGradientModel(random,hyper_parameters,name)
        self.mc_max_depth = hyper_parameters.max_depth
        self.pg_examples = []

    def action(self, environment):
        possible_actions_list = list(environment.possible_moves)
        score_map = defaultdict(list)  # map from action to list of rewards when that action is done
        state_counts = defaultdict(int)
        action_counts = defaultdict(int)
        for i in range(self.num_simulations):
            e = copy.deepcopy(environment)
            action = self.model.simulated_action(environment.state, possible_actions_list, environment.turn, state_counts, action_counts)
            reward = self._find_action_value(action, e, state_counts, action_counts)
            score_map[action].append(reward)
        average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
        return max(average_scores.items(), key=operator.itemgetter(1))[0]


    def _find_action_value(self,action, environment, state_counts, action_counts):
        turn = environment.turn
        state, temp_turn, possible_actions, _, total_rewards, done = environment.move(action)
        depth = 0
        while not done and depth < self.mc_max_depth:
            possible_actions_list = list(possible_actions)
            a = self.model.simulated_action(state, possible_actions_list, temp_turn, state_counts, action_counts)
            state, temp_turn, possible_actions, _, total_rewards, done = environment.move(a)
            depth += 1
        return total_rewards[turn] - total_rewards[(turn + 1) % 2]
    
    def save(self,example):
        self.pg_examples.append(example)

    def train(self):
        self.model.train(self.pg_examples)
        self.pg_examples.clear()


