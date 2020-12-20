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
    def __init__(self, random, hyper_parameters,name="Bilbo",human=False):
        self.num_simulations = hyper_parameters.num_simulations
        self.r = random
        self.model = PolicyGradientModel(random,hyper_parameters,name,human)  #RandomModelWithScoredActions(random)

    def action(self, environment, train,final =False):
        possible_actions_list = list(environment.possible_moves)
        if train:
            score_map = defaultdict(list)  # map from action to list of rewards when that action is done
            for i in range(self.num_simulations):
                e = copy.deepcopy(environment)
                with tf.GradientTape() as tape:
                    action = self.model.simulated_action(environment.state, possible_actions_list, environment.turn)
                    reward = self._find_action_value(action, e)
                    self.model.train(reward, tape)
                    score_map[action].append(reward)
            average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
            
            return max(average_scores.items(), key=operator.itemgetter(1))[0]
        else:
            return self.model.greedy_action(environment.state, possible_actions_list, environment.turn,final)


    def _find_action_value(self,action, environment):
        turn = environment.turn
        state, temp_turn, possible_actions, _, total_rewards, done = environment.move(action)
        while not done:
            possible_actions_list = list(possible_actions)
            a = self.model.simulated_action(state, possible_actions_list, temp_turn)
            state, temp_turn, possible_actions, _, total_rewards, done = environment.move(a)
        return total_rewards[turn] - total_rewards[(turn + 1) % 2]
