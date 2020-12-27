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
        self.pg_examples = []
        self.visited = set()

    def action(self, environment):
        possible_actions_list = list(environment.possible_moves)
        score_map = defaultdict(list)  # map from action to list of rewards when that action is done
        for i in range(self.num_simulations):
            e = copy.deepcopy(environment)
            action = self.model.simulated_action(environment.state, possible_actions_list, environment.turn)
            state_actions, reward = self._find_action_value(action, e)
            for s, a in state_actions:
                self.model.record_action_reward(environment.state, action, reward)
            score_map[action].append(reward)
        average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
        return max(average_scores.items(), key=operator.itemgetter(1))[0]


    def _find_action_value(self,action, environment):
        turn = environment.turn
        state_actions = [(environment.state.to_observable_state(turn).tostring(), action)]
        state, temp_turn, possible_actions, _, total_rewards, done = environment.move(action)
        while not done:
            hashable_state = state.to_observable_state(temp_turn).tostring()
            if hashable_state not in self.visited:
                self.visited.add(hashable_state)
                value = self.calculate_value(total_rewards[turn] - total_rewards[(turn + 1) % 2])
                return state_actions, value
            possible_actions_list = list(possible_actions)
            a = self.model.simulated_action(state, possible_actions_list, temp_turn)
            state_actions.append((hashable_state, a))
            state, temp_turn, possible_actions, _, total_rewards, done = environment.move(a)
        value = self.calculate_value(total_rewards[turn] - total_rewards[(turn + 1) % 2])
        return state_actions, value

    def calculate_value(self, reward):
        returns = []
        if(self.hyper_parameters.pgr == WIN_LOSS):
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
        else:
            reward=min(reward,10)
            reward=max(-10,reward)
            reward=float(reward/10)
            return reward



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
    
    def save(self,example):
        self.pg_examples.append(example)

    def train(self):
        self.model.train(self.pg_examples)
        self.pg_examples.clear()


