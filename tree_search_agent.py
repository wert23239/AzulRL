import copy
from collections import defaultdict
import operator
from random_model_with_scored_actions import RandomModelWithScoredActions
from random_or_override import RandomOrOverride
from statistics import mean

def find_action_value(action, environment):
    turn = environment.turn
    state, _, possible_actions, _, total_rewards, done = environment.move(action)
    while not done:
        possible_actions_list = list(possible_actions)
        possible_actions_map = self.model.action(state, possible_actions_list)
        # probabilistically choose action from possible_actions_map
        action = self.r.weighted_random_choice(possible_actions_list,
            [possible_actions_map[a] for a in possible_actions_list])
        state, _, possible_actions, _, total_rewards, done = environment.move(action)

    return total_rewards[turn] - total_rewards[(turn + 1) % 2]


class TreeSearchAgent:
    def __init__(num_simulations):
        self.num_simulations = num_simulations
        self.r = RandomOrOverride()
        self.model = RandomModelWithScoredActions(r)

    def action(self, environment, _):
        possible_actions_list = list(environment.possible_moves)
        possible_actions_map = self.model.action(environment.state, possible_actions_list)
        score_map = defaultdict(list)  # map from action to list of rewards when that action is done
        for i in xrange(num_simulations):
            # probabilistically choose action from possible_actions_map
            action = self.r.weighted_random_choice(possible_actions_list,
                [possible_actions_map[a] for a in possible_actions_list])
            e = copy.deepcopy(environment)
            score_map[a].append(find_action_value(action, e))
        average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
        return max(average_scores.iteritems(), key=operator.itemgetter(1))[0]

    def save(self, example):
        pass

    def train(self):
        pass

    def updateFinalReward(self,reward):
        pass

