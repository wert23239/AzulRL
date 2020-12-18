import copy
from collections import defaultdict
from environment import Environment
import operator
from random_model_with_scored_actions import RandomModelWithScoredActions
from random_or_override import RandomOrOverride
from statistics import mean


def create_new_environment(state, possible_actions, turn):
    r = RandomOrOverride()
    e = Environment(r, True)  # game_ends_after_round
    e.state = copy.deepcopy(state)
    e.possible_moves = copy.deepcopy(possible_actions)
    e.turn = turn
    e.done = False
    e.previous_rewards = [0, 0]
    return e


def find_action_value(action, environment, scores):
    e = create_new_environment(
        environment.state, environment.possible_moves, environment.turn)
    state, turn, possible_actions, score, done = e.move(action)
    if done:
        return sum(scores) + score

    possible_actions_list = list(possible_actions)
    possible_actions_map = self.model.action(state, possible_actions_list)
    # probabilistically choose action from possible_actions_map
    a = self.r.weighted_random_choice(possible_actions_list,
        [possible_actions_map[possible_a] for possible_a in possible_actions_list])
    return -1 * find_action_value(a, e, scores + [score])


class TreeSearchAgent:
    def __init__(num_simulations=10):
        self.num_simulations = num_simulations
        self.r = RandomOrOverride()
        self.model = RandomModelWithScoredActions(r)

    def action(self, state, possible_actions, turn, _):
        e = create_new_environment(state, possible_actions, turn)
        possible_actions_list = list(possible_actions)
        possible_actions_map = self.model.action(state, possible_actions_list)
        score_map = defaultdict(list)  # map from action to list of rewards when that action is done
        for i in xrange(num_simulations):
            # probabilistically choose action from possible_actions_map
            a = self.r.weighted_random_choice(possible_actions_list,
                [possible_actions_map[possible_a] for possible_a in possible_actions_list])
            val = -1 * find_action_value(a, e, []])
            score_map[a].append(val)
        average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
        return max(average_scores.iteritems(), key=operator.itemgetter(1))[0]

    def save(self, example):
        self.model.save(example)

    def train(self):
        self.model.train(example)
