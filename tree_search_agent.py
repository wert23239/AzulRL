import copy
from collections import defaultdict
import operator
from random_model_with_scored_actions import RandomModelWithScoredActions
from random_or_override import RandomOrOverride
from statistics import mean

class TreeSearchAgent:
    def __init__(self, random, hyper_parameters):
        self.num_simulations = hyper_parameters.num_simulations
        self.r = random
        self.model = RandomModelWithScoredActions(random)

    def action(self, environment, _):
        possible_actions_list = list(environment.possible_moves)
        possible_actions_map = self.model.simulated_action(environment.state, possible_actions_list)
        score_map = defaultdict(list)  # map from action to list of rewards when that action is done
        for i in range(self.num_simulations):
            # probabilistically choose action from possible_actions_map
            action = self.r.weighted_random_choice(possible_actions_list,
                [possible_actions_map[a] for a in possible_actions_list])
            e = copy.deepcopy(environment)
            score_map[action].append(self._find_action_value(action, e))
        average_scores = {action_score: mean(score_map[action_score]) for action_score in score_map}
        
        return max(average_scores.items(), key=operator.itemgetter(1))[0]

    def _find_action_value(self,action, environment):
        turn = environment.turn
        state, _, possible_actions, _, total_rewards, done = environment.move(action)
        while not done:
            possible_actions_list = list(possible_actions)
            possible_actions_map = self.model.simulated_action(state, possible_actions_list)
            # probabilistically choose action from possible_actions_map
            a = self.r.weighted_random_choice(possible_actions_list,
                [possible_actions_map[a] for a in possible_actions_list])
            state, _, possible_actions, _, total_rewards, done = environment.move(a)
        print("Reward")
        print("Turn",turn)
        print("Output",total_rewards)
        print("Result",total_rewards[turn] - total_rewards[(turn + 1) % 2])
        print("action",action)
        return total_rewards[turn] - total_rewards[(turn + 1) % 2]

