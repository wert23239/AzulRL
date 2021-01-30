import random

import environment_state
from action import Action
from random_or_override import RandomOrOverride
from hyper_parameters import HyperParameters
from tree_search_agent import TreeSearchAgent
from util import human_print


class HumanPlayer:
    def __init__(self, name, tree_search_agent):
        random = RandomOrOverride()
        hyper_parameters = HyperParameters()
        self.name = name
        self.tree_search_agent = tree_search_agent

    def action(self,  environment):
        possible_actions = environment.possible_moves
        mosaic_template = [[1, 2, 3, 4, 5],
                       [5, 1, 2, 3, 4],
                       [4, 5, 1, 2, 3],
                       [3, 4, 5, 1, 2],
                       [2, 3, 4, 5, 1]]
        print("name: ", self.name)
        human_print(environment.turn, environment.state)
        user_action_str = input(
            "And refer to the floor of your board as row 5:\n\n")
        while True:
            if user_action_str == 'r':
                random_action_idx = random.randint(0, len(possible_actions))
                return list(possible_actions)[random_action_idx]
            elif user_action_str == 'b':
                return self.tree_search_agent.action(environment)
            user_actions = user_action_str.split(",")
            if len(user_actions) == 3:
                # The following will crash the program if the user's input isn't
                # valid as integers.
                my_action = Action(
                    int(user_actions[0]), int(user_actions[1]), int(user_actions[2]))
                if my_action in possible_actions:
                    return my_action
            user_action_str = input("Invalid action, try again:\n\n")

    def get_action_from_bot(self, turn):
        return Action(0, 0, 0)