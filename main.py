from environment import Environment
from DQN_agent import DQNAgent
from random_or_override import RandomOrOverride
from random_model import RandomModel
from human_player import HumanPlayer
from example import Example
import numpy as np
import sys

"""
Usage: python main.py player1_type player2_type
Acceptable types include 'random', 'bot', and 'human'.
"""


def main(player1_type, player2_type):
    random = RandomOrOverride()
    if player1_type == "bot":
        m1 = DQNAgent(random)
    elif player1_type == "random":
        m1 = RandomModel(random)
    else:
        m1 = HumanPlayer()
    if player2_type == "bot":
        m2 = DQNAgent(random)
    elif player2_type == "random":
        m2 = RandomModel(random)
    else:
        m2 = HumanPlayer()
    e = Environment(random)

    while True:
        state, turn, possible_actions = e.reset()
        done = False
        previous_state = state # FIX LATER
        while not done:
            if turn == 1:
                player = m1
            else:
                player = m2
            action = player.action(state, possible_actions, turn)
            state, turn, possible_actions, reward, done = e.move(action)
            example = Example(reward,action,possible_actions,previous_state.to_observable_state(),state.to_observable_state(),done)
            player.save(example)
            previous_state = state
        print("round over")


if __name__ == "__main__":
    player_types = ["random", "bot", "human"]
    valid_args = (
        len(sys.argv) == 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        if len(sys.argv) == 1:
            main("bot", "random")
        else:
            main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', and 'human'.")
