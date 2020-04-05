from environment import Environment
from DQN_agent import DQNAgent
from random_or_override import RandomOrOverride
from random_model import RandomModel
from human_player import HumanPlayer
from example import Example
import numpy as np
import sys

"""
Usage: python main.py player1_type player2_type <train_bots>
Acceptable types include 'random', 'bot', and 'human'.
Train bots is a boolean that will cause the model to train when it's set to
True and the 'bot' option is used.
"""


def score_to_reward(score):
    if score <= -2:
        return -1
    if score >= 2:
        return 1
    return 0


def main(player1_type, player2_type, train_bots):
    random = RandomOrOverride()
    if player1_type == "bot":
        m1 = DQNAgent(random, train=train_bots)
    elif player1_type == "random":
        m1 = RandomModel(random)
    else:
        m1 = HumanPlayer()
    if player2_type == "bot":
        m2 = DQNAgent(random, train=train_bots)
    elif player2_type == "random":
        m2 = RandomModel(random)
    else:
        m2 = HumanPlayer()
    e = Environment(random)

    player1_scores = []
    iter = 0
    while True:
        state, turn, possible_actions = e.reset()
        done = False
        previous_state = state  # FIX LATER
        while not done:
            if turn == 1:
                player = m1
            else:
                player = m2
            action = player.action(state, possible_actions, turn)
            state, turn, possible_actions, score, done = e.move(action)
            reward = score_to_reward(score)
            example = Example(reward, action, possible_actions, previous_state.to_observable_state(
            ), state.to_observable_state(), done)
            player.save(example)
            previous_state = state
        print("round over")
        if not train:
          player1_scores.append(score)
          iter += 1
          if iter % 100 == 0:
            print(sum(player1_scores) / len(player1_scores))




if __name__ == "__main__":
    player_types = ["random", "bot", "human"]
    valid_args = (
        len(sys.argv) >= 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        train = True  # By default, we train the model.
        if len(sys.argv) == 1:
            main("bot", "random", train)
        else:
          if len(sys.argv) >= 4:
            train = sys.argv[3]
          main(sys.argv[1], sys.argv[2], train)
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', and 'human'.")
