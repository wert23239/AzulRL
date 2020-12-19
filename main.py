import sys
import numpy as np

from environment import Environment
from human_player import HumanPlayer
from hyper_parameters import HyperParameters
from random_agent import RandomAgent
from random_or_override import RandomOrOverride
from tree_search_agent import TreeSearchAgent
from util import assess_agent

from collections import defaultdict

"""
Usage: python main.py player1_type player2_type <train_bots>
Acceptable types include 'random', 'bot', and 'human'.
Train bots is a boolean that will cause the agent to train when it's set to
True and the 'bot' option is used.
"""

def main(player1_type, player2_type, hyper_parameters):
    random = RandomOrOverride()
    best_avg_score=0
    is_playing_bot = True
    if(player2_type == "human"):
        is_playing_bot = False
    bot = TreeSearchAgent(random,hyper_parameters,"Bilbo",not is_playing_bot)
    if player1_type == "bot":
        m1 = bot
    elif player1_type == "random":
        m1 = RandomAgent(random)
    elif player1_type == "human":
        m1 = HumanPlayer("alex")
    else:
        m1 = AIAlgorithm()
    if player2_type == "bot":
        m2 = bot
    elif player2_type == "random":
        m2 = RandomAgent(random)
    elif player2_type == "human":
        m2 = HumanPlayer("erica")
    else:
        m2 = AIAlgorithm()
    e = Environment(round_limit=1, random_seed=hyper_parameters.environment_random_seed)
    wins = 0
    losses = 0
    for number_of_games in range(1,hyper_parameters.max_games):
        turn = e.reset()
        done = False
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            action = player.action(e, is_playing_bot)
            state, turn, _, score_delta, current_scores, done = e.move(action)
            if not is_playing_bot:
                print("score delta: ", score_delta)
                print("current scores: ", current_scores) 
            if(done):
                if(current_scores[0]>current_scores[1]):
                    wins += 1
                else:
                    losses += 1
        if number_of_games % hyper_parameters.accuracy_interval == 0:
            print("win loss ratio: ",wins/(losses+wins))
            print("Epoch: ",number_of_games)
            avg_score=assess_agent(m1, random, e, "player 1",hyper_parameters)
            best_avg_score = max(avg_score,best_avg_score)
            wins = 0
            losses = 0
    return best_avg_score


if __name__ == "__main__":
    hyper_parmeters = HyperParameters()
    player_types = ["random", "bot", "human", "ai"]
    valid_args = (
        len(sys.argv) >= 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        if len(sys.argv) == 1:
            main("bot", "bot", hyper_parmeters)
        else:
            if len(sys.argv) >= 4:
                train = sys.argv[3]
            main(sys.argv[1], sys.argv[2], hyper_parmeters)
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', 'human', and 'ai'.")
