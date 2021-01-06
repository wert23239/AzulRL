import copy
import sys
from collections import defaultdict

import numpy as np

from ai_algorithm import AIAlgorithm
from constants import PER_GAME, PER_TRAIN
from environment import Environment
from human_player import HumanPlayer
from hyper_parameters import HyperParameters
from player_metrics import PlayerMetrics
from random_agent import RandomAgent
from random_or_override import RandomOrOverride
from settings import Settings
from tree_search_agent import TreeSearchAgent
from util import assess_agent

"""
Usage: python main.py player1_type player2_type <train_bots>
Acceptable types include 'random', 'bot', 'ai', and 'human'.
Train bots is a boolean that will cause the agent to train when it's set to
True and the 'bot' option is used.
"""

def main(player1_type, player2_type, hyper_parameters, settings):
    random = RandomOrOverride()
    total_score_against_ai=0
    is_playing_bot = True
    if(player1_type == "human" or player2_type == "human"):
        is_playing_bot = False
        settings.tb_log = False
        settings.load = True
    bot = TreeSearchAgent(random,hyper_parameters,settings,"Bilbo")
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
    e = Environment(round_limit=hyper_parameters.round_limit, random_seed=hyper_parameters.ers)
    wins = 0
    losses = 0
    ties = 0
    assess_count = 0
    for number_of_games in range(1,hyper_parameters.max_games):
        turn = e.reset()
        done = False
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            action = player.action(e)
            state, turn, _, score_delta, current_scores, done = e.move(action)
            if not is_playing_bot:
                print("score delta: ", score_delta)
                print("current scores: ", current_scores) 
            if(done):
                if type(m1) == TreeSearchAgent and settings.reset_time == PER_GAME:
                    m1.clear()
                if(current_scores[0]>current_scores[1]):
                    wins += 1
                elif(current_scores[1]<current_scores[0]):
                    losses +=1
                else:
                    ties += 1
        if (number_of_games % hyper_parameters.accuracy_interval == 0):
            print()
            print("win loss ratio: ",wins/(losses+wins+ties))
            print("Epoch: ",number_of_games)
            if type(m1) == TreeSearchAgent:
                if settings.reset_time == PER_TRAIN:
                    m1.clear()
                legal_ratio = m1.model.legal_moves/(m1.model.legal_moves+m1.model.illegal_moves)
                total_moves = m1.model.legal_moves+m1.model.illegal_moves
                print("for Epoch, legal:illegal moves ratio: ", m1.model.legal_moves/(m1.model.legal_moves+m1.model.illegal_moves))
                print("for Epoch, total moves: ", m1.model.legal_moves+m1.model.illegal_moves)
                player_metrics = PlayerMetrics(wins,losses,ties,legal_ratio,total_moves)
                m1.model.legal_moves = 0
                m1.model.illegal_moves = 0
                m1.model.model.save_weights("tmp.h5")
                m1_shadow = m1
                m1_shadow.train()
                m1.model.model.load_weights("tmp.h5")
                assess_count +=1
                wins=assess_agent(m1_shadow, m1, e, hyper_parameters, assess_count, player_metrics)
                if (wins >= hyper_parameters.assess_model_games/2.0):
                    m1 = m1_shadow
                    print("model got better!!!!")
                else:
                    print("model got worse :(")
                score_against_ai=assess_agent(m1, AIAlgorithm(), e, hyper_parameters, assess_count, player_metrics,True)
                print("score against ai", score_against_ai)
                total_score_against_ai+=score_against_ai
                wins_against_random=assess_agent(m1, RandomAgent(random), e, hyper_parameters, assess_count, player_metrics)
                print("wins against random", wins_against_random)
            wins = 0
            losses = 0
            ties = 0
    return score_against_ai/assess_count


if __name__ == "__main__":
    hyper_parmeters = HyperParameters()
    settings = Settings()
    player_types = ["random", "bot", "human", "ai"]
    valid_args = (
        len(sys.argv) >= 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        if len(sys.argv) == 1:
            main("bot", "bot", hyper_parmeters, settings)
        else:
            main(sys.argv[1], sys.argv[2], hyper_parmeters, settings)
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', 'human', and 'ai'.")
