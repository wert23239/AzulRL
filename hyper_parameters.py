import sys
from dataclasses import dataclass

from constants import PER_GAME, PER_TURN, WIN_LOSS, SCORE


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = 100000
   # Number of games for the model to assess
   assess_model_games : int = 5
   # How fast the model learns
   learning_rate: float = 0.0004
   # How many epoch to evaluate
   accuracy_interval: int = 1
   # Which value function to use
   pgr: int = SCORE 
   # How many simulations of tree search to run
   num_simulations : int = 20
   # Seed to use for random in the environment
   ers : int = 71
   # How many round per game.
   round_limit: int = 1
   # How many hidden layers will exist in the neural net.
   num_hl: int = 5
   # Size of the hidden layers in the neural net.
   hl_size: int = 235
   # Epsilon (controls exploration/exploitation).
   eps: float = 1e-8
   # Number of states saved in history.
   history_size: int = 7