import sys
from dataclasses import dataclass, field

from constants import PER_GAME, PER_TURN, WIN_LOSS, SCORE


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = 10000
   # Number of games for the model to assess
   assess_model_games : int = 20
   # How fast the model learns
   learning_rate: float = 0.005
   # How fast the model changes gradients
   alpha: float = 1e-7
   # Discount factor
   gamma: float = .99
   # Print the model internals.
   print_model_nn: bool = False
   # How many epoch to evaluate
   accuracy_interval: int = 10
   # How many epoch to the model is saved
   save_interval: int = 10
   # Which reward function to use
   reward_function : int = PER_GAME
   pgr: int  = WIN_LOSS
   # How many simulations of tree search to run
   num_simulations : int = 7
   # Seed to use for random in the environment
   ers : int = 69
   # How many round per game.
   round_limit: int = 2
   # How many hidden layers will exist in the neural net.
   num_hl: int = 1
   # Size of the hidden layers in the neural net.
   hl_size: int = 128
   # Max depth for Monte Carlo tree-search.
   max_depth: int = 5
   
