import sys
from dataclasses import dataclass, field

from constants import PER_GAME, PER_TURN, WIN_LOSS, SCORE


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = 100000
   # Number of games for the model to assess
   assess_model_games : int = 10
   # How fast the model learns
   learning_rate: float = 0.0008
   # How fast the model changes gradients
   alpha: float = 1e-7
   # Discount factor
   gamma: float = .968
   # Print the model internals.
   print_model_nn: bool = False
   # How many epoch to evaluate
   accuracy_interval: int = 10
   # How many epoch to the model is saved
   save_interval: int = 10000
   # Which value function to use
   pgr: int  = SCORE 
   # How many simulations of tree search to run
   num_simulations : int = 20
   # Seed to use for random in the environment
   ers : int = 71
   # How many round per game.
   round_limit: int = 1
   # How many hidden layers will exist in the neural net.
   num_hl: int = 2
   # Size of the hidden layers in the neural net.
   hl_size: int = 235
   # Max depth for Monte Carlo tree-search.
   max_depth: int = 3
   # Log to Tensorboard
   tb_log: bool = False
   # Load the model
   load: bool = False
   # Use upper-confidence bounds for exploration.
   use_ucb: bool = True
   
