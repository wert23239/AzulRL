import sys
from dataclasses import dataclass

from constants import PER_GAME, PER_TURN


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = 750
   # How many examples to sample from when training
   batch_size: int  = 128
   # How many examples should the model remember
   memory_length: int = 5000
   # How future reward in the same count towards the next game
   discount_factor: float = .99
   # Minimum  exploration value
   epsilon_min: float = .01
   # The amount of epislon decay per training
   epsilon_decay: float = .999
   # How fast the model learns
   learning_rate: float = .005
   # How fast the target model transfers knowledge
   tau: float = .125
   # Print the model internals.
   print_model_nn: bool = False
   # How many epoch to evaluate
   accuracy_interval: int = 10
   # How many epoch to the model is saved
   save_interval: int = 5
   # Which reward function to use
   reward_function : int = PER_GAME
   # How many simulations of tree search to run
   num_simulations : int = 10
   # Seed to use for random in the environment
   environment_random_seed : int = 69
   
