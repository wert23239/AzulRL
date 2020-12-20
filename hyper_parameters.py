import sys
from dataclasses import dataclass, field

from constants import PER_GAME, PER_TURN


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = 10000
   # Number of games for the model to assess
   assess_model_games : int = 3
   # How fast the model learns
   learning_rate: float = 0.005
   # Print the model internals.
   print_model_nn: bool = False
   # How many epoch to evaluate
   accuracy_interval: int = 100
   # How many epoch to the model is saved
   save_interval: int = 10000
   # Which reward function to use
   reward_function : int = PER_TURN
   # How many simulations of tree search to run
   num_simulations : int = 1
   # Seed to use for random in the environment
   environment_random_seed : int = 69
   # How many round per game.
   round_limit: int = 1
   # Model size. (List of ints, each of which represents the size of a layer in the neural net.)
   model_size: list = field(default_factory=list)
   def add_layer_to_model(self, layer):
      self.model_size.append(layer)
   
