import sys
from dataclasses import dataclass

from constants import PER_GAME, PER_TURN


@dataclass
class HyperParameters:
   # Number of games
   max_games: int = sys.maxsize
   # How many examples to sample from when training
   batch_size: int  = 128
   # How many examples should the model remember
   memory_length: int = 5000
   # How future reward in the same count towards the next game
   discount_factor: float =  .85
   # Minimum  exploration value
   epsilon_min: float = .01
   # The amount of epislon decay per training
   epsilon_decay: float = .999
   # How fast the model learns
   learning_rate: float = .005
   # How fast the target model transfers knowledge
   tau: float = .125
   # How many epochs to train
   train_interval: int = 10
   # How many epoch to evaluate
   accuracy_interval: int = 100
   # How many epochs ot train the target
   target_train_interval: int = 2
   # Which reward function to use
   reward_function : int = PER_TURN
