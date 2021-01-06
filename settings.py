import sys
from dataclasses import dataclass
from constants import PER_GAME, PER_TRAIN

@dataclass
class Settings:
   # Log to Tensorboard
   tb_log: bool = True
   # Load the model
   load: bool = False
   # Save the model during training
   save:bool = True
   # Use upper-confidence bounds for exploration.
   use_ucb: bool = True
   # When does the model reset the visited, state_counts, action_counts, actions_totals
   reset_time: int = PER_GAME 