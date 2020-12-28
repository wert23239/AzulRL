import sys
from dataclasses import dataclass

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