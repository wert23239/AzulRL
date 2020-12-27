import sys
from dataclasses import dataclass

@dataclass
class Settings:
   # Log to Tensorboard
   tb_log: bool = True
   # Load the model
   load: bool = False
   # Use upper-confidence bounds for exploration.
   use_ucb: bool = True