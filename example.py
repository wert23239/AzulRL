from dataclasses import dataclass

@dataclass 
class Example:
   reward: int
   action: int
   state: list
   previous_action: int
   previous_state: list 