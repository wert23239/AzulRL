from dataclasses import dataclass

@dataclass
class Example:
   reward: int
   action: int
   possible_actions : list
   next_possible_actions: list
   state: list
   next_state: list
   done: bool
