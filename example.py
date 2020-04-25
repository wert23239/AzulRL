from dataclasses import dataclass, field


@dataclass
class Example:
   reward: int
   done: bool
   action: int = -1
   possible_actions : list = field(default_factory=list)
   state: list = field(default_factory=list)
   next_possible_actions: list = field(default_factory=list)
   next_state: list = field(default_factory=list)
