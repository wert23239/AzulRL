from dataclasses import dataclass, field


@dataclass
class Example:
   action: int = -1
   possible_actions : list = field(default_factory=list)
   state: list = field(default_factory=list)
