from dataclasses import dataclass, field


@dataclass
class Example:
   policy_vector: int = -1
   possible_actions : list = field(default_factory=list)
   history: list = field(default_factory=list)
