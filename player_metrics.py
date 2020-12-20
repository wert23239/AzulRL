from dataclasses import dataclass, field

@dataclass
class PlayerMetrics:
   wins: int = 0
   losses: int = 0
   ties: int = 0
   illegal_moves: float = 0
   total_moves: int = 0
