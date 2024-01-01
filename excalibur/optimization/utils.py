from dataclasses import dataclass, field
from typing import Dict


DEFAULT_CONSTRAINT_EPS = 1e-4
DEFAULT_RANK_EPS = 1e-4
DEFAULT_GAP_EPS = 1e-6


@dataclass
class MultiThreshold:
    default: float = DEFAULT_CONSTRAINT_EPS
    specific: Dict[int, float] = field(default_factory=dict)

    def set(self, idx: int, thresh: float):
        self.specific[idx] = thresh

    def get(self, idx: int):
        if idx in self.specific:
            return self.specific[idx]
        return self.default
