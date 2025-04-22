from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class StateVectorWithInfo:
    state_vector: np.ndarray
    info: Dict[str, Any]

    def get_info_items(self) -> List[str]:
        return sorted(list(self.info.keys()), key=lambda x: (x != "name", x))

    def get_data(self) -> List[Any]:
        return [str(self.info[key]) for key in self.get_info_items()]
