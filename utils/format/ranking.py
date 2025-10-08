from pydantic import BaseModel
from typing import List, Tuple, Dict

class Ranking(BaseModel):
    scores: Dict[str, float]
    