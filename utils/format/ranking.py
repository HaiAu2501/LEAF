from pydantic import BaseModel
from typing import List, Tuple

class Ranking(BaseModel):
    scores: List[int]
    