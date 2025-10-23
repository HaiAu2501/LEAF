from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

class FeatureWeighting(BaseModel):
    name: str = Field(
        ...,
        description="Name of the feature"
    )
    weight: float = Field(
        ...,
        description="A float value in [0, 1] representing the weight of the feature"
    )

class Weighting(BaseModel):
    weights: List[FeatureWeighting] = Field(
        ...,
        description="List of feature weightings"
    )


    