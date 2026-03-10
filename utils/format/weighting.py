from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

# 1) Feature-level prior
class FeaturePrior(BaseModel):
    name: str = Field(
        ...,
        description="Name of the feature"
    )
    weight: float = Field(
        ...,
        ge = 0.0, le = 1.0,
        description="Weight in [0,1]; higher = more prior mass for using this feature in trees"
    )

class FeatureWeights(BaseModel):
    weights: List[FeaturePrior] = Field(
        ...,
        description="List of feature weightings"
    )
    explanation: str = Field(
        ...,
        description="Short natural-language rationale for the weights"
    )

# 2) Interaction-level prior
class InteractionPrior(BaseModel):
    feature_a: str = Field(
        ...,
        description="Name of the parent feature"
    )
    feature_b: str = Field(
        ...,
        description="Name of the child feature"
    )
    is_symmetric: bool = Field(
        False,
        description="Whether the interaction is symmetric"
    )
    weight: float = Field(
        ...,
        ge = 0.0, le = 1.0,
        description="Weight in [0,1]; higher = more prior mass for using this interaction in trees"
    )

class InteractionWeights(BaseModel):
    weights: List[InteractionPrior] = Field(
        ...,
        description="List of pairwise feature interactions and their weightings"
    )
    explanation: str = Field(
        ...,
        description="Short natural-language rationale for the pairwise interactions"
    )
    