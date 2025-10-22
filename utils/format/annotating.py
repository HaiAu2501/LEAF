from pydantic import BaseModel, Field
from typing import Mapping

class FeatureDescription(BaseModel):
    name: str = Field(
        ...,
        description="Name of the feature"
    )
    description: str = Field(
        ...,
        description="Description of the feature"
    )

class Annotating(BaseModel):
    descriptions: list[FeatureDescription] = Field(
        ...,
        description="Descriptions of the data features"
    )