from pydantic import BaseModel, Field

class Annotating(BaseModel):
    data_description: dict[str, str] = Field(..., description="Description of the data")