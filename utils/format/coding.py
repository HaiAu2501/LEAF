from pydantic import BaseModel, Field

class Coding(BaseModel):
    code: str = Field(
        ...,
        title="Code",
        description="The Python code snippet.",
    )
    explanation: str = Field(
        ...,
        title="Explanation",
        description="The explanation of your method."
    )
