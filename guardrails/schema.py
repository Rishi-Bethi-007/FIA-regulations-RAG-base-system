from pydantic import BaseModel, Field

class AnswerSchema(BaseModel):
    answer: str = Field(..., description="The final answer to the user.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0,1].")
    used_sources: list[str] = Field(default_factory=list, description="Which sources/chunks were used.")
