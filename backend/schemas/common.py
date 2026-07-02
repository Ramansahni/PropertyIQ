from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="Application health status", example="healthy")
