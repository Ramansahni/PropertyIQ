from pydantic import BaseModel, Field
from typing import List

class RecommendationItem(BaseModel):
    property_name: str = Field(..., description="Name of the recommended property", example="M3M Golf Estate")
    similarity_score: float = Field(..., description="Match percentage score (0.0 to 1.0)", example=0.925)

class RecommendationRequest(BaseModel):
    property_name: str = Field(..., description="Name of reference property", example="DLF The Arbour")
    recommendation_type: str = Field(..., description="Similarity criterion ('facilities', 'price', 'location')", example="facilities")
    top_n: int = Field(5, description="Number of results to return (1 to 20)", example=5, ge=1, le=20)

class RecommendationResponse(BaseModel):
    reference_property: str = Field(..., description="Name of reference property queried", example="DLF The Arbour")
    recommendation_type: str = Field(..., description="Similarity criterion queried", example="facilities")
    recommendations: List[RecommendationItem] = Field(..., description="List of similarity matches sorted by highest score")
