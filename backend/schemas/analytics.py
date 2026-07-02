from pydantic import BaseModel, Field
from typing import Dict

class AnalyticsResponse(BaseModel):
    total_properties: int = Field(..., description="Total property listings in database", example=3554)
    median_price_cr: float = Field(..., description="Median property price in Crores", example=1.51)
    sectors_count: int = Field(..., description="Total unique sectors covered", example=104)
    median_built_up_area_sqft: float = Field(..., description="Median built up area in square feet", example=1500.0)
    property_type_distribution: Dict[str, int] = Field(..., description="Breakdown counts by property type (flat vs house)", example={"flat": 2804, "house": 750})
    furnishing_type_distribution: Dict[str, int] = Field(..., description="Breakdown counts by furnishing status", example={"unfurnished": 1500, "semifurnished": 1200, "furnished": 854})
