from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    property_type: str = Field(..., description="Type of property ('flat' or 'house')", example="flat")
    sector: str = Field(..., description="Sector name in Gurgaon", example="Sector 63")
    bedRoom: float = Field(..., description="Number of bedrooms (1.0 to 6.0)", example=3.0, ge=1.0, le=6.0)
    bathroom: float = Field(..., description="Number of bathrooms (1.0 to 6.0)", example=3.0, ge=1.0, le=6.0)
    balcony: str = Field(..., description="Number of balconies ('0', '1', '2', '3', '3+')", example="3")
    agePossession: str = Field(..., description="Age profile of property", example="New Property")
    built_up_area: float = Field(..., description="Built up area in sqft (300.0 to 10000.0)", example=1800.0, ge=300.0, le=10000.0)
    servant_room: float = Field(..., description="Servant room flag (0.0 or 1.0)", example=0.0)
    store_room: float = Field(..., description="Store room flag (0.0 or 1.0)", example=0.0)
    furnishing_type: str = Field(..., description="Furnishing tier ('unfurnished', 'semifurnished', 'furnished')", example="semifurnished")
    luxury_category: str = Field(..., description="Luxury tier ('Low', 'Medium', 'High')", example="Medium")
    floor_category: str = Field(..., description="Floor category ('Low Floor', 'Mid Floor', 'High Floor')", example="Mid Floor")

class PredictionResponse(BaseModel):
    estimated_price_cr: float = Field(..., description="Estimated listing price in Crores", example=2.85)
    lower_bound_cr: float = Field(..., description="Lower bound listing price (-13% confidence limit)", example=2.48)
    upper_bound_cr: float = Field(..., description="Upper bound listing price (+13% confidence limit)", example=3.22)
    price_per_sqft: float = Field(..., description="Price per square foot", example=15833.33)
