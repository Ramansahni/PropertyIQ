import logging
from src.inference.predict import predict_price
from backend.schemas.prediction import PredictionRequest, PredictionResponse

logger = logging.getLogger("backend")

def predict_property_price(pipeline, request: PredictionRequest) -> PredictionResponse:
    """Delegates inputs mapping and prediction calculations to the core ML inference module."""
    # Convert Pydantic request schema to a list matching the exact column order expected by pipeline
    input_data = [
        request.property_type,
        request.sector,
        request.bedRoom,
        request.bathroom,
        request.balcony,
        request.agePossession,
        request.built_up_area,
        request.servant_room,
        request.store_room,
        request.furnishing_type,
        request.luxury_category,
        request.floor_category
    ]
    
    logger.info(f"Generating price prediction for sector={request.sector}, area={request.built_up_area} sqft")
    
    # Delegate to src module
    pred_price = predict_price(pipeline, input_data)
    
    # Calculate bounds and per-sqft price matching UI logic
    lower_bound = pred_price * 0.87
    upper_bound = pred_price * 1.13
    price_per_sqft = (pred_price * 10000000) / request.built_up_area
    
    return PredictionResponse(
        estimated_price_cr=round(pred_price, 4),
        lower_bound_cr=round(lower_bound, 4),
        upper_bound_cr=round(upper_bound, 4),
        price_per_sqft=round(price_per_sqft, 2)
    )
