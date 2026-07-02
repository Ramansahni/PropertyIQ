import logging
from fastapi import APIRouter, Depends, HTTPException, status
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.dependencies import get_pipeline
from backend.services.prediction_service import predict_property_price

logger = logging.getLogger("backend")
router = APIRouter(tags=["Price Prediction"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Property Listing Price",
    description="Estimates the listing price of a property in Gurgaon based on sector, built-up area, BHKs, furnishing type, floor level, and luxury scoring. Returns predicted price in Crores along with +/-13% confidence limits."
)
async def predict(
    request: PredictionRequest,
    pipeline = Depends(get_pipeline)
):
    if pipeline is None:
        logger.error("Predictor pipeline model is not loaded in app state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Please ensure training artifacts exist."
        )
    try:
        response = predict_property_price(pipeline, request)
        return response
    except Exception as e:
        logger.exception("Prediction failed due to internal error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while calculating prediction: {str(e)}"
        )
