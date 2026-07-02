import logging
from fastapi import APIRouter, HTTPException, status
from backend.schemas.analytics import AnalyticsResponse
from backend.services.analytics_service import get_market_analytics

logger = logging.getLogger("backend")
router = APIRouter(tags=["Market Analytics"])

@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Market Analytics Overview",
    description="Loads Gurgaon real estate listing counts, median prices in Crores, total sector coverage, median area sizes, and breakdown categories for furnishing and property types."
)
async def get_analytics():
    try:
        response = get_market_analytics()
        return response
    except Exception as e:
        logger.exception("Failed computing market analytics.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while loading analytics: {str(e)}"
        )
