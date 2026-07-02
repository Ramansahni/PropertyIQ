import logging
from fastapi import APIRouter, Depends, HTTPException, status
from backend.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.dependencies import get_recommender_data
from backend.services.recommendation_service import recommend_similar_properties

logger = logging.getLogger("backend")
router = APIRouter(tags=["Recommendations"])

@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend Similar Properties",
    description="Queries the recommendation engine similarity matrices to retrieve the top N matching properties based on: facilities (TF-IDF advantages), price (BHK spreads), or location coordinates proximity."
)
async def recommend(
    request: RecommendationRequest,
    recommender_data = Depends(get_recommender_data)
):
    cosine_sim1, cosine_sim2, cosine_sim3, location_df_normalized = recommender_data
    if cosine_sim1 is None or location_df_normalized is None:
        logger.error("Recommender similarity data is not loaded in app state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine data is not loaded. Please ensure similarity matrices exist."
        )
    return recommend_similar_properties(recommender_data, request)
