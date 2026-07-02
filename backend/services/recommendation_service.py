import logging
from fastapi import HTTPException
from src.recommendation.recommender import recommend_by_facilities, recommend_by_price, recommend_by_location
from backend.schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationItem

logger = logging.getLogger("backend")

def recommend_similar_properties(recommender_data, request: RecommendationRequest) -> RecommendationResponse:
    """Delegates inputs mapping and calculations to similarity recommendation modules."""
    cosine_sim1, cosine_sim2, cosine_sim3, location_df_normalized = recommender_data
    
    # Check if property exists
    if request.property_name not in location_df_normalized.index:
        logger.warning(f"Property name '{request.property_name}' not found in location index.")
        raise HTTPException(status_code=404, detail=f"Property '{request.property_name}' not found in the index.")
        
    logger.info(f"Generating top_{request.top_n} recommendations for property='{request.property_name}' using type='{request.recommendation_type}'")
    
    rec_type = request.recommendation_type.lower()
    if rec_type == 'facilities':
        rec_df = recommend_by_facilities(request.property_name, cosine_sim1, location_df_normalized, request.top_n)
    elif rec_type == 'price':
        rec_df = recommend_by_price(request.property_name, cosine_sim2, location_df_normalized, request.top_n)
    elif rec_type == 'location':
        rec_df = recommend_by_location(request.property_name, cosine_sim3, location_df_normalized, request.top_n)
    else:
        logger.warning(f"Invalid recommendation type '{request.recommendation_type}' requested.")
        raise HTTPException(status_code=400, detail="Recommendation type must be 'facilities', 'price', or 'location'.")
        
    if rec_df.empty:
        return RecommendationResponse(
            reference_property=request.property_name,
            recommendation_type=request.recommendation_type,
            recommendations=[]
        )
        
    recommendations = []
    for _, row in rec_df.iterrows():
        recommendations.append(RecommendationItem(
            property_name=row['PropertyName'],
            similarity_score=round(float(row['SimilarityScore']), 4)
        ))
        
    return RecommendationResponse(
        reference_property=request.property_name,
        recommendation_type=request.recommendation_type,
        recommendations=recommendations
    )
