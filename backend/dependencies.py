from fastapi import Request

def get_pipeline(request: Request):
    """FastAPI dependency to retrieve the cached ML pipeline model from app state."""
    return request.app.state.pipeline

def get_df(request: Request):
    """FastAPI dependency to retrieve the metadata dataframe from app state."""
    return request.app.state.df

def get_recommender_data(request: Request):
    """FastAPI dependency to retrieve the similarity matrices and locations dataframe from app state."""
    return (
        request.app.state.cosine_sim1,
        request.app.state.cosine_sim2,
        request.app.state.cosine_sim3,
        request.app.state.location_df_normalized
    )
