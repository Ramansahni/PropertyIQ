from fastapi import APIRouter, status
from backend.schemas.common import HealthResponse

router = APIRouter(tags=["System Health"])

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Get System Health Status",
    description="Returns the current readiness and health check status of the FastAPI backend application."
)
async def health_check():
    return HealthResponse(status="healthy")
