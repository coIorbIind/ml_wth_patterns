from fastapi import APIRouter

from ds_model_with_fastapi.api.routes.model import router as model_router


api_router = APIRouter()
api_router.include_router(model_router, prefix='/model', tags=['model'])
