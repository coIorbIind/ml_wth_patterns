from fastapi import FastAPI

from ds_model_with_fastapi.api.exceptions import BaseAPIException, exception_handler
from ds_model_with_fastapi.api.router import api_router


def get_app():
    app = FastAPI()
    app.include_router(api_router, prefix='/api/v1')
    app.exception_handler(BaseAPIException)(exception_handler)

    return app
