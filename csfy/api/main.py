import uvicorn

from cornsnake import util_print
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from csfy import config
from . import predict_router
from . import version

description = """
REST API to predict label for given text, hosting a model previously trained via csfy.
"""

tags_metadata = [
    {
        "name": "admin",
        "description": "Admin endpoints for functions such as Health check",
    },
    {
        "name": "predictor",
        "description": "Label Prediction APIs"
    },
]

app = FastAPI(
    title="csfy",
    description=description,
    version=version.VERSION,
    root_path="/csfy/v1",
    root_path_in_servers=False,
    openapi_tags=tags_metadata,
)

def custom_openapi(fast_api):
    if not fast_api.openapi_schema:
        fast_api.openapi_schema = get_openapi(
            title=fast_api.title,
            version=fast_api.version,
            openapi_version=fast_api.openapi_version,
            description=fast_api.description,
            terms_of_service=fast_api.terms_of_service,
            contact=fast_api.contact,
            license_info=fast_api.license_info,
            routes=fast_api.routes,
            tags=fast_api.openapi_tags,
            servers=fast_api.servers,
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/csfy/admin/health", status_code=200, tags=["admin"])
async def admin_health() -> int:
    return status.HTTP_200_OK

app.include_router(predict_router.predict_router)
custom_openapi(app)

def start():
    util_print.print_result(f"REST API - Swagger is at http://{config.SERVE_HOST}:{config.SERVE_PORT}/docs")
    uvicorn.run(app, host=config.SERVE_HOST, port=config.SERVE_PORT)

if __name__ == "__main__":
    start()
