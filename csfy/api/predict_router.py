from fastapi import APIRouter

from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from .response_codes import http_codes

from csfy.core import predictor

predict_router = APIRouter(prefix="/csfy")

class BaseApiModel(BaseModel):
    class Config:
        alias_generator = to_camel
        populate_by_name = True
        extra = "forbid"

class PredictRequest(BaseApiModel):
    text: str = Field(
        description="The text to classify", examples=["I am sitting on the beach"]
    )

class PredictResponse(BaseApiModel):
    label: str = Field(
        description="The predicted label",
        examples=["neutral"],
    )

@predict_router.post(
    "/predictor/v1/predict_label",
    response_model=PredictResponse,
    responses={**http_codes},
    tags=["predict"],
)
async def predict_label(request: PredictRequest) -> PredictResponse:
    """
    Predict a label for the given text
    """
    label = predictor.predict_label(request.text)
    return PredictResponse(label=label)
