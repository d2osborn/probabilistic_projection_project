from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    horizon: int = Field(
        ..., gt=0, le=365, description="Number of days ahead to forecast"
    )
    credible_interval: float = Field(
        default=0.94, gt=0, lt=1, description="Width of the credible interval"
    )

class ForecastPoint(BaseModel):
    day: int
    mean: float
    lower: float
    upper: float

class ForecastResponse(BaseModel):
    horizon: int
    credible_interval: float
    forecasts: list[ForecastPoint]