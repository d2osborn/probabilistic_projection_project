from contextlib import asynccontextmanager
from fastapi import FastAPI
from schemas import ForecastRequest, ForecastResponse, ForecastPoint
from model import BayesianForecaster

forecaster = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global forecaster
    forecaster = BayesianForecaster(
        trace_path="model_artifacts/trace.nc",
        metadata_path="model_artifacts/metadata.pkl",
    )
    yield

app = FastAPI(
    title="Bayesian Forecast API",
    description="Serves forecasts from a PyMC time series model",
    lifespan=lifespan,
)

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    results = forecaster.predict(
        horizon=request.horizon,
        credible_interval=request.credible_interval,
    )
    return ForecastResponse(
        horizon=request.horizon,
        credible_interval=request.credible_interval,
        forecasts=[ForecastPoint(**r) for r in results],
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": forecaster is not None}
