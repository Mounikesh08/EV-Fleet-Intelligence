from fastapi import FastAPI, HTTPException

from app.api.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    ExplainRiskRequest,
    ExplainRiskResponse,
    HealthResponse,
    FSDRiskRequest,
    FSDRiskResponse,
)

from app.api.predictor import (
    predict_anomaly,
    predict_forecast,
    predict_fsd_risk,
    models_loaded_status,
)

from app.rag.explainer import generate_risk_explanation


app = FastAPI(
    title="EV Fleet Intelligence Platform API",
    description=(
        "Production-ready API for EV battery intelligence, "
        "SOH forecasting, GenAI-style risk explanation, and FSD scenario intelligence."
    ),
    version="1.1.0",
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "EV Fleet Intelligence Platform API",
        "modules": ["Battery Intelligence", "FSD Scenario Intelligence"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    status = models_loaded_status()

    return {
        "status": "ok",
        "service": "EV Fleet Intelligence Platform",
        "anomaly_model_loaded": status["anomaly_model_loaded"],
        "forecast_model_loaded": status["forecast_model_loaded"],
        "fsd_risk_model_loaded": status["fsd_risk_model_loaded"],
    }


@app.post("/predict/anomaly", response_model=AnomalyResponse, tags=["Battery Intelligence"])
def anomaly_prediction(request: AnomalyRequest):
    try:
        result = predict_anomaly(request.telemetry.model_dump())

        return {
            "vehicle_id": request.vehicle_id,
            "anomaly_detected": result["anomaly_detected"],
            "anomaly_score": result["anomaly_score"],
            "risk_level": result["risk_level"],
            "message": result["message"],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/forecast", response_model=ForecastResponse, tags=["Battery Intelligence"])
def forecast_prediction(request: ForecastRequest):
    try:
        forecast = predict_forecast(request.telemetry.model_dump())

        return {
            "vehicle_id": request.vehicle_id,
            "soh_forecast": forecast,
            "message": "SOH forecast generated successfully.",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain/risk", response_model=ExplainRiskResponse, tags=["Battery Intelligence"])
def explain_risk(request: ExplainRiskRequest):
    try:
        record = request.telemetry.model_dump()

        anomaly_result = predict_anomaly(record)
        forecast_result = predict_forecast(record)

        explanation = generate_risk_explanation(
            vehicle_id=request.vehicle_id,
            record=record,
            anomaly_result=anomaly_result,
            forecast_result=forecast_result,
        )

        return explanation

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/fsd-risk", response_model=FSDRiskResponse, tags=["FSD Scenario Intelligence"])
def fsd_risk_prediction(request: FSDRiskRequest):
    try:
        result = predict_fsd_risk(request.scenario.model_dump())

        return {
            "vehicle_id": request.vehicle_id,
            "fsd_risk_detected": result["fsd_risk_detected"],
            "risk_probability": result["risk_probability"],
            "risk_level": result["risk_level"],
            "scenario_summary": result["scenario_summary"],
            "recommended_action": result["recommended_action"],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))