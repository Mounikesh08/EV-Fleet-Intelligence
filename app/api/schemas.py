from typing import Dict, List
from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    voltage: float = Field(..., example=4.12)
    current: float = Field(..., example=-2.1)
    temperature: float = Field(..., example=31.5)
    soc: float = Field(..., example=0.72)
    capacity: float = Field(..., example=96.8)
    cycle_count: int = Field(..., example=210)
    impedance: float = Field(..., example=0.038)
    soh: float = Field(..., example=0.968)


class AnomalyRequest(BaseModel):
    vehicle_id: str = Field(..., example="EV_005")
    telemetry: TelemetryRecord


class AnomalyResponse(BaseModel):
    vehicle_id: str
    anomaly_detected: bool
    anomaly_score: float
    risk_level: str
    message: str


class ForecastRequest(BaseModel):
    vehicle_id: str = Field(..., example="EV_005")
    telemetry: TelemetryRecord


class ForecastResponse(BaseModel):
    vehicle_id: str
    soh_forecast: Dict[str, float]
    message: str


class ExplainRiskRequest(BaseModel):
    vehicle_id: str = Field(..., example="EV_005")
    telemetry: TelemetryRecord


class ExplainRiskResponse(BaseModel):
    vehicle_id: str
    risk_level: str
    summary: str
    retrieved_context: List[str]
    forecast_summary: str
    recommended_action: str


class FSDScenarioRecord(BaseModel):
    speed: float = Field(..., example=52.0)
    lead_vehicle_distance: float = Field(..., example=7.5)
    relative_speed: float = Field(..., example=-19.0)
    lane_offset: float = Field(..., example=0.88)
    steering_angle: float = Field(..., example=10.5)
    brake_pressure: float = Field(..., example=0.76)
    object_count: int = Field(..., example=14)
    pedestrian_count: int = Field(..., example=3)
    autopilot_engaged: int = Field(..., example=1)
    traffic_light_state: str = Field(..., example="red")
    weather_condition: str = Field(..., example="rain")
    road_type: str = Field(..., example="urban")


class FSDRiskRequest(BaseModel):
    vehicle_id: str = Field(..., example="EV_021")
    scenario: FSDScenarioRecord


class FSDRiskResponse(BaseModel):
    vehicle_id: str
    fsd_risk_detected: bool
    risk_probability: float
    risk_level: str
    scenario_summary: str
    recommended_action: str


class HealthResponse(BaseModel):
    status: str
    service: str
    anomaly_model_loaded: bool
    forecast_model_loaded: bool
    fsd_risk_model_loaded: bool