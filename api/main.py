from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from inspector_dashboard import (
    assign_inspection_task,
    build_case_detail,
    complete_inspection_task,
    dashboard_payload,
    filter_predictions_for_area,
    filter_cases,
)
from inspector_manager import create_inspector, delete_inspector, get_all_inspectors
from login import SESSION_COOKIE_NAME, authenticate_user, create_session_token, decode_session_token, ensure_users_file, redirect_user
from src.alert_engine import dispatch_alerts, send_backend_status_message, send_inspector_welcome_message
from src.data_drift_monitor import generate_drift_report
from src.data_generator import generate_smart_meter_data
from src.demand_forecasting import forecast_horizons
from src.energy_efficiency import calculate_efficiency_metrics, summarise_efficiency
from src.explainable_ai import explain_prediction
from src.pole_monitoring import simulate_pole_energy
from src.pole_tamper_detector import detect_pole_tampering
from src.preprocess import (
    aggregate_region_consumption,
    aggregate_weather_impact,
    build_overview_snapshot,
    limit_theft_alerts,
    load_dataset,
    load_training_dataset,
    preprocess_frame,
)
from src.report_generator import generate_daily_report
from src.risk_scoring import risk_distribution_by_area, risk_payload, score_meter_risk
from src.spatial_analysis import build_theft_heatmap
from src.theft_detector import classify_meter_events
from src.federated.experiment import run_experiment
from src.train_models import train_all_models
from src.transformer_forecasting import forecast_transformer_horizons
from src.weather_api import WeatherService
from utils.helpers import dataframe_to_sqlite, ensure_project_dirs, generation_config, load_json, records_for_json, to_builtin
from utils.helpers import AREA_COORDINATES


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_local_env()


class MeterReading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meter_id: str = Field(..., examples=["M0512"])
    timestamp: str | None = None
    region: str = "Bengaluru"
    area: str
    latitude: float
    longitude: float
    voltage: float = Field(..., ge=150, le=300)
    current: float = Field(..., ge=0, le=10000)
    power: float = Field(..., ge=0, le=100000)
    consumption_kwh: float = Field(..., ge=0, le=100000)
    power_factor: float = Field(..., ge=0, le=1)
    temperature: float = Field(..., ge=-50, le=70)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=1000)
    wind_speed: float = Field(..., ge=0, le=200)
    weather_condition: str
    expected_consumption_kwh: float | None = None
    wastage_score: float | None = None
    usage_profile: str = "residential"


class ResearchExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(42, ge=0, le=2**31-1)
    clients: int = Field(8, ge=3, le=32)
    rounds: int = Field(5, ge=1, le=30)
    local_epochs: int = Field(2, ge=1, le=10)
    batch_size: int = Field(64, ge=8, le=512)
    learning_rate: float = Field(.12, gt=0, le=1)
    partition: str = Field("moderate", pattern="^(iid|moderate|severe)$")
    attack: str = Field("none", pattern="^(none|label_flip|scale|sign|slow)$")
    malicious_fraction: float = Field(.25, ge=0, le=.45)
    attack_intensity: float = Field(4.0, gt=0, le=20)


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class InspectorCreateRequest(BaseModel):
    name: str = Field(..., min_length=3)
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    assigned_area: str = Field(..., min_length=2)
    chat_id: str = Field(..., min_length=6)


class AssignInspectionRequest(BaseModel):
    meter_id: str
    inspection_date: str
    inspection_time: str


class CompleteInspectionRequest(BaseModel):
    remarks: str = ""


def _prepare_request_frame(payload: MeterReading | list[MeterReading]) -> pd.DataFrame:
    records = payload if isinstance(payload, list) else [payload]
    rows: list[dict[str, Any]] = []
    for record in records:
        data = record.model_dump()
        if not data.get("timestamp"):
            data["timestamp"] = pd.Timestamp.utcnow().tz_localize(None).floor("h")
        data["expected_consumption_kwh"] = data.get("expected_consumption_kwh") or data["consumption_kwh"]
        if data.get("wastage_score") is None:
            expected = max(float(data["expected_consumption_kwh"]), 0.1)
            data["wastage_score"] = max((float(data["consumption_kwh"]) - expected) / expected, 0.0)
        rows.append(data)
    return preprocess_frame(pd.DataFrame(rows))


def _empty_forecast() -> dict[str, Any]:
    return {
        "next_hour": 0.0,
        "next_day": 0.0,
        "next_week": 0.0,
        "series": [],
        "lstm": {"next_hour": 0.0, "next_day": 0.0, "next_week": 0.0, "series": [], "model_type": "baseline"},
        "transformer": {"next_hour": 0.0, "next_day": 0.0, "next_week": 0.0, "series": [], "model_type": "baseline"},
        "ensemble": {"next_hour": 0.0, "next_day": 0.0, "next_week": 0.0},
        "comparison_series": {"lstm": [], "transformer": []},
    }


def _flatten_drift_report(report: dict[str, Any]) -> pd.DataFrame:
    quality = report.get("data_quality", {})
    concept = report.get("concept_drift", {})
    return pd.DataFrame(
        [
            {
                "generated_at": report.get("generated_at"),
                "method": report.get("method"),
                "drift_detected": int(bool(report.get("drift_detected"))),
                "current_missing_pct": quality.get("current_missing_pct", 0.0),
                "reference_missing_pct": quality.get("reference_missing_pct", 0.0),
                "issue_count": len(quality.get("issues", [])),
                "theft_rate_shift": concept.get("theft_rate_shift", 0.0),
                "prediction_rate_shift": concept.get("prediction_rate_shift", 0.0),
                "concept_drift_detected": int(bool(concept.get("detected"))),
            }
        ]
    )


def _ensure_visible_theft_candidate(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if frame.empty or (frame.get("status", pd.Series(dtype=object)) == "Electricity Theft").any():
        return frame

    seeded = pd.to_numeric(frame.get("seeded_theft_probability", 0.0), errors="coerce").fillna(0.0)
    anomaly = pd.to_numeric(frame.get("anomaly_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    theft = pd.to_numeric(frame.get("theft_probability", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    wastage = pd.to_numeric(frame.get("wastage_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    candidate_score = (0.45 * seeded) + (0.3 * theft) + (0.2 * anomaly) + (0.05 * wastage.clip(upper=1.0))
    candidate_index = candidate_score.idxmax()
    frame.loc[candidate_index, "theft_probability"] = max(float(frame.loc[candidate_index, "theft_probability"]), 0.91)
    if "random_forest_probability" in frame.columns:
        frame.loc[candidate_index, "random_forest_probability"] = max(float(frame.loc[candidate_index, "random_forest_probability"]), 0.86)
    if "xgboost_probability" in frame.columns:
        frame.loc[candidate_index, "xgboost_probability"] = max(float(frame.loc[candidate_index, "xgboost_probability"]), 0.92)
    frame.loc[candidate_index, "status"] = "Electricity Theft"
    return frame


def _sticky_theft_sort_key(frame: pd.DataFrame, meter_id: str | None) -> pd.Series:
    if not meter_id or frame.empty or "meter_id" not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    return (frame["meter_id"].astype(str) == str(meter_id)).astype(int)


def _prioritize_sticky_meter(
    frame: pd.DataFrame,
    meter_id: str | None,
    sort_columns: list[str],
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    ordered = frame.copy()
    ordered["_sticky_theft_priority"] = _sticky_theft_sort_key(ordered, meter_id)
    sort_order = ["_sticky_theft_priority", *sort_columns]
    direction = [False, *(ascending if ascending is not None else [False] * len(sort_columns))]
    return ordered.sort_values(by=sort_order, ascending=direction).drop(columns="_sticky_theft_priority")


class SmartGridRuntime:
    def __init__(self) -> None:
        self.paths = ensure_project_dirs()
        self.weather_service = WeatherService()
        self.lock = Lock()
        self.update_interval = int(os.getenv("SMARTGRID_UPDATE_INTERVAL", "4"))
        self.telegram_status_interval = int(os.getenv("SMARTGRID_TELEGRAM_STATUS_INTERVAL", "60"))
        self.last_telegram_status_at: pd.Timestamp | None = None
        self.demo_mode = os.getenv("SMARTGRID_DEMO_MODE", "0") == "1"
        self.enable_periodic_reports = os.getenv(
            "SMARTGRID_ENABLE_PERIODIC_REPORTS",
            "0" if self.demo_mode else "1",
        ) == "1"
        self.simulation_source = pd.DataFrame()
        self.historical_frame = pd.DataFrame()
        self.latest_predictions = pd.DataFrame()
        self.prediction_buffer: deque[pd.DataFrame] = deque(maxlen=36)
        self.timeline: list[pd.Timestamp] = []
        self.cursor = 0
        self.current_timestamp: pd.Timestamp | None = None
        self.ws_clients: list[WebSocket] = []
        self.cached_forecast: dict[str, Any] = _empty_forecast()
        self.cached_drift_report: dict[str, Any] = {
            "generated_at": None,
            "method": "fallback",
            "drift_detected": False,
            "feature_drift": [],
            "concept_drift": {"detected": False},
            "data_quality": {"issues": []},
        }
        self.cached_alert_results: list[dict[str, Any]] = []
        self.sticky_theft_meter_id: str | None = None
        self.pole_catalog = pd.DataFrame()
        self.historical_pole_frame = pd.DataFrame()
        self.latest_pole_status = pd.DataFrame()
        self.recent_pole_status = pd.DataFrame()

    def _capture_sticky_theft_meter(self, predictions: pd.DataFrame) -> None:
        if self.sticky_theft_meter_id or predictions.empty:
            return
        theft_frame = predictions.loc[predictions.get("status", pd.Series(dtype=object)) == "Electricity Theft"].copy()
        if theft_frame.empty:
            return
        sort_columns = [column for column in ["theft_probability", "anomaly_score"] if column in theft_frame.columns]
        if sort_columns:
            theft_frame = theft_frame.sort_values(sort_columns, ascending=False)
        self.sticky_theft_meter_id = str(theft_frame.iloc[0]["meter_id"])

    def _apply_sticky_theft_meter(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not self.sticky_theft_meter_id or predictions.empty or "meter_id" not in predictions.columns:
            return predictions

        frame = predictions.copy()
        sticky_mask = frame["meter_id"].astype(str) == str(self.sticky_theft_meter_id)
        if not sticky_mask.any():
            return frame

        sticky_index = frame.index[sticky_mask][0]
        frame.loc[sticky_index, "status"] = "Electricity Theft"
        frame.loc[sticky_index, "theft_probability"] = max(float(frame.loc[sticky_index].get("theft_probability", 0.0)), 0.91)
        if "random_forest_probability" in frame.columns:
            frame.loc[sticky_index, "random_forest_probability"] = max(float(frame.loc[sticky_index, "random_forest_probability"]), 0.86)
        if "xgboost_probability" in frame.columns:
            frame.loc[sticky_index, "xgboost_probability"] = max(float(frame.loc[sticky_index, "xgboost_probability"]), 0.92)
        return frame

    def _build_forecast_payload(self) -> dict[str, Any]:
        lstm_forecast = forecast_horizons(metadata_path=self.paths.demand_metadata, model_path=self.paths.lstm_model)
        transformer_forecast = forecast_transformer_horizons(
            metadata_path=self.paths.transformer_metadata,
            model_path=self.paths.transformer_model,
        )
        return {
            **lstm_forecast,
            "lstm": lstm_forecast,
            "transformer": transformer_forecast,
            "ensemble": {
                "next_hour": round((lstm_forecast.get("next_hour", 0.0) + transformer_forecast.get("next_hour", 0.0)) / 2.0, 2),
                "next_day": round((lstm_forecast.get("next_day", 0.0) + transformer_forecast.get("next_day", 0.0)) / 2.0, 2),
                "next_week": round((lstm_forecast.get("next_week", 0.0) + transformer_forecast.get("next_week", 0.0)) / 2.0, 2),
            },
            "comparison_series": {
                "lstm": lstm_forecast.get("series", []),
                "transformer": transformer_forecast.get("series", []),
            },
        }

    def bootstrap(self) -> None:
        full_scale = os.getenv("SMARTGRID_FULL_SCALE", "0") == "1"
        config = generation_config(full_scale=full_scale)
        sample_path = self.paths.data_processed / "smart_meter_sample.csv"
        needs_regeneration = (
            not self.paths.dataset.exists()
            or not self.paths.live_dataset.exists()
            or not sample_path.exists()
            or not self.paths.meter_catalog.exists()
            or not self.paths.pole_catalog.exists()
        )
        if not needs_regeneration:
            with suppress(Exception):
                meter_catalog = pd.read_csv(self.paths.meter_catalog, usecols=["meter_id"])
                live_catalog = pd.read_csv(self.paths.live_dataset, usecols=["meter_id"])
                generation_summary = load_json(self.paths.data_processed / "generation_summary.json", default={}) or {}
                needs_regeneration = (
                    int(meter_catalog["meter_id"].nunique()) != int(config["num_meters"])
                    or int(live_catalog["meter_id"].nunique()) != int(config["simulation_meter_limit"])
                    or not generation_summary.get("live_theft_meter_ids")
                )

        if needs_regeneration:
            generate_smart_meter_data(**config)

        model_schema = load_json(self.paths.model_metadata, default={}) or {}
        stale_features = model_schema.get("feature_schema_version") != 2
        if stale_features or not all(
            [
                self.paths.isolation_forest.exists(),
                self.paths.random_forest.exists(),
                self.paths.xgboost_model.exists(),
                self.paths.model_metadata.exists(),
                self.paths.demand_metadata.exists(),
                self.paths.transformer_metadata.exists(),
            ]
        ):
            train_all_models(dataset_path=sample_path, max_rows=config["sample_rows"], forecast_epochs=4)

        self.historical_frame = load_training_dataset(max_rows=config["sample_rows"])
        if self.historical_frame.empty:
            self.historical_frame = load_dataset(self.paths.dataset, nrows=20000)
        dataframe_to_sqlite(self.historical_frame, "meter_readings")
        self.pole_catalog = pd.read_csv(self.paths.pole_catalog) if self.paths.pole_catalog.exists() else pd.DataFrame()
        self.historical_pole_frame = simulate_pole_energy(self.historical_frame, pole_catalog=self.pole_catalog)
        if not self.historical_pole_frame.empty:
            dataframe_to_sqlite(self.historical_pole_frame, "pole_energy_data")

        self.simulation_source = load_dataset(self.paths.live_dataset)
        if self.simulation_source.empty:
            self.simulation_source = self.historical_frame.sort_values("timestamp").reset_index(drop=True)

        self.timeline = sorted(self.simulation_source["timestamp"].drop_duplicates().tolist())
        self.cursor = 0
        self.prediction_buffer.clear()
        self.sticky_theft_meter_id = None
        self.latest_pole_status = pd.DataFrame()
        self.recent_pole_status = pd.DataFrame()
        self.cached_forecast = self._build_forecast_payload()
        self.advance_tick()

    def recent_predictions(self) -> pd.DataFrame:
        if not self.prediction_buffer:
            return self.latest_predictions.copy()
        return pd.concat(list(self.prediction_buffer), ignore_index=True)

    def advance_tick(self) -> None:
        if not self.timeline:
            return

        timestamp = self.timeline[self.cursor]
        current_frame = self.simulation_source.loc[self.simulation_source["timestamp"] == timestamp].copy()
        predictions = classify_meter_events(current_frame)
        if self.demo_mode:
            predictions = _ensure_visible_theft_candidate(predictions)
            self._capture_sticky_theft_meter(predictions)
            predictions = self._apply_sticky_theft_meter(predictions)
            predictions = limit_theft_alerts(predictions, max_alerts=2, preferred_meter_id=self.sticky_theft_meter_id)
            predictions = _prioritize_sticky_meter(
                predictions,
                self.sticky_theft_meter_id,
                sort_columns=["theft_probability", "anomaly_score"],
            )
        predictions = calculate_efficiency_metrics(score_meter_risk(predictions)).reset_index(drop=True)

        previous_recent = self.recent_predictions()
        recent = pd.concat([previous_recent, predictions], ignore_index=True) if not previous_recent.empty else predictions.copy()
        recent = recent.tail(5000).reset_index(drop=True)
        pole_status = detect_pole_tampering(
            current_frame=simulate_pole_energy(predictions, pole_catalog=self.pole_catalog),
            historical_frame=self.historical_pole_frame if not self.historical_pole_frame.empty else simulate_pole_energy(recent, pole_catalog=self.pole_catalog),
        )
        previous_poles = self.recent_pole_status.copy()
        recent_poles = pd.concat([previous_poles, pole_status], ignore_index=True) if not previous_poles.empty else pole_status.copy()
        recent_poles = recent_poles.tail(4000).reset_index(drop=True)

        forecast_payload = self._build_forecast_payload()

        reference_frame = self.historical_frame.sort_values("timestamp").tail(3500)
        drift_report = generate_drift_report(reference_frame=reference_frame, current_frame=recent.tail(1200))

        alert_results: list[dict[str, Any]] = []
        if os.getenv("SMARTGRID_ENABLE_ALERTS", "0") == "1":
            pole_alerts = pole_status.loc[pole_status["tamper_flag"] == 1, ["pole_id", "area", "tamper_probability"]].copy()
            theft_alerts = predictions.loc[predictions["status"] == "Electricity Theft"].copy()
            alert_frame = pd.concat(
                [
                    theft_alerts,
                    pole_alerts,
                ],
                ignore_index=True,
                sort=False,
            )
            alert_results = dispatch_alerts(alert_frame, limit=None)

        with self.lock:
            self.current_timestamp = pd.Timestamp(timestamp)
            self.latest_predictions = predictions
            self.prediction_buffer.append(predictions.copy())
            self.cached_forecast = forecast_payload
            self.cached_drift_report = drift_report
            self.cached_alert_results = alert_results
            self.latest_pole_status = pole_status
            self.recent_pole_status = recent_poles

            dataframe_to_sqlite(self.latest_predictions, "live_predictions")
            if not self.latest_predictions.empty:
                dataframe_to_sqlite(self.latest_predictions, "live_prediction_events", if_exists="append")
            dataframe_to_sqlite(recent, "recent_predictions")
            dataframe_to_sqlite(self.latest_predictions, "risk_scores")
            dataframe_to_sqlite(self.latest_predictions, "efficiency_metrics")
            dataframe_to_sqlite(_flatten_drift_report(self.cached_drift_report), "drift_reports")
            dataframe_to_sqlite(recent_poles, "pole_energy_data")
            dataframe_to_sqlite(
                pole_status.loc[pole_status["tamper_flag"] == 1].reset_index(drop=True),
                "pole_tamper_events",
            )

            forecast_frames: list[pd.DataFrame] = []
            for model_name in ["lstm", "transformer"]:
                model_series = pd.DataFrame(self.cached_forecast.get(model_name, {}).get("series", []))
                if not model_series.empty:
                    model_series["model_name"] = model_name
                    forecast_frames.append(model_series)
            if forecast_frames:
                dataframe_to_sqlite(pd.concat(forecast_frames, ignore_index=True), "forecast_snapshots")
            if self.cursor % 3 == 0 and {"latitude", "longitude", "theft_probability"}.issubset(self.latest_predictions.columns):
                build_theft_heatmap(self.latest_predictions)
            report_columns = {"timestamp", "meter_id", "area", "consumption_kwh", "temperature"}
            if self.enable_periodic_reports and self.cursor % 6 == 0 and report_columns.issubset(recent.columns):
                generate_daily_report(recent, forecast=self.cached_forecast)

        self.cursor = (self.cursor + 1) % len(self.timeline)

    def _telegram_status_details(self) -> list[str]:
        with self.lock:
            latest = self.latest_predictions.copy()
            current_timestamp = self.current_timestamp
        theft_count = int((latest.get("status", pd.Series(dtype=object)) == "Electricity Theft").sum()) if not latest.empty else 0
        anomaly_count = int((latest.get("is_anomaly", pd.Series(dtype=int)) == 1).sum()) if not latest.empty else 0
        return [
            f"Current tick: {current_timestamp.isoformat() if current_timestamp is not None else 'not ready'}",
            f"Active meters: {int(latest['meter_id'].nunique()) if not latest.empty and 'meter_id' in latest.columns else 0}",
            f"Theft alerts: {theft_count}",
            f"Anomalies: {anomaly_count}",
        ]

    def send_telegram_status(self, title: str = "Smart Grid backend running", *, force: bool = False) -> dict[str, Any]:
        if os.getenv("SMARTGRID_ENABLE_ALERTS", "0") != "1":
            return {"provider": "telegram", "status": "skipped", "detail": "SMARTGRID_ENABLE_ALERTS is disabled."}

        now = pd.Timestamp.utcnow().tz_localize(None)
        if not force and self.last_telegram_status_at is not None:
            elapsed = (now - self.last_telegram_status_at).total_seconds()
            if elapsed < max(self.telegram_status_interval, 1):
                return {"provider": "telegram", "status": "throttled", "seconds_until_next": round(max(self.telegram_status_interval - elapsed, 0), 1)}

        try:
            result = send_backend_status_message(title, self._telegram_status_details())
        except Exception as error:
            result = {"provider": "telegram", "status": "error", "detail": str(error)}

        if result.get("status") == "sent":
            self.last_telegram_status_at = now
        print(f"Telegram backend status notification: {result}", flush=True)
        return result

    async def simulation_loop(self) -> None:
        while True:
            await asyncio.sleep(self.update_interval)
            await asyncio.to_thread(self.advance_tick)
            await asyncio.to_thread(self.send_telegram_status)
            await self.broadcast_snapshot()

    async def register_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self.lock:
            self.ws_clients.append(websocket)
        try:
            await websocket.send_json(self.snapshot_message())
        except WebSocketDisconnect:
            self.unregister_client(websocket)
            raise

    def unregister_client(self, websocket: WebSocket) -> None:
        with self.lock:
            self.ws_clients = [client for client in self.ws_clients if client is not websocket]

    def snapshot_message(self) -> dict[str, Any]:
        return to_builtin({
            "type": "live_tick",
            "overview": self.overview_payload(),
            "theft": self.theft_payload(limit=8),
            "weather": self.weather_payload(),
            "meters": self.meter_payload(limit=16),
            "risk": self.risk_scores_payload(limit=12),
            "efficiency": self.efficiency_payload(limit=10),
            "drift": self.drift_payload(),
            "forecast": self.forecast_payload(),
            "pole": self.pole_status_payload(limit=12),
        })

    async def broadcast_snapshot(self) -> None:
        payload = self.snapshot_message()
        disconnected: list[WebSocket] = []
        for client in list(self.ws_clients):
            try:
                await client.send_json(payload)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.unregister_client(client)

    def overview_payload(self) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_predictions.copy()
            recent = self.recent_predictions().copy()
            forecast = dict(self.cached_forecast)
            drift = dict(self.cached_drift_report)

        summary = build_overview_snapshot(latest)
        if not self.historical_frame.empty:
            summary["total_meters"] = int(self.historical_frame["meter_id"].nunique())
        if recent.empty:
            live_consumption = pd.DataFrame(columns=["timestamp", "total_consumption", "anomalies", "theft"])
        else:
            recent = recent.copy()
            if "consumption_kwh" not in recent.columns:
                recent["consumption_kwh"] = 0.0
            if "is_anomaly" not in recent.columns:
                recent["is_anomaly"] = 0
            if "status" not in recent.columns:
                recent["status"] = "Normal"
            live_consumption = (
                recent.groupby("timestamp", as_index=False)
                .agg(
                    total_consumption=("consumption_kwh", "sum"),
                    anomalies=("is_anomaly", "sum"),
                    theft=("status", lambda values: int((values == "Electricity Theft").sum())),
                )
                .sort_values("timestamp")
                .tail(24)
            )
        region_consumption = aggregate_region_consumption(latest)
        risk_distribution = risk_distribution_by_area(latest)

        return {
            "timestamp": self.current_timestamp.isoformat() if self.current_timestamp is not None else None,
            "summary": summary,
            "live_consumption": records_for_json(live_consumption),
            "region_consumption": records_for_json(region_consumption),
            "risk_distribution": records_for_json(risk_distribution),
            "forecast": forecast,
            "drift_detected": bool(drift.get("drift_detected", False)),
            "alert_results": self.cached_alert_results,
        }

    def meter_payload(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            latest = self.latest_predictions.copy()
        preferred_columns = [
            "meter_id",
            "timestamp",
            "area",
            "latitude",
            "longitude",
            "consumption_kwh",
            "power",
            "voltage",
            "power_factor",
            "anomaly_score",
            "theft_probability",
            "risk_score",
            "risk_level",
            "efficiency_score",
            "status",
        ]
        columns = [column for column in preferred_columns if column in latest.columns]
        return records_for_json(latest[columns].head(limit))

    def anomaly_payload(self, limit: int = 25) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_predictions.copy()
        anomaly_frame = latest.loc[(latest["is_anomaly"] == 1) | (latest["status"] == "Anomaly")].copy()
        anomalies = (
            anomaly_frame
            .sort_values(["anomaly_score", "risk_score"], ascending=False)
            .head(limit)
        )
        return {
            "records": records_for_json(anomalies),
            "count": int(len(anomaly_frame)),
            "summary": {
                "count": int(len(anomaly_frame)),
                "average_score": round(float(anomaly_frame["anomaly_score"].mean()), 3) if not anomaly_frame.empty else 0.0,
                "highest_score": round(float(anomaly_frame["anomaly_score"].max()), 3) if not anomaly_frame.empty else 0.0,
                "impacted_areas": int(anomaly_frame["area"].dropna().nunique()) if "area" in anomaly_frame.columns else 0,
            },
        }

    def theft_payload(self, limit: int = 20) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_predictions.copy()
        theft_frame = latest.loc[latest["status"] == "Electricity Theft"].copy()
        theft_records = _prioritize_sticky_meter(
            theft_frame,
            self.sticky_theft_meter_id,
            sort_columns=["risk_score", "theft_probability"],
        ).head(limit).copy()
        reasons: list[str] = []
        for _, row in theft_records.iterrows():
            explanation = explain_prediction(pd.DataFrame([row]))
            reasons.append(explanation.get("summary", ""))
        theft_records["reason"] = reasons

        return {
            "records": records_for_json(theft_records),
            "count": int(len(theft_frame)),
            "summary": {
                "count": int(len(theft_frame)),
                "average_risk_score": round(float(theft_frame["risk_score"].mean()), 2) if not theft_frame.empty else 0.0,
                "average_theft_probability": round(float(theft_frame["theft_probability"].mean()), 4) if not theft_frame.empty else 0.0,
                "critical_areas": int(theft_frame.loc[theft_frame["risk_score"] >= 80, "area"].dropna().nunique())
                if not theft_frame.empty and "area" in theft_frame.columns
                else 0,
            },
            "heatmap_path": "./theft_heatmap.html",
        }

    def weather_payload(self) -> dict[str, Any]:
        with self.lock:
            recent = self.recent_predictions().copy()
        combined = pd.concat([self.historical_frame.tail(4000), recent], ignore_index=True)
        impact = aggregate_weather_impact(combined)
        scatter = combined[["temperature", "consumption_kwh", "wastage_score", "area"]].tail(240)
        return {
            "bands": records_for_json(impact),
            "scatter": records_for_json(scatter),
            "live_weather": self.weather_service.current_area_weather(),
        }

    def forecast_payload(self) -> dict[str, Any]:
        return self.cached_forecast

    def risk_scores_payload(self, limit: int = 25) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_predictions.copy()
        return risk_payload(latest, limit=limit)

    def efficiency_payload(self, limit: int = 20) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_predictions.copy()
        return summarise_efficiency(latest, limit=limit)

    def pole_status_payload(self, limit: int = 20) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_pole_status.copy()
            recent = self.recent_pole_status.copy()

        if latest.empty:
            return {"summary": {"pole_count": 0, "suspicious_poles": 0, "average_gap": 0.0, "max_tamper_probability": 0.0}, "records": [], "timeline": []}
        suspicious = latest.loc[latest.get("tamper_flag", pd.Series(dtype=int)) == 1].copy()
        pole_records = latest.sort_values(["tamper_flag", "tamper_probability", "energy_gap"], ascending=[False, False, False]).head(limit)
        return {
            "summary": {
                "pole_count": int(latest["pole_id"].nunique()) if not latest.empty and "pole_id" in latest.columns else 0,
                "suspicious_poles": int(suspicious["pole_id"].nunique()) if not suspicious.empty else 0,
                "average_gap": round(float(latest["energy_gap"].mean()), 3) if not latest.empty else 0.0,
                "max_tamper_probability": round(float(latest["tamper_probability"].max()), 4) if not latest.empty else 0.0,
            },
            "records": records_for_json(pole_records),
            "timeline": records_for_json(recent.sort_values("timestamp").tail(120)),
        }

    def pole_tamper_alerts_payload(self, limit: int = 20) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_pole_status.copy()
        if latest.empty:
            return {"count": 0, "records": []}
        alerts = latest.loc[latest.get("tamper_flag", pd.Series(dtype=int)) == 1].copy()
        alerts = alerts.sort_values(["tamper_probability", "energy_gap"], ascending=False).head(limit).copy()
        if not alerts.empty:
            alerts["message"] = alerts.apply(
                lambda row: f"Pole {row['pole_id']} energy mismatch detected. Possible illegal connection.",
                axis=1,
            )
        return {
            "count": int(len(latest.loc[latest.get("tamper_flag", pd.Series(dtype=int)) == 1])) if not latest.empty else 0,
            "records": records_for_json(alerts),
        }

    def pole_energy_balance_payload(self, limit: int = 120) -> dict[str, Any]:
        with self.lock:
            recent = self.recent_pole_status.copy()
        if recent.empty:
            return {"records": [], "heatmap": []}
        balance = recent.sort_values("timestamp").tail(limit).copy()
        heatmap = (
            balance.groupby(["pole_id", "area"], as_index=False)
            .agg(
                avg_gap=("energy_gap", "mean"),
                avg_probability=("tamper_probability", "mean"),
                alert_count=("tamper_flag", "sum"),
            )
            .sort_values(["avg_probability", "avg_gap"], ascending=False)
            .reset_index(drop=True)
        )
        return {
            "records": records_for_json(balance),
            "heatmap": records_for_json(heatmap),
        }

    def drift_payload(self) -> dict[str, Any]:
        return to_builtin(self.cached_drift_report)

    def health_payload(self) -> dict[str, Any]:
        artifacts = {
            "dataset": self.paths.dataset.exists(),
            "live_dataset": self.paths.live_dataset.exists(),
            "meter_catalog": self.paths.meter_catalog.exists(),
            "pole_catalog": self.paths.pole_catalog.exists(),
            "isolation_forest": self.paths.isolation_forest.exists(),
            "random_forest": self.paths.random_forest.exists(),
            "boost_model": self.paths.xgboost_model.exists(),
            "model_metadata": self.paths.model_metadata.exists(),
            "forecast_metadata": self.paths.demand_metadata.exists(),
            "transformer_metadata": self.paths.transformer_metadata.exists(),
            "heatmap": self.paths.map_path.exists(),
            "daily_report": self.paths.daily_report.exists(),
            "drift_report": self.paths.drift_report.exists(),
        }
        healthy = all(
            [
                artifacts["live_dataset"],
                artifacts["isolation_forest"],
                artifacts["random_forest"],
                artifacts["boost_model"],
                artifacts["model_metadata"],
            ]
        )
        return {
            "status": "ok" if healthy else "degraded",
            "timestamp": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            "current_tick": self.current_timestamp.isoformat() if self.current_timestamp is not None else None,
            "websocket_clients": len(self.ws_clients),
            "demo_mode": self.demo_mode,
            "periodic_reports_enabled": self.enable_periodic_reports,
            "data_provenance": {
                "default_runtime_dataset": "synthetic demonstration data",
                "ceew_theft_labels_verified": False,
                "theft_score_is_proof": False,
                "notice": "Anomaly and risk scores are investigation aids, not confirmation of theft."
            },
            "artifacts": artifacts,
        }

    def predict_payload(self, payload: MeterReading | list[MeterReading]) -> list[dict[str, Any]]:
        request_frame = _prepare_request_frame(payload)
        predictions = calculate_efficiency_metrics(score_meter_risk(classify_meter_events(request_frame)))
        responses: list[dict[str, Any]] = []
        for _, row in predictions.iterrows():
            explanation = explain_prediction(pd.DataFrame([row]))
            responses.append(
                {
                    "meter_id": row["meter_id"],
                    "region": row["region"],
                    "area": row["area"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "status": row["status"],
                    "theft_probability": round(float(row["theft_probability"]), 4),
                    "anomaly_score": round(float(row["anomaly_score"]), 4),
                    "risk_score": round(float(row["risk_score"]), 2),
                    "risk_level": row["risk_level"],
                    "efficiency_score": round(float(row["efficiency_score"]), 2),
                    "reason": explanation.get("reason", []),
                }
            )
        return responses


runtime = SmartGridRuntime()


def _current_user(request: Request) -> dict[str, Any] | None:
    return decode_session_token(request.cookies.get(SESSION_COOKIE_NAME))


def _require_user(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _require_role(request: Request, role: str) -> dict[str, Any]:
    user = _require_user(request)
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail=f"{role.title()} access required.")
    return user


def _runtime_predictions() -> pd.DataFrame:
    return runtime.latest_predictions.copy() if not runtime.latest_predictions.empty else pd.DataFrame()


def _available_inspection_areas() -> list[str]:
    sources: list[pd.Series] = []
    if not runtime.latest_predictions.empty and "area" in runtime.latest_predictions.columns:
        sources.append(runtime.latest_predictions["area"])
    if not runtime.historical_frame.empty and "area" in runtime.historical_frame.columns:
        sources.append(runtime.historical_frame["area"])
    if sources:
        values = pd.concat(sources, ignore_index=True).dropna().astype(str).str.strip()
        unique_values = sorted({value for value in values if value})
        if unique_values:
            return unique_values
    return sorted(AREA_COORDINATES.keys())


def _inspector_payload_for_request(
    request: Request,
    *,
    detection_class: str | None = None,
    risk_level: str | None = None,
    location: str | None = None,
    status: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    user = _require_role(request, "inspector")
    assigned_area = str(user.get("assigned_area") or "").strip() or None
    scoped_predictions = _runtime_predictions()
    scoped_history = runtime.historical_frame.copy()
    pole_tamper_records = runtime.pole_tamper_alerts_payload(limit=500).get("records", [])
    payload = dashboard_payload(
        latest_predictions=scoped_predictions,
        historical_frame=scoped_history,
        inspector_username=str(user.get("username") or ""),
        pole_tamper_records=pole_tamper_records,
        assigned_area=assigned_area,
    )
    filtered_cases = filter_cases(
        payload.get("cases", []),
        detection_class=detection_class,
        risk_level=risk_level,
        location=assigned_area or location,
        status=status,
        date=date,
    )
    payload["cases"] = filtered_cases
    if filtered_cases:
        payload["detail_preview"] = build_case_detail(
            filtered_cases[0]["meter_id"],
            scoped_predictions,
            scoped_history,
        )
    else:
        payload["detail_preview"] = None
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.bootstrap()
    await asyncio.to_thread(runtime.send_telegram_status, "Smart Grid backend started", force=True)
    task = asyncio.create_task(runtime.simulation_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Smart Grid Electricity Theft Detection API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("SMARTGRID_CORS_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:8000").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/dashboard", StaticFiles(directory=runtime.paths.dashboard_dir, html=True), name="dashboard")

# Lightweight single-process browser security controls. For multi-worker production,
# move rate-limit counters to a shared store (e.g. Redis) and terminate TLS at the proxy.
_SECURITY_ALLOWED_ORIGINS = {origin.strip().rstrip("/") for origin in os.getenv(
    "SMARTGRID_CORS_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:8000,http://localhost:8000"
).split(",") if origin.strip()}
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_ATTEMPTS_LOCK = Lock()


@app.middleware("http")
async def browser_security_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin:
        # Reject cross-origin browser mutations. Requests without Origin (CLI/test clients)
        # remain usable; deployments should additionally enforce network-level controls.
        if origin.rstrip("/") not in _SECURITY_ALLOWED_ORIGINS:
            return JSONResponse({"detail": "Cross-origin state-changing request rejected."}, status_code=403)
    if request.url.path == "/auth/login" and request.method == "POST":
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _LOGIN_ATTEMPTS_LOCK:
            recent = [stamp for stamp in _LOGIN_ATTEMPTS.get(client_ip, []) if now - stamp < 60.0]
            if len(recent) >= 10:
                _LOGIN_ATTEMPTS[client_ip] = recent
                return JSONResponse({"detail": "Too many login attempts. Try again in one minute."}, status_code=429,
                                    headers={"Retry-After": "60"})
            recent.append(now)
            _LOGIN_ATTEMPTS[client_ip] = recent
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'self'")
    return response


@app.get("/", include_in_schema=False)
def read_root(request: Request) -> RedirectResponse:
    user = _current_user(request)
    if user is None:
        ensure_users_file()
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url=redirect_user(user), status_code=303)


@app.get("/login", include_in_schema=False)
def login_page() -> RedirectResponse:
    ensure_users_file()
    return RedirectResponse(url="/dashboard/login.html", status_code=303)


@app.get("/admin", include_in_schema=False)
def admin_dashboard_entry(request: Request) -> RedirectResponse:
    user = _current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    if user.get("role") != "admin":
        return RedirectResponse(url=redirect_user(user), status_code=303)
    return RedirectResponse(url="/dashboard/index.html", status_code=303)


@app.get("/inspector", include_in_schema=False)
def inspector_dashboard_entry(request: Request) -> RedirectResponse:
    user = _current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    if user.get("role") != "inspector":
        return RedirectResponse(url=redirect_user(user), status_code=303)
    return RedirectResponse(url="/dashboard/inspector/index.html", status_code=303)


@app.post("/auth/login")
def login_user(payload: LoginRequest, response: Response) -> dict[str, Any]:
    ensure_users_file()
    user = authenticate_user(payload.identifier, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email/username or password.")
    session_token = create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=os.getenv("SMARTGRID_COOKIE_SECURE", "1") == "1",
        samesite="lax",
        max_age=12 * 60 * 60,
        path="/",
    )
    return {
        "authenticated": True,
        "role": user.get("role"),
        "name": user.get("name"),
        "assigned_area": user.get("assigned_area"),
        "redirect_to": redirect_user(user),
    }


@app.post("/auth/logout")
def logout_user() -> JSONResponse:
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/auth/session")
def session_status(request: Request) -> dict[str, Any]:
    user = _require_user(request)
    return {
        "authenticated": True,
        "username": user.get("username"),
        "name": user.get("name"),
        "role": user.get("role"),
        "assigned_area": user.get("assigned_area"),
        "expires_at": user.get("expires_at"),
    }


@app.get("/api/inspectors")
def list_inspectors(request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    return {"inspectors": get_all_inspectors()}


@app.get("/api/inspection-areas")
def list_inspection_areas(request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    return {"areas": _available_inspection_areas()}


@app.post("/api/inspectors")
def add_inspector(payload: InspectorCreateRequest, request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    try:
        inspector = create_inspector(payload.name, payload.username, payload.password, payload.assigned_area, payload.chat_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        notification = send_inspector_welcome_message(inspector)
    except Exception as error:
        notification = {"provider": "telegram", "status": "error", "detail": str(error)}
    return {"inspector": inspector, "notification": notification}


@app.delete("/api/inspectors/{username}")
def remove_inspector(username: str, request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    deleted = delete_inspector(username)
    if not deleted:
        raise HTTPException(status_code=404, detail="Inspector not found.")
    return {"deleted": True, "username": username}


@app.get("/api/inspector/dashboard")
def get_inspector_dashboard(
    request: Request,
    detection_class: str | None = None,
    risk_level: str | None = None,
    location: str | None = None,
    status: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    return _inspector_payload_for_request(
        request,
        detection_class=detection_class,
        risk_level=risk_level,
        location=location,
        status=status,
        date=date,
    )


@app.get("/api/inspector/cases/{meter_id}")
def get_inspector_case_detail(meter_id: str, request: Request) -> dict[str, Any]:
    user = _require_role(request, "inspector")
    assigned_area = str(user.get("assigned_area") or "").strip() or None
    return build_case_detail(
        meter_id,
        filter_predictions_for_area(_runtime_predictions(), assigned_area),
        filter_predictions_for_area(runtime.historical_frame.copy(), assigned_area),
    )


@app.post("/api/inspector/tasks/assign")
def assign_case(payload: AssignInspectionRequest, request: Request) -> dict[str, Any]:
    user = _require_role(request, "inspector")
    try:
        task = assign_inspection_task(
            meter_id=payload.meter_id,
            inspection_date=payload.inspection_date,
            inspection_time=payload.inspection_time,
            inspector_username=str(user.get("username") or ""),
            predictions=_runtime_predictions(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"task": task}


@app.post("/api/inspector/tasks/{task_id}/complete")
def complete_case(task_id: str, payload: CompleteInspectionRequest, request: Request) -> dict[str, Any]:
    user = _require_role(request, "inspector")
    try:
        task = complete_inspection_task(
            task_id=task_id,
            inspector_username=str(user.get("username") or ""),
            remarks=payload.remarks,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"task": task}


@app.get("/health")
def get_health() -> dict[str, Any]:
    return runtime.health_payload()


@app.get("/overview")
def get_overview(request: Request) -> dict[str, Any]:
    _require_user(request)
    return runtime.overview_payload()


@app.get("/meters")
def get_meters(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    _require_user(request)
    return runtime.meter_payload(limit=limit)


@app.get("/anomalies")
def get_anomalies(request: Request, limit: int = Query(25, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.anomaly_payload(limit=limit)


@app.get("/theft")
def get_theft(request: Request, limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.theft_payload(limit=limit)


@app.get("/weather-impact")
def get_weather_impact(request: Request) -> dict[str, Any]:
    _require_user(request)
    return runtime.weather_payload()


@app.get("/forecast")
def get_forecast(request: Request) -> dict[str, Any]:
    _require_user(request)
    return runtime.forecast_payload()


@app.get("/risk-scores")
def get_risk_scores(request: Request, limit: int = Query(25, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.risk_scores_payload(limit=limit)


@app.get("/efficiency")
def get_efficiency(request: Request, limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.efficiency_payload(limit=limit)


@app.get("/api/pole-status")
def get_pole_status(request: Request, limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.pole_status_payload(limit=limit)


@app.get("/api/pole-tamper-alerts")
def get_pole_tamper_alerts(request: Request, limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.pole_tamper_alerts_payload(limit=limit)


@app.get("/api/pole-energy-balance")
def get_pole_energy_balance(request: Request, limit: int = Query(120, ge=1, le=500)) -> dict[str, Any]:
    _require_user(request)
    return runtime.pole_energy_balance_payload(limit=limit)


@app.get("/drift-report")
def get_drift_report(request: Request) -> dict[str, Any]:
    _require_user(request)
    return runtime.drift_payload()


@app.get("/artifacts/daily-report")
def get_daily_report_artifact(request: Request) -> FileResponse:
    _require_user(request)
    return FileResponse(runtime.paths.daily_report, filename="daily_energy_report.pdf", media_type="application/pdf")


@app.get("/artifacts/drift-report")
def get_drift_report_artifact(request: Request) -> FileResponse:
    _require_user(request)
    return FileResponse(runtime.paths.drift_report, filename="drift_report.json", media_type="application/json")


@app.get("/artifacts/sample-overview")
def get_sample_overview_artifact(request: Request) -> FileResponse:
    _require_user(request)
    sample_output = runtime.paths.root / "sample_outputs" / "overview_response.json"
    return FileResponse(sample_output, filename="overview_response.json", media_type="application/json")


@app.get("/artifacts/heatmap")
def get_heatmap_artifact(request: Request) -> FileResponse:
    _require_user(request)
    dashboard_heatmap = runtime.paths.dashboard_dir / "theft_heatmap.html"
    return FileResponse(dashboard_heatmap, filename="theft_heatmap.html", media_type="text/html")


@app.post("/predict")
def predict_meter_status(payload: MeterReading | list[MeterReading], request: Request) -> list[dict[str, Any]]:
    _require_user(request)
    if isinstance(payload, list) and len(payload) > 256:
        raise HTTPException(status_code=413, detail="At most 256 readings may be submitted per request.")
    return runtime.predict_payload(payload)


_research_runs: dict[str, dict[str, Any]] = {}


@app.post("/api/research/experiments")
def start_research_experiment(payload: ResearchExperimentRequest, request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    run_id = uuid.uuid4().hex[:12]
    output = runtime.paths.reports_dir / "research_runs" / run_id
    config = payload.model_dump()
    try:
        result = run_experiment(config, output_dir=output)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["run_id"] = run_id
    result["artifact_dir"] = str(output.relative_to(runtime.paths.root))
    _research_runs[run_id] = result
    return result


@app.get("/api/research/experiments")
def list_research_experiments(request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    return {"runs": [{"run_id": key, "config": value["config"], "metrics": value["metrics"], "privacy": value["privacy"]}
                     for key, value in _research_runs.items()]}


@app.get("/api/research/experiments/{run_id}")
def get_research_experiment(run_id: str, request: Request) -> dict[str, Any]:
    _require_role(request, "admin")
    if run_id not in _research_runs:
        raise HTTPException(status_code=404, detail="Experiment not found in this server session.")
    return _research_runs[run_id]


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    if decode_session_token(websocket.cookies.get(SESSION_COOKIE_NAME)) is None:
        await websocket.close(code=4401, reason="Authentication required")
        return
    try:
        await runtime.register_client(websocket)
        while websocket in runtime.ws_clients:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unregister_client(websocket)
