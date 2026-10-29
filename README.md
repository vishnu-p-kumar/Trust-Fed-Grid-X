# GridSentinel-AI-Powered-Electricity-Theft-Grid-Intelligence-System

An end-to-end smart-grid analytics project built on synthetic Bengaluru smart-meter data. The system generates realistic meter readings, trains multiple machine learning models, replays a live monitoring stream through FastAPI, stores runtime state in SQLite, and serves a browser dashboard for operators.

> **Research prototype status:** This repository now includes an experimental, single-process federated learning simulator (FedAvg, trimmed mean, trust-weighted aggregation, and provisional HATTA). Its default fixture is generated classification data, not electricity data. It now includes experimental single-release Gaussian-noise and pairwise-mask simulation utilities, but these are not a formally accounted multi-round DP system or production secure-aggregation protocol. Research scores are not publication-ready benchmark results; see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) and [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

Some later sections in this legacy README and `model.md` still describe earlier demo behavior and interview answers. For current implementation behavior, use `docs/ARCHITECTURE.md`, `docs/METHOD.md`, `docs/EXPERIMENTS.md`, and `docs/IMPLEMENTATION_STATUS.md`.

This repository is useful for:

- data science and machine learning coursework
- electricity theft detection demos and prototypes
- anomaly detection and energy wastage monitoring
- smart-grid dashboard presentations
- full-stack ML system demonstrations

## Table of Contents

- [1. Project Summary](#1-project-summary)
- [2. Key Capabilities](#2-key-capabilities)
- [3. System Architecture](#3-system-architecture)
- [4. Repository Structure](#4-repository-structure)
- [5. Data Generation Pipeline](#5-data-generation-pipeline)
- [6. Dataset Schema](#6-dataset-schema)
- [7. Modeling and Analytics](#7-modeling-and-analytics)
- [8. Runtime and API Behavior](#8-runtime-and-api-behavior)
- [9. Dashboard](#9-dashboard)
- [10. Generated Artifacts](#10-generated-artifacts)
- [11. Installation](#11-installation)
- [12. How to Run](#12-how-to-run)
- [13. Configuration](#13-configuration)
- [14. Docker Support](#14-docker-support)
- [15. API Reference](#15-api-reference)
- [16. Testing](#16-testing)
- [17. Project Workflow Summary](#17-project-workflow-summary)
- [18. Troubleshooting](#18-troubleshooting)
- [19. Presentation Summary](#19-presentation-summary)
- [20. Detailed Model Selection Rationale](#20-detailed-model-selection-rationale)
- [21. In-Depth Interview Questions](#21-in-depth-interview-questions)
- [22. Federated Research Prototype](#22-federated-research-prototype)
  - [CEEW real smart-meter dataset](#ceew-real-smart-meter-dataset)
- [23. Roadmap improvements and evaluation tools](#23-roadmap-improvements-and-evaluation-tools)

## 1. Project Summary

At a high level, the project does five things:

1. Generates hourly synthetic smart-meter data for Bengaluru areas.
2. Simulates electricity theft, anomalies, energy wastage, weather effects, and pole-level tampering.
3. Trains anomaly detection, theft detection, and demand forecasting models.
4. Replays a live simulation through a FastAPI backend that updates every few seconds.
5. Visualizes current and historical analytics in a static dashboard.

The repository is meter-centric but also includes a pole-monitoring layer. Each meter is mapped into a `transformer -> pole -> meter` hierarchy, allowing the system to detect direct pole tapping, supply mismatch, and hidden unmetered consumption in addition to meter-level suspicious behavior.

## 2. Key Capabilities

- Synthetic smart-meter generation across multiple Bengaluru areas
- Weather-aware electricity consumption simulation
- Multiple theft scenarios such as bypass, tampering, illegal connections, and abnormal spikes
- Isolation Forest based anomaly detection
- Random Forest plus boosted-model theft classification
- Classifier probability blending; synthetic generator metadata is excluded from inference
- Meter-level risk scoring and risk categories
- Energy efficiency and wastage analytics
- Pole energy balance monitoring and pole tamper detection
- LSTM and Transformer-based demand forecasting with fallback behavior
- Data drift monitoring between historical and recent windows
- Explainable predictions for suspicious meters
- Optional alert dispatch to email, Slack, and Telegram
- FastAPI REST API plus WebSocket live stream
- Multi-section static dashboard for realtime monitoring and artifact downloads

## 3. System Architecture

The current implementation follows this flow:

1. `run_project.py` loads generation defaults from `utils/helpers.py`.
2. `src/data_generator.py` creates:
   - the main dataset
   - a sampled training dataset
   - a live simulation dataset
   - meter and pole catalogs
   - generation summary metadata
3. `src/train_models.py` trains:
   - Isolation Forest for anomaly detection
   - Random Forest for theft detection
   - XGBoost or a fallback boosted model for theft detection
   - LSTM demand forecaster
   - Transformer demand forecaster
4. `api/main.py` boots the runtime, regenerates missing artifacts when needed, loads models and datasets, and starts a simulation loop.
5. Each live tick is enriched with anomaly scores, theft probabilities, risk levels, efficiency metrics, pole energy balance, drift state, clustering output, and forecast output.
6. Runtime snapshots are written into SQLite tables in `database/meter_data.db`.
7. The dashboard reads API endpoints every few seconds and can also consume the `/ws/live` WebSocket payload.

## 4. Repository Structure

Important folders and files:

- `api/`
  FastAPI backend and live runtime simulation.

- `src/`
  Core Python modules for generation, preprocessing, training, forecasting, scoring, explainability, drift detection, pole monitoring, reporting, and exports.

- `dashboard/`
  Static frontend with HTML sections, shared JavaScript components, and styling.

- `data/raw/`
  Placeholder location for raw source data if the project is extended beyond synthetic generation.

- `data/processed/`
  Processed intermediate outputs such as the sample training set, live simulation data, meter catalog, pole catalog, and generation summary.

- `dataset/`
  Main generated synthetic dataset.

- `models/`
  Trained model artifacts and metadata files.

- `database/`
  SQLite runtime database generated by the API.

- `reports/`
  Generated PDF and JSON reporting artifacts.

- `maps/`
  Generated HTML heatmaps.

- `sample_outputs/`
  Example API request/response payloads.

- `tests/`
  Pytest-based tests for data generation, analytics modules, forecasting, API payloads, and runtime behavior.

- `run_project.py`
  Main bootstrap script for generation, training, exports, reports, and optional API startup.

- `run.md`
  PowerShell-oriented run guide.

- `model.md`
  High-level explanation of the models used in the project.

- `docker-compose.yml`
  Two-service stack for API and dashboard.

## 5. Data Generation Pipeline

The synthetic data generator is implemented in `src/data_generator.py`.

### 5.1 Geographic Coverage

Meters are distributed across Bengaluru locations defined in `utils/helpers.py`:

- Whitefield
- Electronic City
- Indiranagar
- Koramangala
- Marathahalli
- Yelahanka
- BTM Layout
- Jayanagar
- Malleshwaram
- Rajajinagar
- Hebbal
- Bellandur
- HSR Layout
- Banashankari
- Peenya Industrial Area

Coordinates are jittered around each area center so meters appear spatially distinct on the map while still staying inside Bengaluru bounds.

### 5.2 Usage Profiles

Every meter is assigned a usage profile that drives base load behavior:

- `residential`
- `night_usage`
- `industrial`
- `ac_heavy`
- `commercial`

These profiles influence load shape, temperature sensitivity, weekday/weekend patterns, and expected consumption.

### 5.3 Weather Generation

Weather is handled by `src/weather_api.py`.

Behavior:

- If `OPENWEATHER_API_KEY` is configured, the weather service can fetch current live weather from OpenWeather.
- For historical generation and fallback scenarios, the project simulates Bengaluru hourly weather.

Generated weather fields include:

- `temperature`
- `humidity`
- `rainfall`
- `wind_speed`
- `weather_condition`

### 5.4 Theft Scenarios

The dataset simulates these theft patterns:

- `meter_bypass`
- `abnormal_spikes`
- `constant_low_consumption`
- `illegal_connection`
- `tampered_meter`

These patterns modify reported consumption, actual load, voltage, current behavior, and downstream risk signals.

### 5.5 Pole Hierarchy

The project extends the meter dataset with:

- `transformer_id`
- `pole_id`
- `connected_meters`

The pole hierarchy is generated automatically and stored in `data/processed/pole_catalog.csv`. This enables pole-level supply simulation and tamper detection during runtime.

### 5.6 Default Generation Settings

`utils/helpers.py` defines two standard configurations.

Default mode:

- `num_meters = 500`
- `days = 60`
- `chunk_size = 45`
- `sample_rows = 45000`
- `simulation_days = 10`
- `simulation_meter_limit = 425`
- `seed = 42`

Full-scale mode:

- `num_meters = 1000`
- `days = 365`
- `chunk_size = 100`
- `sample_rows = 120000`
- `simulation_days = 14`
- `simulation_meter_limit = 80`
- `seed = 42`

### 5.7 Live Simulation Stability

The live replay dataset intentionally behaves differently from the broader historical training data:

- a deterministic subset of live meters is selected as theft candidates
- the selected live theft meters remain stable during the live replay window
- non-selected live meters can have random theft disabled during replay

This keeps the dashboard easier to follow during presentations because the suspicious meters do not jump randomly every few seconds.

## 6. Dataset Schema

### 6.1 Main Generated Files

The generator writes these key outputs:

- `dataset/smart_meter_data.csv`
  Full synthetic dataset.

- `data/processed/smart_meter_sample.csv`
  Sampled training-friendly dataset.

- `data/processed/live_simulation.csv`
  Dataset used by the FastAPI live runtime.

- `data/processed/meter_catalog.csv`
  Meter metadata.

- `data/processed/pole_catalog.csv`
  Pole and transformer hierarchy.

- `data/processed/generation_summary.json`
  Generation metadata and live theft meter list.

### 6.2 Core Meter Columns

Important source columns include:

- `meter_id`
- `timestamp`
- `region`
- `area`
- `transformer_id`
- `pole_id`
- `connected_meters`
- `latitude`
- `longitude`
- `voltage`
- `current`
- `power`
- `consumption_kwh`
- `power_factor`
- `temperature`
- `humidity`
- `rainfall`
- `wind_speed`
- `weather_condition`
- `is_theft`
- `expected_consumption_kwh`
- `wastage_score`
- `usage_profile`
- `theft_type`
- `seeded_theft_probability`

### 6.3 Preprocessing and Engineered Fields

`src/preprocess.py` and `src/feature_engineering.py` add additional runtime and model features such as:

- `wastage_flag`
- `power_gap`
- `temperature_band`
- `hour_of_day`
- `day_of_week`
- `rolling_average_consumption`
- `consumption_variance`
- `peak_usage_ratio`
- `night_usage_ratio`
- `weather_consumption_ratio`
- `power_factor_loss`
- `voltage_irregularity`
- `current_power_gap`

Categorical feature groups include:

- `region`
- `area`
- `weather_condition`
- `usage_profile`

### 6.4 Runtime-Derived Fields

After live scoring, the runtime adds fields like:

- `anomaly_score`
- `is_anomaly`
- `random_forest_probability`
- `xgboost_probability`
- `theft_probability`
- `status`
- `risk_score`
- `risk_level`
- `efficiency_score`
- `estimated_losses_kwh`
- `tamper_probability`
- `tamper_flag`
- `energy_gap`

## 7. Modeling and Analytics

### 7.1 Anomaly Detection

Files:

- `src/detect_anomaly.py`
- `src/train_models.py`

Model:

- Isolation Forest

Purpose:

- detect unusual readings even when theft is not explicitly confirmed
- produce `anomaly_score` and `is_anomaly`

### 7.2 Theft Detection

Files:

- `src/theft_detector.py`
- `src/train_models.py`

Models:

- Random Forest
- XGBoost when available
- `HistGradientBoostingClassifier` fallback if XGBoost is unavailable

Final theft probability is calibrated using:

- model probabilities
- anomaly score
- seeded theft probability
- wastage score

Outputs include:

- `random_forest_probability`
- `xgboost_probability`
- `theft_probability`
- `status`

### 7.3 Risk Scoring

File:

- `src/risk_scoring.py`

Purpose:

- combine model outputs and electrical heuristics into a single risk score
- assign `Low`, `Medium`, `High`, and `Critical` risk levels

### 7.4 Energy Efficiency and Wastage

File:

- `src/energy_efficiency.py`

Purpose:

- compute efficiency metrics
- estimate losses in kWh
- surface low-efficiency and wastage-sensitive meters

### 7.5 Pole Monitoring and Pole Tamper Detection

Files:

- `src/pole_monitoring.py`
- `src/pole_tamper_detector.py`

Purpose:

- aggregate meter load into pole-level supply behavior
- estimate technical losses and hidden load
- compute energy mismatch between supplied and metered energy
- detect suspicious poles and likely illegal connections

Core idea:

`supplied_energy - (meter_energy_sum + loss_estimate) = energy_gap`

Large or abnormal energy gaps can trigger pole tamper alerts.

### 7.6 Demand Forecasting

Files:

- `src/demand_forecasting.py`
- `src/transformer_forecasting.py`

Models:

- LSTM forecaster
- Transformer forecaster
- seasonal/baseline fallback logic when deep learning dependencies or artifacts are unavailable

Forecast outputs include:

- `next_hour`
- `next_day`
- `next_week`
- forecast `series`
- comparison between LSTM and Transformer outputs

### 7.7 Explainability

File:

- `src/explainable_ai.py`

Methods:

- SHAP when available
- fallback feature-importance style explanations when SHAP is unavailable

Purpose:

- explain suspicious predictions in human-readable form
- support theft investigation tables and API responses

### 7.8 Data Drift Monitoring

File:

- `src/data_drift_monitor.py`

Methods:

- Evidently when available
- fallback statistical drift checks otherwise

Purpose:

- compare recent live data to historical reference data
- identify feature drift, data-quality issues, and theft-rate shifts

### 7.9 Hyperparameter Optimization

File:

- `src/model_optimizer.py`

Method:

- Optuna

Purpose:

- tune anomaly and theft model parameters before training when explicitly requested from the CLI

### 7.10 Reporting and Export Utilities

Files:

- `src/report_generator.py`
- `src/spatial_analysis.py`
- `src/sample_outputs.py`

Outputs include:

- PDF report
- drift JSON report
- HTML heatmap
- example prediction and overview API payloads

## 8. Runtime and API Behavior

The backend lives in `api/main.py`.

### 8.1 Startup Behavior

On startup the runtime:

1. ensures project directories exist
2. regenerates data artifacts if key files are missing or outdated
3. retrains models if required model artifacts are missing
4. loads historical and live datasets
5. initializes forecast, drift, and pole-monitoring state
6. advances one simulation tick immediately so the dashboard has data at first load

### 8.2 Live Tick Processing

Every update cycle:

1. the runtime selects one timestamp slice from `live_simulation.csv`
2. meter readings are classified for anomaly and theft
3. a visible theft candidate may be injected if none is present
4. the first detected theft meter can be kept sticky for demo stability
5. theft alerts are capped for readability
6. risk and efficiency metrics are computed
7. pole energy is simulated and checked for tampering
8. recent windows are updated for drift monitoring
9. forecast data is rebuilt
10. SQLite tables are refreshed
11. the latest snapshot is broadcast to WebSocket clients
12. periodic reports and heatmap outputs are regenerated on schedule

### 8.3 SQLite Tables

The backend writes runtime data into `database/meter_data.db` using tables such as:

- `meter_readings`
- `live_predictions`
- `recent_predictions`
- `risk_scores`
- `efficiency_metrics`
- `drift_reports`
- `forecast_snapshots`
- `pole_energy_data`
- `pole_tamper_events`

### 8.4 Alert Integrations

`src/alert_engine.py` supports optional outbound alert delivery through:

- SMTP email
- Slack webhook
- Telegram bot API

Alerts are sent only when `SMARTGRID_ENABLE_ALERTS=1`.

## 9. Dashboard

The frontend is a static dashboard in `dashboard/`.

Main files:

- `dashboard/index.html`
- `dashboard/main.js`
- `dashboard/style.css`

Shared components:

- `dashboard/components/core.js`
- `dashboard/components/charts.js`
- `dashboard/components/alerts.js`
- `dashboard/components/forecast.js`
- `dashboard/components/heatmap.js`

The dashboard defaults to the API base URL `http://127.0.0.1:8000` and stores UI state such as selected section and theme in local storage.

### 9.1 Dashboard Sections

Each file in `dashboard/sections/` represents one analytics view:

- `overview.html`
  High-level KPIs, live demand, and system health.

- `live_monitoring.html`
  Realtime usage and meter-level monitoring.

- `theft_detection.html`
  Theft counts, suspicious meters, and investigation priorities.

- `anomaly_detection.html`
  Anomaly metrics and suspicious outlier summaries.

- `demand_forecast.html`
  LSTM and Transformer forecast comparisons.

- `energy_efficiency.html`
  Wastage and efficiency views.

- `pole_monitoring.html`
  Pole energy balance, suspicious poles, and illegal connection signals.

- `heatmap.html`
  Interactive theft hotspot map.

- `weather_impact.html`
  Demand behavior against weather conditions.

- `alerts.html`
  Consolidated alerts and operational notifications.

- `reports.html`
  Downloadable reports and sample output references.

### 9.2 Refresh Pattern

The dashboard polls the backend frequently and also supports the `/ws/live` WebSocket snapshot payload. The backend update interval defaults to 4 seconds and can be changed with `SMARTGRID_UPDATE_INTERVAL`.

## 10. Generated Artifacts

Common generated files include:

- `dataset/smart_meter_data.csv`
- `data/processed/smart_meter_sample.csv`
- `data/processed/live_simulation.csv`
- `data/processed/meter_catalog.csv`
- `data/processed/pole_catalog.csv`
- `data/processed/generation_summary.json`
- `models/isolation_forest.pkl`
- `models/random_forest.pkl`
- `models/xgboost_model.pkl`
- `models/lstm_model.h5`
- `models/transformer_forecaster.pt`
- `models/model_metadata.json`
- `models/demand_metadata.json`
- `models/transformer_metadata.json`
- `models/optimizer_best_params.json` when optimization is used
- `database/meter_data.db`
- `reports/daily_energy_report.pdf`
- `reports/drift_report.json`
- `maps/theft_heatmap.html`
- `dashboard/theft_heatmap.html`
- `sample_outputs/predict_request.json`
- `sample_outputs/predict_response.json`
- `sample_outputs/theft_response.json`
- `sample_outputs/overview_response.json`
- `sample_outputs/manifest.json`

## 11. Installation

The repository is currently easiest to run in Windows PowerShell, and `run.md` is written with that workflow in mind.

### 11.1 Python Version

Use Python 3.11 if possible.

Example interpreter path used on this machine:

```powershell
$PYTHON = "C:\Users\vishn\AppData\Local\Programs\Python\Python311\python.exe"
```

If `python` is already configured in your PATH, you can use `python` instead of `$PYTHON`.

### 11.2 Create `.env`

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

### 11.3 Install Dependencies

Core dependencies from `requirements.txt` include:

- FastAPI
- Uvicorn
- pandas
- NumPy
- scikit-learn
- XGBoost
- TensorFlow
- matplotlib
- seaborn
- Plotly
- folium
- geopandas
- joblib
- requests
- websockets
- SHAP
- pydantic

Install the main stack:

```powershell
& $PYTHON -m pip install -r requirements.txt
```

Install testing tools:

```powershell
& $PYTHON -m pip install -r requirements-test.txt
```

Optional advanced dependencies:

```powershell
& $PYTHON -m pip install -r requirements-advanced.txt
```

Advanced extras add packages such as:

- Optuna
- Evidently
- PyTorch

## 12. How to Run

### 12.1 Quick Start

```powershell
cd "c:\Users\vishn\Desktop\College\SEMISTER\CSE 6th SEM\Data Science\Project 1\electricity_theft_and_anomaly_detection"
$PYTHON = "C:\Users\vishn\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& $PYTHON -m pip install -r requirements.txt
& $PYTHON -m pip install -r requirements-test.txt
& $PYTHON run_project.py
& $PYTHON -m pytest
& $PYTHON -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd "c:\Users\vishn\Desktop\College\SEMISTER\CSE 6th SEM\Data Science\Project 1\electricity_theft_and_anomaly_detection\dashboard"
$PYTHON = "C:\Users\vishn\AppData\Local\Programs\Python\Python311\python.exe"
& $PYTHON -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/index.html
```

### 12.2 Bootstrap Data and Models

```powershell
& $PYTHON run_project.py
```

This command:

- generates synthetic datasets
- writes meter and pole catalogs
- trains anomaly, theft, and forecasting models
- exports sample outputs
- builds the theft heatmap
- generates the PDF report

### 12.3 Generate Data Only

```powershell
& $PYTHON run_project.py --skip-training
```

### 12.4 Useful CLI Options

`run_project.py` currently supports:

- `--full-scale`
  Generate the 1000-meter, 365-day configuration.

- `--num-meters`
  Override the number of meters.

- `--days`
  Override the number of simulation days.

- `--skip-training`
  Generate data without model training.

- `--forecast-epochs`
  Set LSTM training epochs.

- `--skip-sample-export`
  Skip writing example API payloads.

- `--skip-report`
  Skip generating the PDF report.

- `--start-api`
  Start FastAPI immediately after bootstrapping.

- `--optimize-models`
  Run Optuna-based optimization before training.

- `--optimization-trials`
  Set the number of Optuna trials.

### 12.5 Start the API Manually

```powershell
& $PYTHON -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

### 12.6 Demo Mode

To reduce periodic report generation during a demo:

```powershell
$env:SMARTGRID_DEMO_MODE = "1"
& $PYTHON -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 12.7 Start the Frontend

```powershell
Set-Location dashboard
& $PYTHON -m http.server 8080
```

### 12.8 Run Everything with a Single Bootstrap Command

```powershell
& $PYTHON run_project.py --start-api
```

This generates artifacts first and then launches the backend.

## 13. Configuration

### 13.1 Main Environment Variables

The `.env.example` file currently includes:

- `OPENWEATHER_API_KEY`
- `SMARTGRID_UPDATE_INTERVAL`
- `SMARTGRID_FULL_SCALE`
- `SMARTGRID_ENABLE_ALERTS`
- `SMARTGRID_SLACK_WEBHOOK`
- `SMARTGRID_TELEGRAM_BOT_TOKEN`
- `SMARTGRID_TELEGRAM_CHAT_ID`
- `SMARTGRID_SMTP_HOST`
- `SMARTGRID_SMTP_PORT`
- `SMARTGRID_SMTP_USER`
- `SMARTGRID_SMTP_PASSWORD`
- `SMARTGRID_ALERT_EMAIL_FROM`
- `SMARTGRID_ALERT_EMAIL_TO`

Additional runtime variables used in code:

- `SMARTGRID_DEMO_MODE`
  Disables periodic reporting by default for presentation-friendly runtime behavior.

- `SMARTGRID_ENABLE_PERIODIC_REPORTS`
  Explicitly controls periodic report generation in the runtime loop.

### 13.2 Common Configuration Notes

- Set `SMARTGRID_FULL_SCALE=1` to make the API bootstrap using the full generation profile.
- Set `SMARTGRID_ENABLE_ALERTS=1` only when email, Slack, or Telegram configuration is ready.
- Leave `OPENWEATHER_API_KEY` empty if synthetic weather is acceptable.
- Adjust `SMARTGRID_UPDATE_INTERVAL` if you want a faster or slower live replay loop.

## 14. Docker Support

The project includes both a `Dockerfile` and `docker-compose.yml`.

### 14.1 Dockerfile

The API image:

- uses Python 3.11
- installs `requirements.txt`
- exposes port `8000`
- starts `uvicorn api.main:app --host 0.0.0.0 --port 8000`

### 14.2 Docker Compose Services

The compose setup defines two services:

- `api`
  FastAPI backend with mounted volumes for data, models, reports, maps, database, and sample outputs.

- `dashboard`
  Nginx-based static dashboard server.

Exposed ports:

- API: `8000`
- Dashboard: `8080`

### 14.3 Run with Docker Compose

```powershell
docker compose up --build
```

Then open:

- API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:8080`

## 15. API Reference

### 15.1 Main Endpoints

- `GET /`
- `GET /health`
- `GET /overview`
- `GET /meters`
- `GET /anomalies`
- `GET /theft`
- `GET /weather-impact`
- `GET /forecast`
- `GET /risk-scores`
- `GET /efficiency`
- `GET /api/pole-status`
- `GET /api/pole-tamper-alerts`
- `GET /api/pole-energy-balance`
- `GET /drift-report`
- `POST /predict`

### 15.2 Artifact Endpoints

- `GET /artifacts/daily-report`
- `GET /artifacts/drift-report`
- `GET /artifacts/sample-overview`
- `GET /artifacts/heatmap`

### 15.3 WebSocket Endpoint

- `WS /ws/live`

### 15.4 `POST /predict` Input

The prediction endpoint accepts one meter reading or a list of readings. Important fields include:

- `meter_id`
- `timestamp` optional
- `region`
- `area`
- `latitude`
- `longitude`
- `voltage`
- `current`
- `power`
- `consumption_kwh`
- `power_factor`
- `temperature`
- `humidity`
- `rainfall`
- `wind_speed`
- `weather_condition`
- `expected_consumption_kwh` optional
- `wastage_score` optional
- `usage_profile`
- `theft_type`
- `seeded_theft_probability`

### 15.5 `POST /predict` Output

Each prediction response returns fields such as:

- `meter_id`
- `region`
- `area`
- `latitude`
- `longitude`
- `status`
- `theft_probability`
- `anomaly_score`
- `risk_score`
- `risk_level`
- `efficiency_score`
- `reason`

## 16. Testing

Pytest coverage exists for major system layers.

Current test modules:

- `tests/test_api.py`
- `tests/test_data_pipeline.py`
- `tests/test_forecasting_and_exports.py`
- `tests/test_advanced_analytics.py`
- `tests/test_theft_detector.py`

Covered areas include:

- API health and payload structure
- live theft candidate injection behavior
- sticky theft runtime behavior
- theft and anomaly payload counts
- pole endpoint responses and pole payload helpers
- meter catalog geographic bounds
- stable live theft generation rules
- preprocessing and feature engineering outputs
- pole hierarchy generation and energy-gap detection
- risk scoring and efficiency metrics
- forecasting fallback behavior
- export generation
- theft probability calibration
- WebSocket registration failure handling

Run tests with:

```powershell
& $PYTHON -m pytest
```

## 17. Project Workflow Summary

A typical workflow for this repository is:

1. Install dependencies.
2. Generate synthetic data and train models with `run_project.py`.
3. Start the FastAPI backend.
4. Start the static dashboard server.
5. Use `/health` to confirm the backend is ready.
6. Open the dashboard and point it to `http://127.0.0.1:8000`.
7. Monitor theft, anomalies, forecasts, pole alerts, and report artifacts.

## 18. Troubleshooting

### 18.1 Deep Learning Library Warnings

Warnings from TensorFlow or PyTorch about CPU optimizations, missing GPU support, or oneDNN are usually informational.

### 18.2 Optional Dependency Fallbacks

The codebase includes several fallbacks:

- boosted theft model fallback when XGBoost is unavailable
- forecast fallback when TensorFlow or PyTorch artifacts are unavailable
- drift fallback when Evidently is unavailable
- explainability fallback when SHAP is unavailable
- synthetic weather fallback when OpenWeather is not configured

### 18.3 Dashboard Shows No Data

Check that:

- the API is running on `127.0.0.1:8000`
- the frontend static server is running on `8080`
- the dashboard API base URL is correct
- `run_project.py` or API bootstrap has generated required artifacts

### 18.4 Missing Reports or Heatmaps

Regenerate artifacts with:

```powershell
& $PYTHON run_project.py
```

### 18.5 Alert Delivery Not Working

Check that:

- `SMARTGRID_ENABLE_ALERTS=1`
- SMTP, Slack, or Telegram variables are correctly configured
- the machine has outbound network access

### 18.6 Runtime Regenerates Data Unexpectedly

The API bootstrap intentionally regenerates artifacts when required files are missing or when the live dataset and metadata do not match the expected configuration.

## 19. Presentation Summary

This project simulates a Bengaluru smart-grid intelligence platform. It generates synthetic smart-meter data with weather, theft, anomaly, wastage, and pole-level tamper behavior; trains multiple ML models; replays the data as a live FastAPI stream; stores runtime snapshots in SQLite; and visualizes the full monitoring workflow in a multi-section dashboard. The result is a complete demonstration of electricity theft detection, anomaly monitoring, energy efficiency analysis, demand forecasting, weather impact analysis, drift detection, and pole tamper surveillance.

## 20. Detailed Model Selection Rationale

This project does not use only one model because the problem is not only one machine learning task. Electricity theft detection involves abnormal-pattern discovery, supervised theft classification, demand forecasting, risk ranking, pole-level energy-balance analysis, explainability, and drift monitoring. Each part has a different data shape and operational goal.

### 20.1 Models Actually Used

| Project Need | Model or Method Used | Main File | Why It Is Used |
| --- | --- | --- | --- |
| Meter anomaly detection | Isolation Forest | `src/train_models.py`, `src/detect_anomaly.py` | Detects rare or unusual meter behavior without needing every possible theft pattern to be labelled. |
| Theft classification | Random Forest Classifier | `src/train_models.py`, `src/theft_detector.py` | Strong, stable model for structured tabular data; handles nonlinear rules and mixed feature interactions well. |
| Theft classification | XGBoost Classifier | `src/train_models.py`, `src/theft_detector.py` | Boosted trees usually perform well on fraud-like structured data and can capture sharp decision boundaries. |
| Theft classification fallback | HistGradientBoostingClassifier | `src/train_models.py` | Keeps the pipeline working even if the `xgboost` package is unavailable. |
| Demand forecasting | LSTM | `src/demand_forecasting.py` | Learns sequential consumption behavior such as hourly and daily demand cycles. |
| Demand forecasting | Transformer Regressor | `src/transformer_forecasting.py` | Provides an advanced time-series model that can learn longer-range temporal relationships. |
| Forecast fallback | Seasonal baseline forecaster | `src/demand_forecasting.py`, `src/transformer_forecasting.py` | Keeps forecast APIs and dashboards functional even when deep learning libraries or trained artifacts are missing. |
| Pole tamper detection | Isolation Forest plus energy-balance heuristics | `src/pole_tamper_detector.py` | Detects abnormal pole-level energy mismatch, hidden load, and possible illegal connections. |
| Risk ranking | Weighted risk scoring | `src/risk_scoring.py` | Converts model outputs and electrical signals into an operational priority score. |
| Explainability | SHAP or fallback feature importance | `src/explainable_ai.py` | Explains why a meter is suspicious, which is important for inspector decisions. |
| Drift monitoring | Evidently or fallback statistical drift checks | `src/data_drift_monitor.py` | Checks whether live data has shifted away from historical training data. |
| Hyperparameter tuning | Optuna | `src/model_optimizer.py` | Optionally searches better parameters for Isolation Forest, Random Forest, and XGBoost. |

### 20.2 Why Isolation Forest Was Chosen

Isolation Forest is used for anomaly detection because theft and electrical faults are rare compared with normal readings. The model works by isolating unusual points faster than common points. In this project, it is trained mainly on normal records and produces an `anomaly_score`. A high score means the reading behaves differently from the learned normal pattern.

It is useful here because not every real theft case will look exactly like the synthetic labels. For example, a meter bypass, voltage drop, power-factor issue, and unusual night usage may appear in different combinations. Isolation Forest gives the system a way to flag suspicious readings even before the supervised classifier is fully confident.

Why not only Isolation Forest:

- It is unsupervised, so it does not directly learn the difference between theft, wastage, and harmless unusual demand.
- It can flag genuine high consumption as anomalous.
- It gives anomaly evidence, but it is not enough for final theft classification.

### 20.3 Why Random Forest Was Chosen

Random Forest is used as a supervised theft classifier because the dataset contains labelled `is_theft` examples generated from known theft scenarios. It works well with tabular features such as voltage, current, consumption, power factor, weather, usage profile, area, rolling averages, night usage ratio, and wastage score.

It was chosen because:

- it is robust on tabular data
- it handles nonlinear relationships
- it is less sensitive to feature scaling
- it gives reliable baseline performance
- it can handle noisy synthetic data better than a single decision tree
- it supports class balancing with `class_weight="balanced_subsample"`

Why not only Random Forest:

- Boosted models often provide better probability separation on fraud-style tabular data.
- Random Forest may average many trees and become less sharp near difficult boundaries.
- It may be less efficient than boosting when the goal is high recall on rare suspicious events.

### 20.4 Why XGBoost Was Chosen

XGBoost is used as the stronger boosted theft classifier. It trains trees sequentially, where each new tree focuses on errors from previous trees. This is useful in theft detection because the suspicious class may be rare, and important patterns may depend on feature interactions.

It was chosen because:

- it performs very well on structured/tabular ML problems
- it can capture complex theft signatures
- it usually gives strong ranking probabilities
- it handles mixed signal strength better than simple linear models
- it is common in fraud detection, credit risk, and anomaly-heavy classification

Why a fallback model is included:

The project should still run on machines where XGBoost is not installed. In that case, `HistGradientBoostingClassifier` is used as a scikit-learn fallback. The saved artifact path remains `models/xgboost_model.pkl`, but the metadata records whether the actual model is `xgboost` or `hist_gradient_boosting_fallback`.

### 20.5 Why Random Forest and XGBoost Are Combined

The final theft probability is not taken from only one classifier. In `src/theft_detector.py`, the project blends the probabilities:

```text
blended_probability = 0.45 * random_forest_probability + 0.55 * xgboost_probability
```

Then the project calibrates the final score using:

- blended supervised model probability
- seeded theft probability from the generator
- anomaly score
- wastage score

The theft probability is now a blend of the supervised classifier outputs only. Anomaly and wastage remain separate signals, and the current fallback decision threshold is 0.5 unless the saved model metadata specifies another threshold. This threshold still requires selection on a validation set for research evaluation.

This design was chosen because a practical theft system should not depend on one signal. A meter may be suspicious because of model confidence, abnormal usage, wastage, voltage irregularity, or seeded theft behavior in the live demo.

### 20.6 Why LSTM Was Chosen for Forecasting

Electricity demand is time-series data. Hourly consumption depends on previous hours, time of day, day of week, weather, and usage profile. LSTM is designed to learn sequential patterns and can remember context over a lookback window.

In this project:

- readings are grouped by timestamp
- total consumption is converted into a demand series
- the model uses a 24-hour lookback window
- it predicts future values recursively
- outputs are summarized as next hour, next day, and next week

Why not only a statistical model:

Simple methods like moving average or ARIMA are easier to explain, but they may struggle when demand behavior is nonlinear and influenced by weather, area mix, and usage profile changes.

### 20.7 Why Transformer Forecasting Was Added

The Transformer forecaster gives a second advanced forecasting approach. Unlike LSTM, a Transformer encoder can learn relationships across positions in the input sequence using attention. In this project it is intentionally compact, CPU-friendly, and used for comparison rather than as a huge deep learning model.

It was chosen because:

- it represents a modern time-series modeling approach
- it can learn longer temporal relationships
- it gives a useful comparison against LSTM
- it makes the forecasting module more complete for academic explanation

Why not only Transformer:

- it may need more data to outperform simpler methods
- it depends on PyTorch availability
- it is less straightforward to explain than LSTM or tree models
- for a small demo dataset, a baseline may sometimes be more stable

### 20.8 Why Baseline Fallbacks Are Important

The project includes fallback behavior intentionally. If TensorFlow, PyTorch, XGBoost, SHAP, Evidently, or Optuna is unavailable, the system should still run.

Fallbacks are important because:

- project demos should not fail due to optional library installation issues
- API endpoints should keep returning valid payloads
- dashboards should remain usable
- tests can validate behavior without requiring every heavy dependency
- users can run the core project on normal laptops

### 20.9 Other Models That Could Be Used

The following models are possible alternatives or future improvements.

| Model | Where It Could Be Used | Why It Could Help | Why It Was Not the Main Choice |
| --- | --- | --- | --- |
| Logistic Regression | Theft classification | Simple, interpretable baseline | Too linear for complex theft patterns. |
| Decision Tree | Theft classification | Easy to visualize and explain | Overfits easily and is weaker than Random Forest. |
| Support Vector Machine | Theft classification/anomaly | Can work well on smaller datasets | Scaling and probability calibration are harder for large tabular streams. |
| KNN | Anomaly or theft classification | Simple distance-based reasoning | Slow at prediction time and sensitive to scaling. |
| Naive Bayes | Baseline classification | Fast and simple | Feature independence assumption is unrealistic for electrical data. |
| CatBoost | Theft classification | Excellent for categorical tabular features | Adds another external dependency; XGBoost is already common and available. |
| LightGBM | Theft classification | Very fast gradient boosting | Another dependency; XGBoost/scikit fallback is enough for this project. |
| Autoencoder | Anomaly detection | Learns compressed normal behavior | Needs neural-network training and tuning; less interpretable than Isolation Forest. |
| One-Class SVM | Anomaly detection | Classic one-class anomaly method | Can be slow and sensitive to kernel/scaling choices. |
| Local Outlier Factor | Anomaly detection | Good for local density anomalies | Less convenient for stable deployed prediction on new streaming points. |
| ARIMA/SARIMA | Demand forecasting | Strong classical time-series baseline | Less flexible for nonlinear effects and multiple external signals. |
| Prophet | Demand forecasting | Good trend/seasonality decomposition | Extra dependency and less aligned with custom deep learning comparison. |
| GRU | Demand forecasting | Lighter alternative to LSTM | Similar purpose; LSTM is more commonly explained in coursework. |
| Temporal CNN | Demand forecasting | Fast sequence modeling | Less familiar for many interview explanations. |
| Graph Neural Network | Grid/pole topology modeling | Could model transformer-pole-meter relationships directly | More complex and not necessary for this demo scale. |

### 20.10 Did This Project Choose Only One Model?

No. The project uses a multi-model approach:

- Isolation Forest finds unusual meter behavior.
- Random Forest learns labelled theft patterns.
- XGBoost or HistGradientBoosting improves supervised theft scoring.
- LSTM forecasts future demand.
- Transformer forecasting provides an advanced comparison model.
- Pole tamper detection combines energy-balance rules with Isolation Forest.
- Risk scoring blends model outputs into operational severity.

The operational status combines supervised classification with separate anomaly and wastage statuses. Synthetic labels and seeded generation context must not be treated as prediction evidence.

### 20.11 Final Model Decision Summary

The selected models are appropriate because the project uses structured smart-meter data, rare-event theft labels, time-series demand behavior, and operational dashboard requirements.

- For anomaly detection, Isolation Forest is simple, fast, and suitable for rare abnormal readings.
- For theft classification, Random Forest gives robustness and XGBoost gives stronger boosted performance.
- For forecasting, LSTM and Transformer models match the sequential nature of electricity demand.
- For deployment reliability, fallback models keep the system usable even when optional ML libraries are missing.
- For real-world usefulness, explainability, drift monitoring, and risk scoring turn raw predictions into inspection-ready information.

## 21. In-Depth Interview Questions

## 22. Federated Research Prototype

The research experiment code is in `src/federated/`. It provides deterministic client partitions, local logistic training, FedAvg, coordinate trimmed mean, a provisional temporal trust candidate, poisoning hooks, metrics, and JSON/checkpoint artifacts. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for scope and limitations.

Run automated tests with `python -m pytest -q`. Run a measured CPU smoke experiment and save results with:

```powershell
python -c "from src.federated.experiment import run_experiment; print(run_experiment({'samples': 400, 'features': 10, 'clients': 4, 'rounds': 3, 'local_epochs': 2, 'partition': 'severe', 'attack': 'sign', 'seed': 42}, output_dir='reports/research_runs/smoke'))"
```

The dashboard's **Federated Research** view uses admin-only API endpoints. Configure `SMARTGRID_ADMIN_PASSWORD`, a random `SMARTGRID_SESSION_SECRET` of at least 32 characters, and `SMARTGRID_COOKIE_SECURE` in `.env` before login. For local HTTP development only, set `SMARTGRID_COOKIE_SECURE=0`. Do not commit `.env`.

The experiment's validation set is visible to the coordinator. Client simulation runs in one process and is not a deployment privacy boundary. Differential privacy and secure aggregation are disabled. The attack simulations do not yet include targeted backdoor success measurement. Do not describe the provisional method as novel or superior without further experiments and literature review.

### CEEW real smart-meter dataset

The supplied CEEW Mathura/Bareilly measurements are integrated as unlabeled real consumption data. The source files contain three-minute meter readings and separate daily aggregate series; none contains verified theft labels. Use them for consumption forecasting, supply-quality/heterogeneity analysis, and unlabeled anomaly triage only. Do not report theft-detection classification metrics from these observations. See [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for the inspected schema, quality report, license citation, exact results, and limitations.

Use the chunked ingestion command with paths to files you have acquired; raw readings stay outside version control and normalized outputs are written under ignored `data/processed/ceew/`:

```powershell
python scripts/prepare_ceew_dataset.py --profile-only --output reports/research_runs/ceew_profile --input "C:\path\to\source.csv"
python scripts/prepare_ceew_dataset.py --output data/processed/ceew --input "C:\path\to\SM Cleaned Data BR Aggregated.csv" "C:\path\to\SM Cleaned Data MH Aggregated.csv"
```

The daily aggregate data supports a meter-partitioned, chronological consumption forecast comparison. See [`docs/DATASETS.md`](docs/DATASETS.md) for the full command and validation protocol. The federated electricity-theft classifier and trust-detection results remain on their separate synthetic labeled fixture.

The forecast runner supports deterministic complete-meter client groups and a two-client city split via `--client-partition location --clients 2`; both modes avoid splitting a meter across clients. The city split is an exploratory non-IID pilot only.

These questions are based directly on this project. They cover data generation, preprocessing, model selection, training, backend runtime behavior, API design, dashboard design, testing, and deployment.

### 21.1 Project Overview Questions

1. What real-world problem does this project solve, and why is electricity theft detection important for smart grids?
2. Why did you build this system as an end-to-end project instead of only a machine learning notebook?
3. What are the main modules of the project, and how does data flow from generation to dashboard?
4. What is the difference between electricity theft, anomaly, and power wastage in this project?
5. How does the project simulate real smart-meter behavior for Bengaluru?
6. Why did you include both meter-level and pole-level monitoring?
7. What are the main outputs of the system for an electricity board or field inspector?
8. How would you explain this project to a non-technical electricity department officer?
9. What makes this project different from a basic binary classification project?
10. What are the limitations of using synthetic data for electricity theft detection?

### 21.2 Dataset and Data Generation Questions

1. Why did you generate synthetic data instead of using a public real-world dataset?
2. What assumptions did you make while generating meter consumption data?
3. Which Bengaluru areas are included, and why is location useful in theft detection?
4. How are usage profiles such as residential, commercial, industrial, night usage, and AC-heavy simulated?
5. How does temperature affect expected electricity consumption in the generated data?
6. How are rainfall, humidity, and wind speed used in the data?
7. What is the difference between `expected_consumption_kwh`, `consumption_kwh`, and `power`?
8. How are theft scenarios injected into the dataset?
9. Explain the five theft types used in this project.
10. What is `seeded_theft_probability`, and why is it useful in live simulation?
11. How does the project keep live theft meters stable for dashboard demonstrations?
12. What is the purpose of `meter_catalog.csv`?
13. What is the purpose of `pole_catalog.csv`?
14. How does the transformer-pole-meter hierarchy help identify illegal connections?
15. How would the system change if real smart-meter data were available?

### 21.3 Preprocessing and Feature Engineering Questions

1. Why is feature engineering important for smart-meter theft detection?
2. What are the base numerical features used for model training?
3. Which categorical features are one-hot encoded?
4. Why is `hour_of_day` important?
5. Why is `day_of_week` important?
6. How is `rolling_average_consumption` calculated, and why does it help?
7. What does `consumption_variance` indicate?
8. What is `peak_usage_ratio`, and how can it reveal abnormal usage?
9. What is `night_usage_ratio`, and why is night usage important in theft cases?
10. Why is `weather_consumption_ratio` useful?
11. How can `power_factor_loss` indicate inefficient or suspicious behavior?
12. How does `voltage_irregularity` help detect tampering?
13. What is `current_power_gap`, and what electrical inconsistency can it capture?
14. How does the code handle missing columns in incoming prediction requests?
15. Why must prediction-time feature columns match training-time feature columns?

### 21.4 Anomaly Detection Questions

1. Why did you choose Isolation Forest for anomaly detection?
2. How does Isolation Forest detect abnormal data points?
3. Why is anomaly detection useful even when labelled theft data exists?
4. What is the meaning of `anomaly_score` in this project?
5. How is `is_anomaly` decided?
6. Why is the anomaly threshold stored in model metadata?
7. Why is the Isolation Forest trained mainly using normal records?
8. What can cause a false positive anomaly in electricity data?
9. What can cause a false negative anomaly in electricity data?
10. How would you tune the contamination parameter?
11. Why might Isolation Forest be better than KNN for this project?
12. Why might an autoencoder be a possible future alternative?
13. How would you evaluate anomaly detection without perfect labels?
14. How does anomaly detection influence the final theft probability?
15. How would you explain Isolation Forest to a panel in simple language?

### 21.5 Theft Classification Questions

1. Why is theft detection treated as a supervised classification problem?
2. Why are Random Forest and XGBoost both used?
3. How does Random Forest work internally?
4. How does XGBoost work internally?
5. Why can boosted trees perform well on fraud-like tabular data?
6. What is the final theft probability formula in the project?
7. Why does the project give 55 percent weight to XGBoost and 45 percent to Random Forest in the initial blend?
8. Why are anomaly score, wastage score, and seeded theft probability added after model prediction?
9. Why is `Electricity Theft` assigned when theft probability is at least `0.9`?
10. How does the classifier distinguish between theft and power wastage?
11. What does `class_weight="balanced_subsample"` do in Random Forest?
12. How are precision, recall, F1 score, accuracy, and ROC-AUC used?
13. In this project, which metric matters more: precision or recall? Why?
14. What is the risk of high false positives in theft detection?
15. What is the risk of high false negatives in theft detection?
16. Why is `HistGradientBoostingClassifier` used as a fallback?
17. How would you improve theft classification with real customer billing data?
18. How would you handle class imbalance in real theft datasets?
19. Why should model probability be calibrated before operational use?
20. How would you explain a theft prediction to an inspector?

### 21.6 Forecasting Questions

1. Why is demand forecasting included in a theft detection project?
2. What is the difference between theft detection and demand forecasting?
3. Why is electricity demand a time-series problem?
4. How does the project build the demand series before training?
5. Why is a 24-hour lookback window used?
6. How does an LSTM remember sequence information?
7. What are the main layers in the LSTM model?
8. Why is MinMaxScaler used before LSTM training?
9. How does the project forecast next hour, next day, and next week demand?
10. What is recursive forecasting?
11. What problems can happen in recursive forecasting?
12. Why was a Transformer model added?
13. What is attention in a Transformer?
14. How is the project Transformer different from a large language model?
15. Why does the project include a baseline seasonal fallback?
16. When might a simple baseline outperform LSTM or Transformer?
17. How would weather forecast data improve demand forecasting?
18. How would you evaluate forecasting accuracy?
19. What are MAE, RMSE, and MAPE?
20. How would poor demand forecasts affect grid operations?

### 21.7 Risk Scoring Questions

1. Why is a risk score needed if the model already predicts theft probability?
2. Which signals are combined in `score_meter_risk`?
3. Why are anomaly and theft components heavily weighted?
4. Why are voltage irregularity and night usage included in risk scoring?
5. How are risk levels such as Low, Medium, High, and Critical assigned?
6. How would you decide the best threshold for Critical risk?
7. What is the difference between model probability and business risk?
8. How can risk scoring help prioritize field inspections?
9. What are the dangers of using a fixed risk threshold?
10. How would you calibrate risk scoring with feedback from inspectors?

### 21.8 Pole Monitoring Questions

1. Why is pole-level monitoring important in electricity theft detection?
2. What is the formula behind pole energy mismatch?
3. What is `energy_gap`?
4. What is `energy_gap_ratio`?
5. What are technical losses?
6. How can a pole show theft even when individual meters look normal?
7. How does the project detect possible illegal connections?
8. Why does pole tamper detection combine heuristics and Isolation Forest?
9. What is `tamper_probability`?
10. What conditions can set `tamper_flag` to 1?
11. How would real transformer and pole sensor data improve this module?
12. What are possible false positives in pole tamper detection?
13. How would you validate pole-level theft detection in the real world?
14. Why is `connected_meters` stored in the pole catalog?
15. How can pole alerts support field inspectors?

### 21.9 API and Runtime Questions

1. Why was FastAPI chosen for the backend?
2. What happens during backend startup?
3. Why does the API regenerate missing artifacts?
4. What is the purpose of the live simulation loop?
5. Why does the runtime advance one tick at startup?
6. What does `/health` return?
7. What does `/overview` return?
8. What does `/predict` do?
9. Why does `/predict` accept both one reading and a list of readings?
10. How does the WebSocket endpoint `/ws/live` work?
11. Why is CORS enabled?
12. What runtime data is stored in SQLite?
13. Why use SQLite instead of PostgreSQL for this project?
14. How would you scale this API for production?
15. How would you secure the API in a real deployment?
16. Why are generated artifacts exposed through artifact endpoints?
17. How does the API support inspector workflows?
18. What role does authentication play in the dashboard?
19. How would you handle API failure in the frontend?
20. How would you monitor this API in production?

### 21.10 Dashboard Questions

1. What is the purpose of the dashboard?
2. Which dashboard sections are available?
3. How does the dashboard get data from the backend?
4. Why does the dashboard show overview, live monitoring, theft, anomaly, forecast, efficiency, heatmap, and pole monitoring separately?
5. How would an electricity board operator use this dashboard?
6. How would an inspector use this dashboard?
7. Why is a heatmap useful for theft detection?
8. What KPIs should be shown on the overview page?
9. How would you prevent information overload in the dashboard?
10. How would you improve the dashboard for mobile inspectors?

### 21.11 Explainability and Drift Questions

1. Why is explainability important in electricity theft detection?
2. What is SHAP?
3. How does the fallback explanation method work when SHAP is unavailable?
4. Why should an inspector not trust only a black-box probability?
5. What is data drift?
6. What is concept drift?
7. Why can smart-meter data drift over time?
8. How does weather change create drift?
9. How can consumer behavior changes create drift?
10. What does Evidently provide?
11. Why is a fallback statistical drift check included?
12. How would drift affect model accuracy?
13. What should happen when drift is detected?
14. How often should the model be retrained?
15. How would inspector feedback improve future training?

### 21.12 Testing and Validation Questions

1. What parts of the project are tested with pytest?
2. Why is testing important in an ML application?
3. What should be tested in the data generation pipeline?
4. What should be tested in feature engineering?
5. What should be tested in theft probability calibration?
6. What should be tested in API responses?
7. What should be tested in the inspector workflow?
8. How do tests help prevent dashboard-breaking API changes?
9. Why should fallback behavior be tested?
10. What additional tests would you add before production deployment?
11. How would you test model performance on real data?
12. How would you test false positive and false negative behavior?
13. How would you validate heatmap accuracy?
14. How would you test WebSocket live updates?
15. How would you test alert delivery safely?

### 21.13 Deployment and Production Questions

1. How can this project be run locally?
2. How can it be run using Docker Compose?
3. What services are defined in `docker-compose.yml`?
4. What files are generated after running the project?
5. Which files should not be committed in a real production repository?
6. How would you deploy this project on a cloud server?
7. How would you replace synthetic data with real streaming data?
8. How would you store large smart-meter datasets?
9. Why might SQLite not be enough for production?
10. How would you use Kafka or another queue in this system?
11. How would you schedule periodic model retraining?
12. How would you handle model versioning?
13. How would you monitor model performance after deployment?
14. How would you protect customer privacy?
15. What security risks exist in a smart-grid theft detection system?

### 21.14 Advanced Machine Learning Questions

1. How would you handle severe class imbalance in real theft data?
2. How would you choose between Random Forest, XGBoost, LightGBM, and CatBoost?
3. How would you calibrate predicted probabilities?
4. What is threshold tuning, and why is it important here?
5. Why can accuracy be misleading in theft detection?
6. Why might recall be more important than precision in early theft screening?
7. Why might precision be more important before sending legal notices?
8. What is the difference between anomaly detection and fraud classification?
9. How would you use semi-supervised learning in this project?
10. How would you use active learning with inspector feedback?
11. How would you use graph neural networks for transformer-pole-meter topology?
12. How would you detect coordinated theft in an area?
13. How would you detect meter tampering from voltage and current waveforms?
14. How would you use smart-meter event logs in the model?
15. How would you reduce model bias across different areas or consumer types?

### 21.15 Scenario-Based Interview Questions

1. Suppose the model marks many industrial meters as theft during working hours. What would you check?
2. Suppose theft probability is low but pole energy gap is high. What does that mean?
3. Suppose the dashboard shows no live data. How would you debug it?
4. Suppose XGBoost is not installed on the evaluator's machine. What happens?
5. Suppose TensorFlow is missing. Will forecasting still work?
6. Suppose the generated dataset has very low theft rate. How does that affect training?
7. Suppose the model has high accuracy but low recall. Is it acceptable?
8. Suppose an inspector says many flagged cases are false positives. What changes would you make?
9. Suppose a new area is added to Bengaluru. What code or data must change?
10. Suppose real weather API calls fail. How does the project continue?
11. Suppose live data has missing `expected_consumption_kwh`. How does prediction handle it?
12. Suppose a meter has high night usage but belongs to a night-shift factory. How should the model avoid false positives?
13. Suppose voltage is abnormal for an entire area. Is it theft or grid fault?
14. Suppose pole tamper alerts are frequent after rain. What would you investigate?
15. Suppose demand forecasting suddenly becomes inaccurate. What drift or data issues would you check?

### 21.16 Questions You Should Be Ready To Answer Personally

1. What was your exact contribution to this project?
2. Which module was the most difficult to build and why?
3. Why did you choose this project topic?
4. What did you learn about smart grids?
5. What did you learn about anomaly detection?
6. What did you learn about full-stack ML systems?
7. Which model performed best and how do you know?
8. What would you improve if you had more time?
9. What are the ethical concerns of electricity theft prediction?
10. How would you make this project production-ready?
11. How would you explain a false accusation risk to the interview panel?
12. How would you collect real labels for theft?
13. How would field-inspection feedback be added to the system?
14. How would you convince an electricity board to trust this system?
15. What is the strongest technical feature of your project?


## 23. Roadmap improvements and evaluation tools

This version includes additional leakage-aware evaluation scripts, dataset provenance checks, a chronological forecasting baseline benchmark, experimental privacy primitives, a durable single-host SQLite job queue utility, browser security middleware, and a dashboard decision-support notice. See [`docs/ROADMAP_IMPLEMENTATION.md`](docs/ROADMAP_IMPLEMENTATION.md) for exact capabilities and remaining production gates.

### Evaluate a labeled theft dataset

The project does not bundle a large labeled theft dataset. The public SGCC research benchmark is described at https://ieee-pes-data-sharing.org/datasets/detail/663b8e0e-3ace-48f8-b983-4cb59b39fa6f; check its license before use. It is China-origin, not Indian data. The CEEW Mathura/Bareilly dataset is real Indian consumption data but does not have verified theft labels. Dataset notes are in [`data/labels/README.md`](data/labels/README.md).

```powershell
python scripts/validate_theft_dataset.py --input data/labels/sgcc.csv --manifest data/labels/sgcc.provenance.json
python scripts/evaluate_theft_models.py --input data/labels/sgcc.csv
python scripts/evaluate_forecasting.py --input data/processed/ceew/daily_aggregate/your.normalized.csv.gz
python scripts/check_model_compatibility.py
python -m pytest -q
```

The validation command checks structure and declared provenance; it cannot independently prove labels are true. The theft evaluation splits by customer and reports class-sensitive metrics. The forecasting benchmark uses chronological holdout data and compares simple baselines.
#   T r u s t - F e d - G r i d - X  
 