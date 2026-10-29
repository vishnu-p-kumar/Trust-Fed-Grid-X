from __future__ import annotations

import argparse

from src.data_generator import generate_smart_meter_data
from src.energy_efficiency import calculate_efficiency_metrics
from src.model_optimizer import optimize_detection_models
from src.preprocess import limit_theft_alerts, load_dataset
from src.report_generator import generate_daily_report
from src.risk_scoring import score_meter_risk
from src.sample_outputs import export_sample_outputs
from src.spatial_analysis import build_theft_heatmap
from src.theft_detector import classify_meter_events
from src.train_models import train_all_models
from utils.helpers import ensure_project_dirs, generation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Grid Electricity Theft Detection System")
    parser.add_argument("--full-scale", action="store_true", help="Generate the full 1000-meter, 1-year dataset.")
    parser.add_argument("--num-meters", type=int, default=None, help="Override number of meters.")
    parser.add_argument("--days", type=int, default=None, help="Override number of simulated days.")
    parser.add_argument("--skip-training", action="store_true", help="Generate data only.")
    parser.add_argument("--forecast-epochs", type=int, default=5, help="LSTM training epochs.")
    parser.add_argument("--skip-sample-export", action="store_true", help="Skip writing API reference sample outputs.")
    parser.add_argument("--skip-report", action="store_true", help="Skip writing the PDF analytics report.")
    parser.add_argument("--start-api", action="store_true", help="Start the FastAPI server after bootstrapping.")
    parser.add_argument("--optimize-models", action="store_true", help="Run the Optuna-based model optimization pass before training.")
    parser.add_argument("--optimization-trials", type=int, default=8, help="Number of optimization trials when --optimize-models is enabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_project_dirs()
    config = generation_config(full_scale=args.full_scale)
    if args.num_meters is not None:
        config["num_meters"] = args.num_meters
    if args.days is not None:
        config["days"] = args.days

    print("Generating synthetic smart meter data...")
    sample_df = generate_smart_meter_data(**config)
    print(f"Sample dataset ready: {len(sample_df):,} rows")

    if args.optimize_models:
        print("Optimizing model hyperparameters...")
        optimization_summary = optimize_detection_models(sample_df, trials=args.optimization_trials)
        print("Optimization summary:", optimization_summary)

    if not args.skip_training:
        print("Training anomaly, theft, and demand forecasting models...")
        summary = train_all_models(
            dataset_path=paths.data_processed / "smart_meter_sample.csv",
            max_rows=config["sample_rows"],
            forecast_epochs=args.forecast_epochs,
        )
        print("Training summary:", summary["classification"])

        live_frame = load_dataset(paths.live_dataset)
        predicted = calculate_efficiency_metrics(score_meter_risk(limit_theft_alerts(classify_meter_events(live_frame), max_alerts=2)))
        heatmap_path = build_theft_heatmap(predicted)
        if not args.skip_sample_export:
            sample_output_dir = export_sample_outputs(predicted, forecast=summary.get("forecasting", {}))
            print(f"Reference sample outputs written to {sample_output_dir}")
        if not args.skip_report:
            report_path = generate_daily_report(predicted, forecast=summary.get("forecasting", {}))
            print(f"Daily report written to {report_path}")
        print(f"Theft heatmap written to {heatmap_path}")

    if args.start_api:
        import uvicorn

        print("Starting FastAPI server on http://127.0.0.1:8000 ...")
        uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
