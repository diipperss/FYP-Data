import argparse
from pathlib import Path

import pandas as pd

from trading.run_simulation import (
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    build_features,
    build_model_factories,
    download_history,
    evaluate_universal_model,
    parse_float_list,
    parse_str_list,
    split_development_holdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export only the universal stock model bundle."
    )
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,NVDA",
        help="Comma-separated tickers used to train the universal model.",
    )
    parser.add_argument("--start", default="2018-01-01", help="Historical data start date.")
    parser.add_argument("--end", default="2025-12-31", help="Historical data end date.")
    parser.add_argument(
        "--market-ticker",
        default="SPY",
        help="Benchmark market ticker used for regime/context features.",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.2,
        help="Final chronological holdout fraction used only after model and policy selection.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation folds on the development set.",
    )
    parser.add_argument(
        "--calibration-splits",
        type=int,
        default=3,
        help="Maximum number of time-series splits for probability calibration.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon in trading days. Also used as the purge gap.",
    )
    parser.add_argument(
        "--bull-threshold",
        type=float,
        default=0.02,
        help="Future return threshold for BULLISH_SIGNAL.",
    )
    parser.add_argument(
        "--bear-threshold",
        type=float,
        default=-0.02,
        help="Future return threshold for BEARISH_SIGNAL.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Starting portfolio cash for each trader during tuning/evaluation.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=300,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--threshold-candidates",
        default="0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated bullish probability thresholds used for trader policy tuning.",
    )
    parser.add_argument(
        "--margin-candidates",
        default="0.00,0.03,0.05,0.07,0.10,0.15",
        help="Comma-separated probability margin thresholds used for trader tuning.",
    )
    parser.add_argument(
        "--allocation-scales",
        default="0.75,1.00,1.25",
        help="Comma-separated multipliers applied to each trader's base max allocation during tuning.",
    )
    parser.add_argument(
        "--neutral-options",
        default="hold,cash",
        help="Comma-separated neutral-signal behaviours to consider during tuning.",
    )
    parser.add_argument(
        "--output-dir",
        default="trading/output_v3",
        help="Directory for universal CSV results, saved model, and charts.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display chart windows in addition to saving the figures.",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.001,
        help="Commission cost applied to trade notional.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0005,
        help="Slippage cost applied to trade price.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["classification", "trading", "hybrid"],
        default="trading",
        help="How to rank candidate models during selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    threshold_candidates = parse_float_list(args.threshold_candidates)
    margin_candidates = parse_float_list(args.margin_candidates)
    allocation_scales = parse_float_list(args.allocation_scales)
    neutral_options = parse_str_list(args.neutral_options)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    market_benchmark_history = download_history(args.market_ticker, args.start, args.end)
    model_factories = build_model_factories(
        rf_estimators=args.rf_estimators,
        calibration_splits=args.calibration_splits,
        horizon=args.horizon,
    )

    availability_rows = [
        {"model": "xgboost", "available": HAS_XGBOOST},
        {"model": "lightgbm", "available": HAS_LIGHTGBM},
    ]
    pd.DataFrame(availability_rows).to_csv(output_dir / "optional_model_availability.csv", index=False)

    full_featured_by_ticker: dict[str, pd.DataFrame] = {}
    development_by_ticker: dict[str, pd.DataFrame] = {}
    holdout_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        print(f"[progress] universal-only: preparing {ticker}", flush=True)
        price_history = download_history(ticker, args.start, args.end)
        featured = build_features(
            price_history,
            benchmark_data=market_benchmark_history,
            horizon=args.horizon,
            bull_threshold=args.bull_threshold,
            bear_threshold=args.bear_threshold,
        )
        development, holdout = split_development_holdout(
            frame=featured,
            holdout_size=args.holdout_size,
            horizon=args.horizon,
        )
        full_featured_by_ticker[ticker] = featured.copy()
        development_by_ticker[ticker] = development.copy()
        holdout_by_ticker[ticker] = holdout.copy()

    print("\n========== UNIVERSAL MODEL ONLY ==========", flush=True)
    (
        universal_selection_table,
        universal_holdout_summary,
        universal_selected_models,
        universal_leave_one_ticker_out_summary,
    ) = evaluate_universal_model(
        model_factories=model_factories,
        full_featured_by_ticker=full_featured_by_ticker,
        development_by_ticker=development_by_ticker,
        holdout_by_ticker=holdout_by_ticker,
        threshold_candidates=threshold_candidates,
        margin_candidates=margin_candidates,
        allocation_scales=allocation_scales,
        neutral_options=neutral_options,
        args=args,
        output_dir=output_dir,
    )

    print("Universal model selection summary:")
    print(universal_selection_table.to_string(index=False))
    print("Universal holdout summary:")
    print(universal_holdout_summary.to_string(index=False))
    print("Universal leave-one-ticker-out summary:")
    print(universal_leave_one_ticker_out_summary.to_string(index=False))
    print("Universal selected models:")
    print(universal_selected_models.to_string(index=False))
    print(f"Saved universal outputs to: {output_dir / 'universal'}")


if __name__ == "__main__":
    main()
