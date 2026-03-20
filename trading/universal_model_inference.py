import argparse
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf


FEATURE_COLUMNS = [
    "return",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma5_gap",
    "ma20_gap",
    "ma50_gap",
    "volatility_10",
    "volume_change",
    "high_low_range",
    "overnight_gap",
    "rsi_14",
    "atr_14",
    "spy_return",
    "spy_ma50_gap",
    "excess_return",
    "relative_strength_20",
]

UNIVERSAL_ONLY_COLUMNS = {
    "atr_pct_14",
    "log_dollar_volume",
    "beta_60_spy",
    "volatility_regime_20_60",
}

TARGET_LABELS = {
    -1: "BEARISH_SIGNAL",
    0: "NEUTRAL_SIGNAL",
    1: "BULLISH_SIGNAL",
}


@dataclass(frozen=True)
class TraderProfile:
    name: str
    min_bull_probability: float
    max_allocation: float
    use_trend_filter: bool
    neutral_behavior: str = "hold"
    min_probability_margin: float = 0.0


BASE_TRADERS = [
    TraderProfile(
        name="conservative",
        min_bull_probability=0.70,
        max_allocation=0.25,
        use_trend_filter=True,
        neutral_behavior="cash",
        min_probability_margin=0.10,
    ),
    TraderProfile(
        name="balanced",
        min_bull_probability=0.60,
        max_allocation=0.50,
        use_trend_filter=True,
        neutral_behavior="hold",
        min_probability_margin=0.05,
    ),
    TraderProfile(
        name="aggressive",
        min_bull_probability=0.55,
        max_allocation=0.80,
        use_trend_filter=False,
        neutral_behavior="hold",
        min_probability_margin=0.00,
    ),
]


def download_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price history returned for ticker {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~flat_mask, 50.0)
    return rsi


def compute_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def build_features_for_inference(
    data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> pd.DataFrame:
    frame = data.copy()
    benchmark = benchmark_data.copy()

    benchmark["spy_return"] = benchmark["Close"].pct_change()
    benchmark["spy_ma50"] = benchmark["Close"].rolling(50).mean()
    benchmark["spy_ma50_gap"] = (benchmark["Close"] / benchmark["spy_ma50"]) - 1.0
    benchmark["spy_ret_20"] = benchmark["Close"].pct_change(20)
    benchmark = benchmark[["spy_return", "spy_ma50_gap", "spy_ret_20"]]
    frame = frame.join(benchmark, how="left")

    frame["return"] = frame["Close"].pct_change()
    frame["ret_3"] = frame["Close"].pct_change(3)
    frame["ret_5"] = frame["Close"].pct_change(5)
    frame["ret_10"] = frame["Close"].pct_change(10)
    frame["ret_20"] = frame["Close"].pct_change(20)
    frame["ma5"] = frame["Close"].rolling(5).mean()
    frame["ma20"] = frame["Close"].rolling(20).mean()
    frame["ma50"] = frame["Close"].rolling(50).mean()
    frame["ma5_gap"] = (frame["Close"] / frame["ma5"]) - 1.0
    frame["ma20_gap"] = (frame["Close"] / frame["ma20"]) - 1.0
    frame["ma50_gap"] = (frame["Close"] / frame["ma50"]) - 1.0
    frame["volatility_10"] = frame["return"].rolling(10).std()
    frame["volume_change"] = frame["Volume"].pct_change()
    frame["high_low_range"] = (frame["High"] - frame["Low"]) / frame["Close"]
    frame["overnight_gap"] = (frame["Open"] / frame["Close"].shift(1)) - 1.0
    frame["rsi_14"] = compute_rsi(frame["Close"], window=14)
    frame["atr_14"] = compute_atr(frame, window=14)
    frame["excess_return"] = frame["return"] - frame["spy_return"]
    frame["relative_strength_20"] = frame["ret_20"] - frame["spy_ret_20"]

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.dropna(subset=FEATURE_COLUMNS + ["ma50"]).copy()


def add_universal_generalization_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["Close"].replace(0, np.nan)
    enriched["atr_pct_14"] = enriched["atr_14"] / close
    dollar_volume = (enriched["Close"].clip(lower=0.0) * enriched["Volume"].clip(lower=0.0)).fillna(0.0)
    log_dollar_volume = np.log1p(dollar_volume)
    rolling_log_median = log_dollar_volume.rolling(60, min_periods=20).median()
    enriched["log_dollar_volume"] = log_dollar_volume - rolling_log_median

    rolling_cov = enriched["return"].rolling(60, min_periods=20).cov(enriched["spy_return"])
    rolling_var = enriched["spy_return"].rolling(60, min_periods=20).var()
    enriched["beta_60_spy"] = rolling_cov / rolling_var.replace(0, np.nan)
    enriched["beta_60_spy"] = enriched["beta_60_spy"].clip(-5.0, 5.0)

    volatility_20 = enriched["return"].rolling(20, min_periods=10).std()
    volatility_60 = enriched["return"].rolling(60, min_periods=20).std()
    enriched["volatility_regime_20_60"] = (volatility_20 / volatility_60.replace(0, np.nan)) - 1.0

    derived_columns = [
        "atr_pct_14",
        "log_dollar_volume",
        "beta_60_spy",
        "volatility_regime_20_60",
    ]
    enriched[derived_columns] = enriched[derived_columns].replace([np.inf, -np.inf], np.nan)
    return enriched.dropna(subset=derived_columns).copy()


def load_model_bundle(path: str | Path) -> dict:
    model_bundle = joblib.load(path)
    if model_bundle.get("ticker") != "universal":
        raise ValueError(f"Expected a universal model bundle, got {model_bundle.get('ticker')!r}.")
    return model_bundle


def predict_with_probabilities(model: object, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        columns = getattr(model, "classes_", sorted(np.unique(predictions)))
        probability_frame = pd.DataFrame(probabilities, columns=columns, index=X.index)
    else:
        classes = np.array(sorted(np.unique(predictions)))
        probability_frame = pd.DataFrame(0.0, index=X.index, columns=classes)
        for class_value in classes:
            probability_frame.loc[predictions == class_value, class_value] = 1.0

    for class_value in TARGET_LABELS:
        if class_value not in probability_frame.columns:
            probability_frame[class_value] = 0.0

    probability_frame = probability_frame[sorted(probability_frame.columns)]
    return predictions, probability_frame


def prepare_latest_inference_frame(
    model_bundle: dict,
    price_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> pd.DataFrame:
    feature_frame = build_features_for_inference(
        data=price_history,
        benchmark_data=benchmark_history,
    )
    feature_columns = model_bundle["feature_columns"]

    if any(column in UNIVERSAL_ONLY_COLUMNS for column in feature_columns):
        feature_frame = add_universal_generalization_features(feature_frame)

    missing_columns = [column for column in feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(
            "Could not build all required features for inference. "
            f"Missing columns: {missing_columns}"
        )

    if feature_frame.empty:
        raise ValueError("No valid rows available for inference after feature construction.")

    return feature_frame


def predict_latest_signal(
    model_bundle: dict,
    latest_inference_frame: pd.DataFrame,
) -> dict:
    feature_columns = model_bundle["feature_columns"]
    latest_feature_row = latest_inference_frame.tail(1)[feature_columns].copy()
    pred_values, probability_frame = predict_with_probabilities(model_bundle["model"], latest_feature_row)
    sorted_probs = np.sort(probability_frame.to_numpy(), axis=1)
    predicted_label = int(pred_values[0])
    return {
        "predicted_signal": TARGET_LABELS[predicted_label],
        "confidence": float(probability_frame.max(axis=1).iloc[0]),
        "bullish_probability": float(probability_frame[1].iloc[0]),
        "probability_margin": float(sorted_probs[:, -1][0] - sorted_probs[:, -2][0]),
    }


def compute_strength(probability: float, min_probability: float) -> float:
    if probability <= min_probability:
        return 0.0
    return max(0.0, min(1.0, (probability - min_probability) / (1.0 - min_probability)))


def decide_action(
    trader: TraderProfile,
    latest_inference_frame: pd.DataFrame,
    signal: dict,
) -> dict:
    latest_row = latest_inference_frame.iloc[-1]
    trend_passed = (not trader.use_trend_filter) or bool(latest_row["Close"] > latest_row["ma50"])
    signal_is_actionable = (
        signal["bullish_probability"] >= trader.min_bull_probability
        and signal["probability_margin"] >= trader.min_probability_margin
    )

    if signal["predicted_signal"] == TARGET_LABELS[1] and trend_passed and signal_is_actionable:
        strength = compute_strength(signal["bullish_probability"], trader.min_bull_probability)
        target_allocation = trader.max_allocation * strength
        action = "BUY"
        reason = "bullish_signal"
    elif signal["predicted_signal"] == TARGET_LABELS[-1]:
        target_allocation = 0.0
        action = "SELL"
        reason = "bearish_exit"
    else:
        target_allocation = 0.0 if trader.neutral_behavior == "cash" else None
        action = "HOLD"
        reason = "neutral_cash" if trader.neutral_behavior == "cash" else "neutral_hold"

    return {
        "action": action,
        "trend_filter_passed": trend_passed,
        "signal_is_actionable": signal_is_actionable,
        "target_allocation": target_allocation,
        "latest_close": float(latest_row["Close"]),
        "latest_ma50": float(latest_row["ma50"]),
        "decision_reason": reason,
        "explanation": (
            f"{trader.name.title()} trader saw {signal['predicted_signal']} with "
            f"bullish_probability={signal['bullish_probability']:.2f}, "
            f"confidence={signal['confidence']:.2f}, "
            f"probability_margin={signal['probability_margin']:.2f}, "
            f"trend_filter_passed={trend_passed}, rule={reason}."
        ),
    }


def get_universal_trade_decision(
    model_path: str | Path,
    ticker: str,
    personality: str,
    market_ticker: str = "SPY",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    trader_lookup = {profile.name: profile for profile in BASE_TRADERS}
    if personality not in trader_lookup:
        raise ValueError(f"Unknown personality {personality!r}. Expected one of: {sorted(trader_lookup)}")

    end_ts = pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start).normalize() if start else (end_ts - timedelta(days=550))

    model_bundle = load_model_bundle(model_path)
    price_history = download_history(ticker, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    benchmark_history = download_history(market_ticker, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    latest_inference_frame = prepare_latest_inference_frame(model_bundle, price_history, benchmark_history)
    signal = predict_latest_signal(model_bundle, latest_inference_frame)
    decision = decide_action(trader_lookup[personality], latest_inference_frame, signal)

    return {
        "ticker": ticker,
        "market_ticker": market_ticker,
        "personality": personality,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "history_start": start_ts.strftime("%Y-%m-%d"),
        "history_end": end_ts.strftime("%Y-%m-%d"),
        "feature_date": latest_inference_frame.index[-1].strftime("%Y-%m-%d"),
        "profile": asdict(trader_lookup[personality]),
        **signal,
        **decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal stock model inference with trader personality rules.")
    parser.add_argument("--model-path", required=True, help="Path to universal_stock_model.joblib")
    parser.add_argument("--ticker", required=True, help="Ticker to score, for example AAPL")
    parser.add_argument(
        "--personality",
        choices=[profile.name for profile in BASE_TRADERS],
        default="balanced",
        help="Trader personality to apply on top of the universal model signal.",
    )
    parser.add_argument("--market-ticker", default="SPY", help="Benchmark ticker used for market context features.")
    parser.add_argument("--start", default=None, help="Optional history start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Optional history end date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = get_universal_trade_decision(
        model_path=args.model_path,
        ticker=args.ticker,
        personality=args.personality,
        market_ticker=args.market_ticker,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
