import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False


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

TARGET_LABELS = {
    -1: "BEARISH_SIGNAL",
    0: "NEUTRAL_SIGNAL",
    1: "BULLISH_SIGNAL",
}

BENCHMARK_ONLY_MODELS = {"dummy_most_frequent"}


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


class EncodedTargetClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_series = pd.Series(y)
        self.classes_ = np.array(sorted(y_series.unique()))
        mapping = {label: i for i, label in enumerate(self.classes_)}
        y_encoded = y_series.map(mapping).astype(int).to_numpy()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_encoded)
        return self

    def predict(self, X):
        encoded_predictions = np.asarray(self.estimator_.predict(X), dtype=int)
        return self.classes_[encoded_predictions]

    def predict_proba(self, X):
        probabilities = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        if probabilities.shape[1] != len(self.classes_):
            aligned = np.zeros((len(X), len(self.classes_)))
            width = min(probabilities.shape[1], len(self.classes_))
            aligned[:, :width] = probabilities[:, :width]
            return aligned
        return probabilities


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware walk-forward stock signal training, calibration, threshold tuning, and AI trader simulation."
    )
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,NVDA",
        help="Comma-separated tickers to download from Yahoo Finance.",
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
        help="Future return threshold for BULLISH_SIGNAL, e.g. 0.02 = +2%% over horizon.",
    )
    parser.add_argument(
        "--bear-threshold",
        type=float,
        default=-0.02,
        help="Future return threshold for BEARISH_SIGNAL, e.g. -0.02 = -2%% over horizon.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Starting portfolio cash for each trader.",
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
        help="Directory for CSV results, saved models, and charts.",
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
        help="Commission cost applied to trade notional, e.g. 0.001 = 0.1%%.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0005,
        help="Slippage cost applied to trade price, e.g. 0.0005 = 0.05%%.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["classification", "trading", "hybrid"],
        default="trading",
        help="How to rank candidate models during selection.",
    )
    return parser.parse_args()


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


def build_features(
    data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    horizon: int,
    bull_threshold: float,
    bear_threshold: float,
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

    future_return = (frame["Close"].shift(-horizon) / frame["Close"]) - 1.0
    frame["future_return_horizon"] = future_return
    frame["target"] = 0
    frame.loc[future_return >= bull_threshold, "target"] = 1
    frame.loc[future_return <= bear_threshold, "target"] = -1
    move_scale = max(abs(bull_threshold), abs(bear_threshold), 0.01)
    frame["sample_weight"] = 1.0 + np.clip(np.abs(frame["future_return_horizon"]) / move_scale, 0.0, 5.0)

    frame = frame[future_return.notna()].copy()
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.dropna(subset=FEATURE_COLUMNS + ["target", "ma50"]).copy()


def split_development_holdout(
    frame: pd.DataFrame,
    holdout_size: float,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - holdout_size))
    development_end = split_index - horizon
    if development_end <= 0 or split_index >= len(frame):
        raise ValueError("Holdout split produced an empty development or test set.")
    development = frame.iloc[:development_end].copy()
    holdout = frame.iloc[split_index:].copy()
    return development, holdout


def make_calibrated_estimator(
    estimator_factory: Callable[[], object],
    sample_count: int,
    calibration_splits: int,
    horizon: int,
) -> object:
    base_estimator = estimator_factory()

    # sklearn's calibration utility does not reliably recognize this thin wrapper
    # as a classifier across versions, so keep these models uncalibrated.
    if isinstance(base_estimator, EncodedTargetClassifier):
        return base_estimator

    if sample_count < 300:
        return base_estimator

    max_splits = min(calibration_splits, max(2, min(5, sample_count // 120)))
    if sample_count <= max_splits + horizon + 1:
        return base_estimator

    return CalibratedClassifierCV(
        estimator=base_estimator,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=max_splits, gap=horizon),
    )


def build_model_factories(
    rf_estimators: int,
    calibration_splits: int,
    horizon: int,
) -> dict[str, Callable[[int], object]]:
    def logistic_factory(_: int) -> object:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    def random_forest_factory(_: int) -> object:
        return RandomForestClassifier(
            n_estimators=rf_estimators,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    def calibrated_random_forest_factory(sample_count: int) -> object:
        return make_calibrated_estimator(
            estimator_factory=lambda: RandomForestClassifier(
                n_estimators=rf_estimators,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            sample_count=sample_count,
            calibration_splits=calibration_splits,
            horizon=horizon,
        )

    def hist_gb_factory(_: int) -> object:
        return HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=20,
            random_state=42,
        )

    def calibrated_hist_gb_factory(sample_count: int) -> object:
        return make_calibrated_estimator(
            estimator_factory=lambda: HistGradientBoostingClassifier(
                max_depth=5,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=20,
                random_state=42,
            ),
            sample_count=sample_count,
            calibration_splits=calibration_splits,
            horizon=horizon,
        )

    factories: dict[str, Callable[[int], object]] = {
        "dummy_most_frequent": lambda _: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": logistic_factory,
        "random_forest": random_forest_factory,
        "calibrated_random_forest": calibrated_random_forest_factory,
        "hist_gradient_boosting": hist_gb_factory,
        "calibrated_hist_gradient_boosting": calibrated_hist_gb_factory,
    }

    if HAS_XGBOOST:
        def xgboost_factory(_: int) -> object:
            return EncodedTargetClassifier(
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    eval_metric="mlogloss",
                )
            )

        factories["xgboost"] = xgboost_factory

    if HAS_LIGHTGBM:
        def lightgbm_factory(_: int) -> object:
            return EncodedTargetClassifier(
                LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    class_weight="balanced",
                    verbose=-1,
                )
            )

        factories["lightgbm"] = lightgbm_factory

    return factories


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


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def fit_with_optional_sample_weight(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
) -> object:
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return model
    except (TypeError, ValueError):
        pass

    if isinstance(model, Pipeline):
        last_step_name = model.steps[-1][0]
        try:
            model.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weight})
            return model
        except (TypeError, ValueError):
            pass

    model.fit(X, y)
    return model


def walk_forward_validate(
    model_factory: Callable[[int], object],
    development: pd.DataFrame,
    n_splits: int,
    horizon: int,
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(development), start=1):
        if progress_label is not None:
            print(
                f"[progress] {progress_label}: fold {fold_number}/{n_splits}",
                flush=True,
            )
        train = development.iloc[train_idx].copy()
        valid = development.iloc[valid_idx].copy()

        model = model_factory(len(train))
        model = fit_with_optional_sample_weight(
            model=model,
            X=train[FEATURE_COLUMNS],
            y=train["target"],
            sample_weight=train["sample_weight"],
        )

        pred_values, probability_frame = predict_with_probabilities(model, valid[FEATURE_COLUMNS])
        prediction_confidence = probability_frame.max(axis=1)
        bullish_probability = probability_frame[1]
        sorted_probs = np.sort(probability_frame.to_numpy(), axis=1)
        second_best_probability = sorted_probs[:, -2]
        probability_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        metrics = evaluate_predictions(valid["target"], pd.Series(pred_values, index=valid.index))
        fold_rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train),
                "valid_rows": len(valid),
                **metrics,
            }
        )

        fold_prediction_frame = pd.DataFrame(
            {
                "fold": fold_number,
                "date": valid.index,
                "y_true": valid["target"].values,
                "y_pred": pred_values,
                "confidence": prediction_confidence.values,
                "bullish_probability": bullish_probability.values,
                "second_best_probability": second_best_probability,
                "probability_margin": probability_margin,
            }
        )
        prediction_rows.extend(fold_prediction_frame.to_dict(orient="records"))

    fold_metrics = pd.DataFrame(fold_rows)
    cv_predictions = pd.DataFrame(prediction_rows).sort_values("date").reset_index(drop=True)
    overall_metrics = evaluate_predictions(cv_predictions["y_true"], cv_predictions["y_pred"])
    overall_summary = pd.DataFrame([overall_metrics])
    return fold_metrics, cv_predictions, overall_summary


def fit_final_model(
    model_factory: Callable[[int], object],
    development: pd.DataFrame,
) -> object:
    model = model_factory(len(development))
    model = fit_with_optional_sample_weight(
        model=model,
        X=development[FEATURE_COLUMNS],
        y=development["target"],
        sample_weight=development["sample_weight"],
    )
    return model


def evaluate_on_holdout(
    model: object,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], str, pd.DataFrame]:
    pred_values, probability_frame = predict_with_probabilities(model, holdout[FEATURE_COLUMNS])
    confidence = probability_frame.max(axis=1)
    bullish_probability = probability_frame[1]
    sorted_probs = np.sort(probability_frame.to_numpy(), axis=1)
    second_best_probability = sorted_probs[:, -2]
    probability_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    predictions = pd.DataFrame(
        {
            "date": holdout.index,
            "y_true": holdout["target"].values,
            "y_pred": pred_values,
            "confidence": confidence.values,
            "bullish_probability": bullish_probability.values,
            "second_best_probability": second_best_probability,
            "probability_margin": probability_margin,
        }
    )

    metrics = evaluate_predictions(holdout["target"], predictions["y_pred"])
    report = classification_report(
        holdout["target"],
        predictions["y_pred"],
        labels=[-1, 0, 1],
        target_names=[TARGET_LABELS[-1], TARGET_LABELS[0], TARGET_LABELS[1]],
        zero_division=0,
    )
    confusion = pd.DataFrame(
        confusion_matrix(holdout["target"], predictions["y_pred"], labels=[-1, 0, 1]),
        index=[f"actual_{TARGET_LABELS[label]}" for label in [-1, 0, 1]],
        columns=[f"pred_{TARGET_LABELS[label]}" for label in [-1, 0, 1]],
    )
    return predictions, metrics, report, confusion


def can_show_plot() -> bool:
    return "agg" not in matplotlib.get_backend().lower()


def compute_strength(probability: float, min_probability: float) -> float:
    if probability <= min_probability:
        return 0.0
    return max(0.0, min(1.0, (probability - min_probability) / (1.0 - min_probability)))


def compute_max_drawdown_pct(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    running_peak = values.cummax()
    drawdown = (values / running_peak) - 1.0
    return abs(float(drawdown.min())) * 100.0


def compute_sharpe_like_ratio(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    returns = values.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_sharpe = returns.mean() / returns.std()
    return float(daily_sharpe * np.sqrt(252))


def compute_trading_selection_score(summary: dict) -> float:
    score = (
        0.10 * float(summary["excess_return_vs_bh"])
        + 1.00 * float(summary["sharpe_like"])
        - 0.12 * float(summary["max_drawdown_pct"])
    )

    num_trades = int(summary["num_trades"])

    if num_trades < 5:
        score -= 1.0

    if num_trades > 100:
        score -= 0.01 * (num_trades - 100)

    return round(score, 6)


def filter_trading_selection_candidates(selection_table: pd.DataFrame) -> pd.DataFrame:
    candidates = selection_table[~selection_table["model"].isin(BENCHMARK_ONLY_MODELS)].copy()
    if candidates.empty:
        candidates = selection_table.copy()

    beating_benchmark = candidates[candidates["avg_trader_excess_return_vs_bh"] > 0].copy()
    if not beating_benchmark.empty:
        candidates = beating_benchmark

    positive_sharpe = candidates[candidates["avg_trader_sharpe"] > 0].copy()
    if not positive_sharpe.empty:
        candidates = positive_sharpe

    return candidates


def build_global_basket_trading_ranking(cv_summary: pd.DataFrame) -> pd.DataFrame:
    if cv_summary.empty:
        return pd.DataFrame()

    ranking_source = cv_summary[~cv_summary["model"].isin(BENCHMARK_ONLY_MODELS)].copy()
    if ranking_source.empty:
        ranking_source = cv_summary.copy()

    grouped = ranking_source.groupby("model")
    ranking = grouped.agg(
        ticker_count=("ticker", "nunique"),
        mean_trading_score=("trading_score", "mean"),
        mean_excess_return_vs_bh=("avg_trader_excess_return_vs_bh", "mean"),
        median_excess_return_vs_bh=("avg_trader_excess_return_vs_bh", "median"),
        mean_sharpe_like=("avg_trader_sharpe", "mean"),
        mean_drawdown_pct=("avg_trader_drawdown_pct", "mean"),
        positive_excess_hit_rate=("avg_trader_excess_return_vs_bh", lambda s: float((s > 0).mean())),
        mean_macro_f1=("macro_f1", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
    ).reset_index()

    ranking["basket_selection_score"] = (
        0.35 * ranking["mean_excess_return_vs_bh"]
        + 0.30 * ranking["median_excess_return_vs_bh"]
        + 12.0 * ranking["positive_excess_hit_rate"]
        + 2.0 * ranking["mean_sharpe_like"]
        - 0.20 * ranking["mean_drawdown_pct"]
    )

    ranking["basket_selection_score"] = ranking["basket_selection_score"].round(6)
    ranking["mean_trading_score"] = ranking["mean_trading_score"].round(6)
    ranking["mean_excess_return_vs_bh"] = ranking["mean_excess_return_vs_bh"].round(4)
    ranking["median_excess_return_vs_bh"] = ranking["median_excess_return_vs_bh"].round(4)
    ranking["mean_sharpe_like"] = ranking["mean_sharpe_like"].round(4)
    ranking["mean_drawdown_pct"] = ranking["mean_drawdown_pct"].round(4)
    ranking["positive_excess_hit_rate"] = ranking["positive_excess_hit_rate"].round(4)
    ranking["mean_macro_f1"] = ranking["mean_macro_f1"].round(4)
    ranking["mean_balanced_accuracy"] = ranking["mean_balanced_accuracy"].round(4)

    return ranking.sort_values(
        [
            "basket_selection_score",
            "median_excess_return_vs_bh",
            "mean_excess_return_vs_bh",
            "positive_excess_hit_rate",
            "mean_sharpe_like",
            "mean_drawdown_pct",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def apply_buy_trade(
    cash: float,
    shares: int,
    shares_to_buy: int,
    trade_price: float,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[float, int, int]:
    if shares_to_buy <= 0 or trade_price <= 0:
        return cash, shares, 0

    effective_price = trade_price * (1.0 + slippage_rate)
    max_affordable = int(cash / (effective_price * (1.0 + commission_rate)))
    executed_shares = min(shares_to_buy, max_affordable)
    if executed_shares <= 0:
        return cash, shares, 0

    trade_notional = executed_shares * effective_price
    commission = trade_notional * commission_rate
    cash -= trade_notional + commission
    shares += executed_shares
    return cash, shares, executed_shares


def apply_sell_trade(
    cash: float,
    shares: int,
    shares_to_sell: int,
    trade_price: float,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[float, int, int]:
    if shares_to_sell <= 0 or trade_price <= 0:
        return cash, shares, 0

    executed_shares = min(shares_to_sell, shares)
    if executed_shares <= 0:
        return cash, shares, 0

    effective_price = trade_price * (1.0 - slippage_rate)
    trade_notional = executed_shares * effective_price
    commission = trade_notional * commission_rate
    cash += trade_notional - commission
    shares -= executed_shares
    return cash, shares, executed_shares


def simulate_trader(
    trader: TraderProfile,
    market_frame: pd.DataFrame,
    signal_predictions: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, float]:
    if market_frame.empty or signal_predictions.empty:
        return pd.DataFrame(), initial_cash

    cash = initial_cash
    shares = 0
    rebalance_band = 0.05
    rows: list[dict] = []

    signal_lookup = signal_predictions.set_index("date").sort_index()
    market_positions = {timestamp: position for position, timestamp in enumerate(market_frame.index)}

    for signal_date, signal in signal_lookup.iterrows():
        if signal_date not in market_positions:
            continue

        signal_position = market_positions[signal_date]
        if signal_position + 1 >= len(market_frame):
            continue

        signal_row = market_frame.iloc[signal_position]
        trade_date = market_frame.index[signal_position + 1]
        trade_row = market_frame.iloc[signal_position + 1]

        predicted_label = int(signal["y_pred"])
        bullish_probability = float(signal["bullish_probability"])
        confidence = float(signal["confidence"])
        probability_margin = float(signal["probability_margin"])
        trend_passed = (not trader.use_trend_filter) or bool(signal_row["Close"] > signal_row["ma50"])

        trade_price = float(trade_row["Open"])
        market_close = float(trade_row["Close"])

        current_position_value_open = shares * trade_price
        portfolio_value_open = cash + current_position_value_open
        current_allocation = (
            current_position_value_open / portfolio_value_open if portfolio_value_open > 0 else 0.0
        )

        signal_is_actionable = (
            bullish_probability >= trader.min_bull_probability
            and probability_margin >= trader.min_probability_margin
        )

        if predicted_label == 1 and trend_passed and signal_is_actionable:
            strength = compute_strength(bullish_probability, trader.min_bull_probability)
            target_allocation = trader.max_allocation * strength
            reason = "bullish_signal"
        elif predicted_label == -1:
            target_allocation = 0.0
            reason = "bearish_exit"
        else:
            target_allocation = current_allocation if trader.neutral_behavior == "hold" else 0.0
            reason = "neutral_hold" if trader.neutral_behavior == "hold" else "neutral_cash"

        if abs(target_allocation - current_allocation) < rebalance_band:
            target_shares = shares
        else:
            target_position_value = portfolio_value_open * target_allocation
            target_shares = int(target_position_value // trade_price) if trade_price > 0 else 0

        if target_shares > shares:
            desired_buy = target_shares - shares
            cash, shares, executed_shares = apply_buy_trade(
                cash=cash,
                shares=shares,
                shares_to_buy=desired_buy,
                trade_price=trade_price,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )
            action = "BUY" if executed_shares > 0 else "HOLD"
        elif target_shares < shares:
            desired_sell = shares - target_shares
            cash, shares, executed_shares = apply_sell_trade(
                cash=cash,
                shares=shares,
                shares_to_sell=desired_sell,
                trade_price=trade_price,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )
            action = "EXIT TO CASH" if executed_shares > 0 and target_shares == 0 else "REDUCE"
            if executed_shares == 0:
                action = "HOLD"
        else:
            action = "HOLD"

        position_value_close = shares * market_close
        portfolio_value_close = cash + position_value_close

        rows.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "trade_price": trade_price,
                "market_close": market_close,
                "predicted_signal": TARGET_LABELS[predicted_label],
                "confidence": confidence,
                "bullish_probability": bullish_probability,
                "trend_filter_passed": trend_passed,
                "target_allocation": target_allocation,
                "action": action,
                "cash": cash,
                "shares": shares,
                "position_value": position_value_close,
                "portfolio_value": portfolio_value_close,
                "explanation": (
                    f"{trader.name.title()} trader saw {TARGET_LABELS[predicted_label]} with "
                    f"bullish_probability={bullish_probability:.2f}, confidence={confidence:.2f}, "
                    f"probability_margin={probability_margin:.2f}, "
                    f"trend_filter_passed={trend_passed}, rule={reason}, "
                    f"target_allocation={target_allocation:.2%}."
                ),
            }
        )

    history = pd.DataFrame(rows)
    final_value = float(history["portfolio_value"].iloc[-1]) if not history.empty else initial_cash
    return history, final_value


def simulate_buy_and_hold(
    market_frame: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, float]:
    if len(market_frame) < 2:
        return pd.DataFrame(), initial_cash

    entry_row = market_frame.iloc[1]
    entry_price = float(entry_row["Open"])
    cash = initial_cash
    shares = 0
    cash, shares, bought_shares = apply_buy_trade(
        cash=cash,
        shares=shares,
        shares_to_buy=int(initial_cash // entry_price) if entry_price > 0 else 0,
        trade_price=entry_price,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    rows: list[dict] = []
    for position in range(1, len(market_frame)):
        trade_date = market_frame.index[position]
        market_close = float(market_frame.iloc[position]["Close"])
        position_value = shares * market_close
        portfolio_value = cash + position_value
        rows.append(
            {
                "signal_date": market_frame.index[position - 1],
                "trade_date": trade_date,
                "trade_price": entry_price if position == 1 else float(market_frame.iloc[position]["Open"]),
                "market_close": market_close,
                "predicted_signal": "BUY_AND_HOLD",
                "confidence": 1.0,
                "bullish_probability": 1.0,
                "trend_filter_passed": True,
                "target_allocation": 1.0,
                "action": "BUY" if position == 1 and bought_shares > 0 else "HOLD",
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "portfolio_value": portfolio_value,
                "explanation": "Benchmark entered at next-day open with trading costs and then held.",
            }
        )

    history = pd.DataFrame(rows)
    final_value = float(history["portfolio_value"].iloc[-1]) if not history.empty else initial_cash
    return history, final_value


def summarize_history(trader_name: str, history: pd.DataFrame, initial_cash: float) -> dict:
    final_value = float(history["portfolio_value"].iloc[-1]) if not history.empty else initial_cash
    return {
        "trader": trader_name,
        "final_portfolio_value": round(final_value, 2),
        "return_pct": round(((final_value / initial_cash) - 1.0) * 100, 2),
        "num_days": int(len(history)),
        "num_trades": int((history["action"] != "HOLD").sum()) if "action" in history.columns else 0,
        "max_drawdown_pct": round(compute_max_drawdown_pct(history["portfolio_value"]), 2)
        if not history.empty
        else 0.0,
        "sharpe_like": round(compute_sharpe_like_ratio(history["portfolio_value"]), 3)
        if not history.empty
        else 0.0,
    }


def tune_trader_profile(
    base_profile: TraderProfile,
    market_frame: pd.DataFrame,
    signal_predictions: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
    threshold_candidates: list[float],
    allocation_scales: list[float],
    neutral_options: list[str],
    margin_candidates: list[float],
) -> tuple[TraderProfile, pd.DataFrame]:
    rows: list[dict] = []
    benchmark_history, _ = simulate_buy_and_hold(
        market_frame=market_frame,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    benchmark_summary = summarize_history("buy_and_hold", benchmark_history, initial_cash)
    benchmark_return = float(benchmark_summary["return_pct"])

    for min_probability in threshold_candidates:
        for min_margin in margin_candidates:
            for scale in allocation_scales:
                tuned_allocation = max(0.05, min(1.0, base_profile.max_allocation * scale))
                for neutral_behavior in neutral_options:
                    candidate = replace(
                        base_profile,
                        min_bull_probability=min_probability,
                        max_allocation=tuned_allocation,
                        neutral_behavior=neutral_behavior,
                        min_probability_margin=min_margin,
                    )
                    history, _ = simulate_trader(
                        trader=candidate,
                        market_frame=market_frame,
                        signal_predictions=signal_predictions,
                        initial_cash=initial_cash,
                        commission_rate=commission_rate,
                        slippage_rate=slippage_rate,
                    )
                    summary = summarize_history(candidate.name, history, initial_cash)
                    summary["excess_return_vs_bh"] = round(
                        float(summary["return_pct"]) - benchmark_return,
                        4,
                    )
                    summary["selection_score"] = compute_trading_selection_score(summary)
                    summary.update(
                        {
                            "min_bull_probability": round(min_probability, 4),
                            "min_probability_margin": round(min_margin, 4),
                            "max_allocation": round(tuned_allocation, 4),
                            "neutral_behavior": neutral_behavior,
                        }
                    )
                    rows.append(summary)

    tuning_results = pd.DataFrame(rows).sort_values(
        ["selection_score", "excess_return_vs_bh", "sharpe_like", "max_drawdown_pct", "num_trades"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    best = tuning_results.iloc[0]
    tuned_profile = replace(
        base_profile,
        min_bull_probability=float(best["min_bull_probability"]),
        min_probability_margin=float(best["min_probability_margin"]),
        max_allocation=float(best["max_allocation"]),
        neutral_behavior=str(best["neutral_behavior"]),
    )
    return tuned_profile, tuning_results


def plot_histories(
    histories: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
    show_plot: bool,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, history in histories.items():
        if history.empty:
            continue
        plt.plot(history["trade_date"], history["portfolio_value"], label=name.title())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    if show_plot and can_show_plot():
        plt.show()
    plt.close()


def resolve_underlying_estimator(model: object) -> object:
    if isinstance(model, Pipeline):
        return resolve_underlying_estimator(model.named_steps["clf"])
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        base_estimators = []
        for calibrated in model.calibrated_classifiers_:
            estimator = getattr(calibrated, "estimator", None)
            if estimator is None:
                estimator = getattr(calibrated, "base_estimator", None)
            if estimator is not None:
                base_estimators.append(estimator)
        if base_estimators:
            return resolve_underlying_estimator(base_estimators[0])
    if isinstance(model, EncodedTargetClassifier) and hasattr(model, "estimator_"):
        return resolve_underlying_estimator(model.estimator_)
    return model


def export_model_importance(model: object, output_dir: Path) -> pd.DataFrame:
    estimator = resolve_underlying_estimator(model)
    importance = pd.DataFrame(columns=["feature", "importance"])

    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": np.asarray(estimator.feature_importances_, dtype=float),
            }
        )
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 2:
            values = np.mean(np.abs(coefficients), axis=0)
        else:
            values = np.abs(coefficients)
        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": values,
            }
        )

    if not importance.empty:
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        importance.to_csv(output_dir / "best_model_importance.csv", index=False)

    return importance


def save_model_bundle(
    ticker: str,
    model: object,
    output_dir: Path,
    metadata: dict,
) -> Path:
    bundle_path = output_dir / f"{ticker}_best_model.joblib"
    joblib.dump(
        {
            "ticker": ticker,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_labels": TARGET_LABELS,
            "metadata": metadata,
        },
        bundle_path,
    )
    return bundle_path


def load_model_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def predict_latest_signal(model_bundle: dict, latest_feature_row: pd.DataFrame) -> dict:
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    pred_values, probability_frame = predict_with_probabilities(model, latest_feature_row[feature_columns])
    sorted_probs = np.sort(probability_frame.to_numpy(), axis=1)
    predicted_label = int(pred_values[0])
    return {
        "predicted_signal": TARGET_LABELS[predicted_label],
        "confidence": float(probability_frame.max(axis=1).iloc[0]),
        "bullish_probability": float(probability_frame[1].iloc[0]),
        "probability_margin": float(sorted_probs[:, -1][0] - sorted_probs[:, -2][0]),
    }


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

    all_cv_rows: list[dict] = []
    all_holdout_rows: list[dict] = []
    ticker_selection_rows: list[dict] = []

    for ticker in tickers:
        print(f"\n========== {ticker} ==========")
        print(
            f"[progress] {ticker}: downloading data and building features",
            flush=True,
        )
        price_history = download_history(ticker, args.start, args.end)
        featured = build_features(
            price_history,
            benchmark_data=market_benchmark_history,
            horizon=args.horizon,
            bull_threshold=args.bull_threshold,
            bear_threshold=args.bear_threshold,
        )
        development, holdout = split_development_holdout(
            featured,
            holdout_size=args.holdout_size,
            horizon=args.horizon,
        )

        ticker_dir = output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        selection_rows: list[dict] = []
        cv_predictions_by_model: dict[str, pd.DataFrame] = {}
        tuned_profiles_by_model: dict[str, list[TraderProfile]] = {}
        tuned_summary_by_model: dict[str, pd.DataFrame] = {}
        full_tuning_results_by_model: dict[str, dict[str, pd.DataFrame]] = {}
        total_models = len(model_factories)

        for model_index, (model_name, model_factory) in enumerate(model_factories.items(), start=1):
            print(
                f"[progress] {ticker}: model {model_index}/{total_models} -> {model_name} (walk-forward validation)",
                flush=True,
            )
            fold_metrics, cv_predictions, overall_summary = walk_forward_validate(
                model_factory=model_factory,
                development=development,
                n_splits=args.n_splits,
                horizon=args.horizon,
                progress_label=f"{ticker} {model_name}",
            )
            cv_predictions_by_model[model_name] = cv_predictions.copy()
            model_tuned_profiles: list[TraderProfile] = []
            model_tuning_rows: list[dict] = []
            model_full_tuning_results: dict[str, pd.DataFrame] = {}

            for trader_index, base_profile in enumerate(BASE_TRADERS, start=1):
                print(
                    f"[progress] {ticker}: model {model_name} tuning trader {trader_index}/{len(BASE_TRADERS)} -> {base_profile.name}",
                    flush=True,
                )
                tuned_profile, tuning_results = tune_trader_profile(
                    base_profile=base_profile,
                    market_frame=development,
                    signal_predictions=cv_predictions,
                    initial_cash=args.initial_cash,
                    commission_rate=args.commission_rate,
                    slippage_rate=args.slippage_rate,
                    threshold_candidates=threshold_candidates,
                    allocation_scales=allocation_scales,
                    neutral_options=neutral_options,
                    margin_candidates=margin_candidates,
                )
                model_tuned_profiles.append(tuned_profile)
                model_full_tuning_results[base_profile.name] = tuning_results.copy()
                model_tuning_rows.append(
                    {
                        "trader": base_profile.name,
                        "selected_min_bull_probability": tuned_profile.min_bull_probability,
                        "selected_min_probability_margin": tuned_profile.min_probability_margin,
                        "selected_max_allocation": tuned_profile.max_allocation,
                        "selected_neutral_behavior": tuned_profile.neutral_behavior,
                        "cv_return_pct": float(tuning_results.iloc[0]["return_pct"]),
                        "cv_excess_return_vs_bh": float(tuning_results.iloc[0]["excess_return_vs_bh"]),
                        "cv_sharpe_like": float(tuning_results.iloc[0]["sharpe_like"]),
                        "cv_num_trades": int(tuning_results.iloc[0]["num_trades"]),
                        "cv_max_drawdown_pct": float(tuning_results.iloc[0]["max_drawdown_pct"]),
                        "cv_selection_score": float(tuning_results.iloc[0]["selection_score"]),
                    }
                )

            model_tuning_summary = pd.DataFrame(model_tuning_rows)
            tuned_profiles_by_model[model_name] = model_tuned_profiles
            tuned_summary_by_model[model_name] = model_tuning_summary
            full_tuning_results_by_model[model_name] = model_full_tuning_results
            trading_score = float(model_tuning_summary["cv_selection_score"].mean())
            average_trader_return = float(model_tuning_summary["cv_return_pct"].mean())
            average_trader_excess_return = float(model_tuning_summary["cv_excess_return_vs_bh"].mean())
            average_trader_sharpe = float(model_tuning_summary["cv_sharpe_like"].mean())
            average_trader_drawdown = float(model_tuning_summary["cv_max_drawdown_pct"].mean())
            selection_score = trading_score + float(overall_summary.loc[0, "macro_f1"])

            fold_metrics.insert(0, "model", model_name)
            fold_metrics.insert(0, "ticker", ticker)
            cv_predictions.insert(0, "model", model_name)
            cv_predictions.insert(0, "ticker", ticker)
            overall_summary.insert(0, "model", model_name)
            overall_summary.insert(0, "ticker", ticker)
            overall_summary["trading_score"] = trading_score
            overall_summary["avg_trader_return_pct"] = average_trader_return
            overall_summary["avg_trader_excess_return_vs_bh"] = average_trader_excess_return
            overall_summary["avg_trader_sharpe"] = average_trader_sharpe
            overall_summary["avg_trader_drawdown_pct"] = average_trader_drawdown
            overall_summary["selection_score"] = selection_score

            fold_metrics.to_csv(ticker_dir / f"{model_name}_cv_fold_metrics.csv", index=False)
            cv_predictions.to_csv(ticker_dir / f"{model_name}_cv_predictions.csv", index=False)
            overall_summary.to_csv(ticker_dir / f"{model_name}_cv_summary.csv", index=False)

            all_cv_rows.extend(overall_summary.to_dict(orient="records"))
            selection_rows.extend(overall_summary.to_dict(orient="records"))

        print(f"[progress] {ticker}: ranking models", flush=True)
        selection_table = pd.DataFrame(selection_rows)
        if args.selection_mode == "classification":
            selection_table = selection_table.sort_values(
                ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
                ascending=False,
            ).reset_index(drop=True)
        elif args.selection_mode == "trading":
            selection_table = filter_trading_selection_candidates(selection_table).sort_values(
                [
                    "trading_score",
                    "avg_trader_excess_return_vs_bh",
                    "avg_trader_sharpe",
                    "avg_trader_drawdown_pct",
                ],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
        else:
            selection_table = filter_trading_selection_candidates(selection_table).sort_values(
                ["selection_score", "trading_score", "macro_f1", "balanced_accuracy"],
                ascending=False,
            ).reset_index(drop=True)
        selection_table.to_csv(ticker_dir / "model_selection_summary.csv", index=False)

        best_row = selection_table.iloc[0]
        best_model_name = str(best_row["model"])
        print(f"[progress] {ticker}: fitting selected model -> {best_model_name}", flush=True)
        best_model = fit_final_model(model_factories[best_model_name], development)

        print(f"[progress] {ticker}: evaluating holdout", flush=True)
        holdout_predictions, holdout_metrics, holdout_report, holdout_confusion = evaluate_on_holdout(
            best_model,
            holdout,
        )
        holdout_predictions.to_csv(ticker_dir / "holdout_predictions.csv", index=False)
        holdout_confusion.to_csv(ticker_dir / "holdout_confusion_matrix.csv")

        with open(ticker_dir / "holdout_classification_report.txt", "w", encoding="utf-8") as handle:
            handle.write(holdout_report)

        importance = export_model_importance(best_model, ticker_dir)

        tuned_profiles = tuned_profiles_by_model[best_model_name]
        tuned_summary = tuned_summary_by_model[best_model_name].copy().sort_values("trader")
        for trader_name, tuning_results in full_tuning_results_by_model[best_model_name].items():
            tuning_results.to_csv(ticker_dir / f"{trader_name}_tuning_grid.csv", index=False)
        tuned_summary.to_csv(ticker_dir / "trader_threshold_tuning_summary.csv", index=False)

        metadata = {
            "start": args.start,
            "end": args.end,
            "horizon": args.horizon,
            "bull_threshold": args.bull_threshold,
            "bear_threshold": args.bear_threshold,
            "holdout_size": args.holdout_size,
            "n_splits": args.n_splits,
            "calibration_splits": args.calibration_splits,
            "selection_mode": args.selection_mode,
            "market_ticker": args.market_ticker,
            "selected_model": best_model_name,
            "development_rows": len(development),
            "holdout_rows": len(holdout),
            "threshold_candidates": threshold_candidates,
            "margin_candidates": margin_candidates,
            "allocation_scales": allocation_scales,
            "neutral_options": neutral_options,
            "tuned_profiles": [profile.__dict__ for profile in tuned_profiles],
        }
        model_path = save_model_bundle(ticker, best_model, ticker_dir, metadata)

        histories: dict[str, pd.DataFrame] = {}
        trader_summary_rows: list[dict] = []

        for trader_index, trader in enumerate(tuned_profiles, start=1):
            print(
                f"[progress] {ticker}: holdout simulation {trader_index}/{len(tuned_profiles)} -> {trader.name}",
                flush=True,
            )
            trader_history, _ = simulate_trader(
                trader=trader,
                market_frame=holdout,
                signal_predictions=holdout_predictions,
                initial_cash=args.initial_cash,
                commission_rate=args.commission_rate,
                slippage_rate=args.slippage_rate,
            )
            histories[trader.name] = trader_history
            trader_history.to_csv(ticker_dir / f"{trader.name}_trades.csv", index=False)

            trader_summary = summarize_history(trader.name, trader_history, args.initial_cash)
            trader_summary.update(
                {
                    "selected_min_bull_probability": trader.min_bull_probability,
                    "selected_min_probability_margin": trader.min_probability_margin,
                    "selected_max_allocation": trader.max_allocation,
                    "selected_neutral_behavior": trader.neutral_behavior,
                }
            )
            trader_summary_rows.append(trader_summary)

        print(f"[progress] {ticker}: running buy-and-hold benchmark", flush=True)
        buy_and_hold_history, _ = simulate_buy_and_hold(
            market_frame=holdout,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_rate=args.slippage_rate,
        )
        histories["buy_and_hold"] = buy_and_hold_history
        buy_and_hold_history.to_csv(ticker_dir / "buy_and_hold.csv", index=False)
        trader_summary_rows.append(summarize_history("buy_and_hold", buy_and_hold_history, args.initial_cash))

        trader_summary = pd.DataFrame(trader_summary_rows).sort_values(
            "final_portfolio_value",
            ascending=False,
        )
        trader_summary.to_csv(ticker_dir / "trader_summary.csv", index=False)

        buy_and_hold_row = trader_summary[trader_summary["trader"] == "buy_and_hold"].iloc[0]
        non_benchmark_rows = trader_summary[trader_summary["trader"] != "buy_and_hold"]
        best_trader_row = non_benchmark_rows.iloc[0] if not non_benchmark_rows.empty else None
        holdout_summary_row = {
            "ticker": ticker,
            "selected_model": best_model_name,
            **holdout_metrics,
            "best_trader": best_trader_row["trader"] if best_trader_row is not None else np.nan,
            "best_trader_return_pct": float(best_trader_row["return_pct"]) if best_trader_row is not None else np.nan,
            "best_trader_max_drawdown_pct": float(best_trader_row["max_drawdown_pct"]) if best_trader_row is not None else np.nan,
            "best_trader_sharpe_like": float(best_trader_row["sharpe_like"]) if best_trader_row is not None else np.nan,
            "buy_and_hold_return_pct": float(buy_and_hold_row["return_pct"]),
            "excess_return_vs_bh": (
                float(best_trader_row["return_pct"]) - float(buy_and_hold_row["return_pct"])
                if best_trader_row is not None
                else np.nan
            ),
        }
        all_holdout_rows.append(holdout_summary_row)

        print(f"[progress] {ticker}: plotting and writing outputs", flush=True)
        plot_histories(
            histories=histories,
            output_path=ticker_dir / "portfolio_values.png",
            title=f"{ticker} AI Trader Portfolio Value Over Time",
            show_plot=args.plot,
        )

        ticker_selection_rows.append(
            {
                "ticker": ticker,
                "selected_model": best_model_name,
                "best_cv_macro_f1": round(float(best_row["macro_f1"]), 4),
                "best_cv_balanced_accuracy": round(float(best_row["balanced_accuracy"]), 4),
                "holdout_macro_f1": round(float(holdout_metrics["macro_f1"]), 4),
                "holdout_balanced_accuracy": round(float(holdout_metrics["balanced_accuracy"]), 4),
                "saved_model_path": str(model_path),
            }
        )

        print(f"Rows after feature engineering: {len(featured)}")
        print(f"Development rows: {len(development)}")
        print(f"Holdout rows: {len(holdout)}")
        print("Model selection summary:")
        print(selection_table.to_string(index=False))
        print("Selected trader thresholds:")
        print(tuned_summary.to_string(index=False))
        print("Holdout performance:")
        print(pd.DataFrame([holdout_summary_row]).to_string(index=False))
        if not importance.empty:
            print("Best model importance:")
            print(importance.to_string(index=False))
        print("Trader performance:")
        print(trader_summary.to_string(index=False))
        print(f"Saved best model to: {model_path}")

    cv_summary = pd.DataFrame(all_cv_rows)
    holdout_summary = pd.DataFrame(all_holdout_rows)
    selected_models = pd.DataFrame(ticker_selection_rows)

    cv_summary.to_csv(output_dir / "all_tickers_cv_summary.csv", index=False)
    holdout_summary.to_csv(output_dir / "all_tickers_holdout_summary.csv", index=False)
    selected_models.to_csv(output_dir / "selected_models.csv", index=False)

    global_model_ranking = (
        cv_summary.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
        .mean()
        .sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
        .reset_index()
    )
    global_model_ranking.to_csv(output_dir / "global_model_ranking.csv", index=False)
    global_trading_ranking = (
        cv_summary.groupby("model")[
            ["trading_score", "avg_trader_excess_return_vs_bh", "avg_trader_sharpe", "avg_trader_drawdown_pct"]
        ]
        .mean()
        .sort_values(
            ["trading_score", "avg_trader_excess_return_vs_bh", "avg_trader_sharpe", "avg_trader_drawdown_pct"],
            ascending=[False, False, False, True],
        )
        .reset_index()
    )
    global_trading_ranking.to_csv(output_dir / "global_trading_model_ranking.csv", index=False)
    global_basket_trading_ranking = build_global_basket_trading_ranking(cv_summary)
    global_basket_trading_ranking.to_csv(output_dir / "global_basket_trading_ranking.csv", index=False)

    print("\n========== GLOBAL SUMMARY ==========")
    if not global_basket_trading_ranking.empty:
        best_basket_model = str(global_basket_trading_ranking.iloc[0]["model"])
        print("Basket trading ranking across all tickers:")
        print(global_basket_trading_ranking.to_string(index=False))
        print(f"Basket-selected model: {best_basket_model}")
    print("Trading-first model ranking across all tickers:")
    print(global_trading_ranking.to_string(index=False))
    print("Classification model ranking across all tickers:")
    print(global_model_ranking.to_string(index=False))
    print("Holdout summary:")
    print(holdout_summary.to_string(index=False))
    print("Selected models:")
    print(selected_models.to_string(index=False))
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
