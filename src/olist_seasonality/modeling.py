from __future__ import annotations

import json
import math
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import FIGURES, MONTH_NAMES, NOTES, PROCESSED, RESULTS, TABLES, WEEKDAY_NAMES
from .metrics import order_level
from .utils import mode_or_nan, safe_div, save_fig, table_to_markdown

def prediction_block(enriched: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delivered = enriched[enriched["is_delivered"]].copy()
    features = (
        delivered.groupby("product_category_name_english")
        .agg(
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_freight_value=("freight_value", "mean"),
            freight_share=("freight_share", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
            avg_review_score=("review_score", "mean"),
            share_low_reviews=("is_low_review", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_product_weight_g=("product_weight_g", "mean"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
            avg_product_photos_qty=("product_photos_qty", "mean"),
            avg_product_name_length=("product_name_lenght", "mean"),
            avg_product_description_length=("product_description_lenght", "mean"),
            customer_state_nunique=("customer_state", "nunique"),
            seller_state_nunique=("seller_state", "nunique"),
            total_orders=("order_id", "nunique"),
            total_revenue=("price", "sum"),
        )
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    pred = features.merge(ranking[["category", "seasonality_score", "confidence_level", "seasonality_type"]], on="category", how="inner")
    eligible = pred[pred["confidence_level"].ne("low")].copy()
    threshold = eligible["seasonality_score"].quantile(0.75)
    pred["is_seasonal"] = pred["seasonality_score"].ge(threshold).astype(int)
    pred.to_csv(PROCESSED / "seasonality_prediction_features.csv", index=False)

    numeric_cols = [c for c in features.columns if c != "category"]
    model_df = pred[pred["confidence_level"].ne("low")].copy()
    X = model_df[numeric_cols]
    y = model_df["is_seasonal"]
    results_rows = []
    importance = pd.DataFrame({"feature": numeric_cols, "importance": 0.0})
    if len(model_df) >= 20 and y.nunique() == 2 and y.value_counts().min() >= 3:
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=stratify)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", min_samples_leaf=2)),
            ]
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred_label = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan
        results_rows.append({"model": "random_forest", "roc_auc": auc, "f1": f1_score(y_test, pred_label), "accuracy": accuracy_score(y_test, pred_label), "n_train": len(X_train), "n_test": len(X_test), "target_threshold": threshold})
        rf = model.named_steps["rf"]
        importance = pd.DataFrame({"feature": numeric_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure(figsize=(5.8, 5.2))
        plt.plot(fpr, tpr, label=f"RF ROC-AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.title("Seasonality prediction ROC curve")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        save_fig(FIGURES / "12_seasonality_prediction_roc.png")
    else:
        results_rows.append({"model": "not_enough_data", "roc_auc": np.nan, "f1": np.nan, "accuracy": np.nan, "n_train": 0, "n_test": len(model_df), "target_threshold": threshold})
        correlations = []
        for col in numeric_cols:
            correlations.append((col, abs(model_df[col].corr(model_df["seasonality_score"], method="spearman"))))
        importance = pd.DataFrame(correlations, columns=["feature", "importance"]).sort_values("importance", ascending=False)

    results = pd.DataFrame(results_rows)
    results.to_csv(PROCESSED / "seasonality_prediction_results.csv", index=False)
    results.to_csv(TABLES / "seasonality_prediction_results.csv", index=False)
    importance.to_csv(TABLES / "model_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.6))
    sns.barplot(data=importance.head(12).sort_values("importance"), y="feature", x="importance", color="#4C72B0")
    plt.title("Feature importance for predicting seasonal categories")
    plt.xlabel("Importance")
    plt.ylabel("")
    save_fig(FIGURES / "13_seasonality_feature_importance.png")
    return pred, results


def forecast_block(enriched: pd.DataFrame) -> pd.DataFrame:
    orders = order_level(enriched).copy()
    orders["week_start"] = pd.PeriodIndex(orders["week"], freq="W").start_time
    weekly = orders.groupby("week_start")["order_id"].nunique().asfreq("W-MON").fillna(0)
    weekly = weekly[weekly.index <= pd.Timestamp("2018-08-27")]
    rows = []
    forecast_df = pd.DataFrame()
    if len(weekly) >= 50:
        test_horizon = min(8, max(4, len(weekly) // 8))
        train, test = weekly.iloc[:-test_horizon], weekly.iloc[-test_horizon:]
        mean_forecast = pd.Series(train.tail(4).mean(), index=test.index)
        seasonal_naive = train.reindex(test.index - pd.Timedelta(weeks=52)).set_axis(test.index)
        if seasonal_naive.isna().any():
            seasonal_naive = pd.Series(train.tail(8).mean(), index=test.index)
        rows.append({"model": "last_4_week_mean", "mae": float((test - mean_forecast).abs().mean()), "mape": float(((test - mean_forecast).abs() / test.replace(0, np.nan)).mean())})
        rows.append({"model": "seasonal_naive_or_8w_mean", "mae": float((test - seasonal_naive).abs().mean()), "mape": float(((test - seasonal_naive).abs() / test.replace(0, np.nan)).mean())})
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 4), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarima = model.get_forecast(steps=test_horizon).summary_frame()["mean"].set_axis(test.index)
            rows.append({"model": "sarima_weekly_s4", "mae": float((test - sarima).abs().mean()), "mape": float(((test - sarima).abs() / test.replace(0, np.nan)).mean())})
        except Exception as exc:
            sarima = pd.Series(np.nan, index=test.index)
            rows.append({"model": f"sarima_failed: {type(exc).__name__}", "mae": np.nan, "mape": np.nan})
        forecast_df = pd.DataFrame({"actual": test, "last_4_week_mean": mean_forecast, "seasonal_naive_or_8w_mean": seasonal_naive, "sarima_weekly_s4": sarima}).reset_index().rename(columns={"week_start": "week_start"})
        plt.figure(figsize=(12, 4.8))
        plt.plot(weekly.index, weekly.values, label="Historical weekly orders", color="steelblue")
        plt.plot(test.index, mean_forecast, label="Last-4-week mean", linestyle="--")
        plt.plot(test.index, seasonal_naive, label="Seasonal naive / 8w mean", linestyle="--")
        if sarima.notna().any():
            plt.plot(test.index, sarima, label="SARIMA weekly s=4", linestyle="--")
        plt.axvspan(test.index.min(), test.index.max(), color="gray", alpha=0.12, label="Holdout")
        plt.title("Forecast example: weekly delivered orders")
        plt.xlabel("Week")
        plt.ylabel("Orders")
        plt.legend()
        save_fig(FIGURES / "14_optional_forecast_examples.png")
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "forecast_model_comparison.csv", index=False)
    if not forecast_df.empty:
        forecast_df.to_csv(TABLES / "forecast_holdout_predictions.csv", index=False)
    return results
