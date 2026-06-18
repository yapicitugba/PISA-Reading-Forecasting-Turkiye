"""
Create metric tables and comparison figures for MAE, MSE, RMSE, and DTW.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs/metrics")
OUT.mkdir(parents=True, exist_ok=True)

baseline_metrics = pd.DataFrame({
    "Model": ["GRU", "LSTM", "ARIMA", "Prophet"],
    "MAE": [8.9, 10.6, 12.4, 13.9],
    "MSE": [129.8, 151.2, 176.5, 198.7],
    "RMSE": [11.39, 12.30, 13.29, 14.09],
    "DTW": [37.2, 42.8, 49.6, 55.1],
})

metrics = {
    "GRU": pd.DataFrame({
        "Model": ["M0", "M1", "M2", "M3", "M4"],
        "Included Variables": [
            "Reading Literacy",
            "Reading Literacy + PARED",
            "Reading Literacy + PARED + CULTPOS",
            "Reading Literacy + PARED + CULTPOS + HISEI",
            "Reading Literacy + PARED + CULTPOS + HISEI + BELONG",
        ],
        "MAE": [8.9, 7.8, 7.4, 7.2, 7.25],
        "MSE": [129.8, 115.4, 108.6, 105.3, 106.2],
        "RMSE": [11.39, 10.74, 10.42, 10.26, 10.30],
        "DTW": [37.2, 31.5, 29.8, 28.9, 29.2],
    }),
    "LSTM": pd.DataFrame({
        "Model": ["M0", "M1", "M2", "M3", "M4"],
        "Included Variables": [
            "Reading Literacy",
            "Reading Literacy + PARED",
            "Reading Literacy + PARED + CULTPOS",
            "Reading Literacy + PARED + CULTPOS + HISEI",
            "Reading Literacy + PARED + CULTPOS + HISEI + BELONG",
        ],
        "MAE": [10.6, 9.6, 9.2, 9.0, 9.1],
        "MSE": [151.2, 138.4, 132.1, 129.5, 130.8],
        "RMSE": [12.30, 11.77, 11.49, 11.38, 11.44],
        "DTW": [42.8, 37.9, 35.8, 34.9, 35.3],
    }),
    "ARIMA": pd.DataFrame({
        "Model": ["M0", "M1", "M2", "M3", "M4"],
        "Included Variables": [
            "Reading Literacy",
            "Reading Literacy + PARED",
            "Reading Literacy + PARED + CULTPOS",
            "Reading Literacy + PARED + CULTPOS + HISEI",
            "Reading Literacy + PARED + CULTPOS + HISEI + BELONG",
        ],
        "MAE": [12.4, 11.6, 11.3, 11.1, 11.2],
        "MSE": [176.5, 162.3, 155.8, 152.6, 154.1],
        "RMSE": [13.29, 12.74, 12.48, 12.35, 12.40],
        "DTW": [49.6, 45.8, 43.9, 42.7, 43.1],
    }),
    "Prophet": pd.DataFrame({
        "Model": ["M0", "M1", "M2", "M3", "M4"],
        "Included Variables": [
            "Reading Literacy",
            "Reading Literacy + PARED",
            "Reading Literacy + PARED + CULTPOS",
            "Reading Literacy + PARED + CULTPOS + HISEI",
            "Reading Literacy + PARED + CULTPOS + HISEI + BELONG",
        ],
        "MAE": [13.9, 13.2, 12.9, 12.7, 12.8],
        "MSE": [198.7, 185.4, 179.8, 176.3, 177.5],
        "RMSE": [14.09, 13.62, 13.41, 13.28, 13.33],
        "DTW": [55.1, 51.2, 49.6, 48.5, 48.9],
    }),
}

baseline_metrics.to_excel(OUT / "baseline_metric_scores.xlsx", index=False)
with pd.ExcelWriter(OUT / "metric_scores_by_model.xlsx") as writer:
    for model_name, df in metrics.items():
        df.to_excel(writer, sheet_name=model_name, index=False)

def plot_metric(metric: str):
    combined = pd.DataFrame({"Model": ["M0", "M1", "M2", "M3", "M4"]})
    for model_name, df in metrics.items():
        combined[model_name] = df[metric].values

    ax = combined.set_index("Model").plot(kind="bar", figsize=(10, 6), width=0.75)
    ax.set_title(f"Comparison of Models Across Variable Combinations ({metric})")
    ax.set_xlabel("Models (M0–M4)")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT / f"{metric.lower()}_comparison.png", dpi=300)
    plt.close()

for metric in ["MAE", "MSE", "RMSE", "DTW"]:
    plot_metric(metric)

print("Metric tables and figures saved to:", OUT)
