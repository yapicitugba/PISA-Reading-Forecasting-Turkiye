"""
Create forecast output tables and GRU forecast figures.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs/forecasts")
OUT.mkdir(parents=True, exist_ok=True)

forecast_data = {
    "GRU": [
        ["Model 0", "Reading Literacy", 459, 457, 461],
        ["Model 1", "Reading Literacy + PARED", 463, 461, 465],
        ["Model 2", "Reading Literacy + PARED + CULTPOS", 465, 463, 467],
        ["Model 3", "Reading Literacy + PARED + CULTPOS + HISEI", 466, 464, 468],
        ["Model 4", "Reading Literacy + PARED + CULTPOS + HISEI + BELONG", 466, 464, 468],
    ],
    "LSTM": [
        ["Model 0", "Reading Literacy", 458, 456, 460],
        ["Model 1", "Reading Literacy + PARED", 461, 459, 463],
        ["Model 2", "Reading Literacy + PARED + CULTPOS", 463, 461, 465],
        ["Model 3", "Reading Literacy + PARED + CULTPOS + HISEI", 464, 462, 466],
        ["Model 4", "Reading Literacy + PARED + CULTPOS + HISEI + BELONG", 464, 462, 466],
    ],
    "ARIMA": [
        ["Model 0", "Reading Literacy", 457, 455, 459],
        ["Model 1", "Reading Literacy + PARED", 459, 457, 461],
        ["Model 2", "Reading Literacy + PARED + CULTPOS", 461, 459, 463],
        ["Model 3", "Reading Literacy + PARED + CULTPOS + HISEI", 462, 460, 464],
        ["Model 4", "Reading Literacy + PARED + CULTPOS + HISEI + BELONG", 462, 460, 464],
    ],
    "Prophet": [
        ["Model 0", "Reading Literacy", 456, 454, 458],
        ["Model 1", "Reading Literacy + PARED", 458, 456, 460],
        ["Model 2", "Reading Literacy + PARED + CULTPOS", 459, 457, 461],
        ["Model 3", "Reading Literacy + PARED + CULTPOS + HISEI", 460, 458, 462],
        ["Model 4", "Reading Literacy + PARED + CULTPOS + HISEI + BELONG", 460, 458, 462],
    ],
}

with pd.ExcelWriter(OUT / "forecast_results_all_models.xlsx") as writer:
    for model_name, rows in forecast_data.items():
        df = pd.DataFrame(rows, columns=["Model", "Added Independent Variables", "2025", "2028", "2031"])
        df.to_excel(writer, sheet_name=model_name, index=False)
        df.to_csv(OUT / f"{model_name.lower()}_forecasts.csv", index=False)

# Türkiye observed values
observed_cycles = [2003, 2006, 2009, 2012, 2015, 2018, 2022]
observed_scores = [441, 447, 464, 475, 428, 466, 456]

gru = pd.DataFrame(forecast_data["GRU"], columns=["Model", "Variables", "2025", "2028", "2031"])
forecast_cycles = [2025, 2028, 2031]

plt.figure(figsize=(10, 6))
plt.plot(observed_cycles, observed_scores, marker="o", linewidth=2.5, color="black", label="Measured Values")
for _, row in gru.iterrows():
    plt.plot(forecast_cycles, [row["2025"], row["2028"], row["2031"]], marker="o", linestyle="--", label=f"{row['Model']}: {row['Variables'].replace('Reading Literacy', 'Reading Literacy')}")
plt.title("GRU-Based Forecasts with Incremental Effects of Independent Variables")
plt.xlabel("PISA Cycle")
plt.ylabel("PISA Reading Literacy Scores")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUT / "gru_based_forecasts.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
for _, row in gru.iterrows():
    plt.plot(forecast_cycles, [row["2025"], row["2028"], row["2031"]], marker="o", linestyle="--", label=f"{row['Model']}: {row['Variables']}")
plt.title("Incremental Contribution of Independent Variables to GRU Forecasts")
plt.xlabel("PISA Cycle")
plt.ylabel("PISA Reading Literacy Scores")
plt.xticks([2025, 2028, 2031])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "gru_incremental_contribution.png", dpi=300)
plt.close()

print("Forecast outputs and figures saved to:", OUT)
