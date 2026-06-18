"""
Estimate ARIMAX-based variable weighting for the sequential variable inclusion strategy.

The reported coefficients in the article are:
PARED = 0.64, CULTPOS = 0.43, HISEI = 0.28, BELONG = 0.12.

This script also provides a reproducible ARIMAX estimation template using statsmodels.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUT = Path("outputs/arimax")
OUT.mkdir(parents=True, exist_ok=True)

data = pd.DataFrame({
    "Cycle": [2003, 2006, 2009, 2012, 2015, 2018, 2022],
    "Reading_Literacy": [441, 447, 464, 475, 428, 466, 456],
    "PARED": [8.98, 8.65, 8.77, 8.76, 9.67, 10.87, 11.43],
    "CULTPOS": [-0.11, -0.001, 0.52, -0.12, -0.26, -0.77, -1.45],
    "HISEI": [41.90, 39.83, 41.16, 35.14, 36.39, 37.40, 37.78],
    "BELONG": [1.84, 1.27, 0.70, 0.13, -0.44, -0.14, -0.31],
})

reported_coefficients = pd.DataFrame({
    "Independent Variable": ["PARED", "CULTPOS", "HISEI", "BELONG"],
    "Weighting Factor (beta)": [0.64, 0.43, 0.28, 0.12],
}).sort_values("Weighting Factor (beta)", ascending=False)

reported_coefficients.to_excel(OUT / "reported_arimax_coefficients.xlsx", index=False)

# Reproducible estimation template
y = data["Reading_Literacy"]
X = data[["PARED", "CULTPOS", "HISEI", "BELONG"]]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

try:
    model = SARIMAX(
        y,
        exog=X_scaled,
        order=(1, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    estimated = result.params[X_scaled.columns].reset_index()
    estimated.columns = ["Independent Variable", "Estimated Coefficient"]
    estimated["Absolute Coefficient"] = estimated["Estimated Coefficient"].abs()
    estimated = estimated.sort_values("Absolute Coefficient", ascending=False)
    estimated.to_excel(OUT / "estimated_arimax_coefficients_template.xlsx", index=False)
    print(result.summary())
except Exception as e:
    print("ARIMAX estimation could not be completed with the current small sample.")
    print("Reason:", e)

print("ARIMAX coefficient files saved to:", OUT)
