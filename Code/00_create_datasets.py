"""
Create raw, processed, and final model datasets for the PISA Reading Literacy forecasting study.
"""

from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs/datasets")
OUT.mkdir(parents=True, exist_ok=True)

cycles = [2003, 2006, 2009, 2012, 2015, 2018, 2022]

reading_raw = pd.DataFrame({
    "Country": ["Mexico", "Brazil", "Chile", "Argentina", "Colombia", "Romania", "Bulgaria", "Serbia", "Malaysia", "Peru", "Russia", "Türkiye"],
    2003: [400, 403, 410, 389, 385, 396, 403, 412, 398, 347, 442, 441],
    2006: [410, 393, 442, 391, 385, 396, 402, 401, np.nan, np.nan, 440, 447],
    2009: [425, 412, 449, 374, 413, 424, 429, 442, 414, 370, 459, 464],
    2012: [424, 410, 441, 396, 403, 438, 436, 446, 398, 384, 475, 475],
    2015: [423, 407, 459, 425, 425, 434, 432, 413, 431, 398, 495, 428],
    2018: [420, 413, 452, 402, 412, 428, 420, 439, 415, 401, 479, 466],
    2022: [415, 410, 448, 401, 409, 428, 404, 440, 388, 408, 456, 456],
})

independent_raw = pd.DataFrame({
    "Cycle": cycles,
    "HISEI": [41.90, 39.83, 41.16, 35.14, 36.39, 37.40, 37.78],
    "PARED": [8.98, 8.65, 8.77, 8.76, 9.67, 10.87, 11.43],
    "CULTPOS": [-0.11, -0.001, 0.52, -0.12, -0.26, -0.77, np.nan],
    "BELONG": [np.nan, np.nan, np.nan, 0.13, -0.44, -0.14, -0.31],
})

reading_processed = reading_raw.copy()
# Linear interpolation for missing 2006 values
reading_processed.loc[reading_processed["Country"] == "Malaysia", 2006] = (398 + 414) / 2
reading_processed.loc[reading_processed["Country"] == "Peru", 2006] = (347 + 370) / 2

independent_processed = independent_raw.copy()
# CULTPOS 2022: linear extrapolation from the 2015-2018 decreasing trend
independent_processed.loc[independent_processed["Cycle"] == 2022, "CULTPOS"] = -1.45

# BELONG 2003-2009: backward extrapolation using the 2012-2015 trend
independent_processed.loc[independent_processed["Cycle"] == 2009, "BELONG"] = 0.70
independent_processed.loc[independent_processed["Cycle"] == 2006, "BELONG"] = 1.27
independent_processed.loc[independent_processed["Cycle"] == 2003, "BELONG"] = 1.84

turkiye_reading = reading_processed[reading_processed["Country"] == "Türkiye"].melt(
    id_vars="Country", var_name="Cycle", value_name="Reading_Literacy"
)
turkiye_reading = turkiye_reading.drop(columns=["Country"])
turkiye_reading["Cycle"] = turkiye_reading["Cycle"].astype(int)

final_model_dataset = turkiye_reading.merge(independent_processed, on="Cycle", how="left")
final_model_dataset = final_model_dataset[["Cycle", "Reading_Literacy", "PARED", "CULTPOS", "HISEI", "BELONG"]]

reading_raw.to_excel(OUT / "reading_scores_raw.xlsx", index=False)
independent_raw.to_excel(OUT / "independent_variables_raw.xlsx", index=False)
reading_processed.to_excel(OUT / "reading_scores_processed.xlsx", index=False)
independent_processed.to_excel(OUT / "independent_variables_processed.xlsx", index=False)
final_model_dataset.to_excel(OUT / "final_turkiye_model_dataset.xlsx", index=False)

print("Datasets saved to:", OUT)
