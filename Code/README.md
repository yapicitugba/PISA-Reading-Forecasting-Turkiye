# PISA Reading Literacy Forecasting Dataset and Code Package

# Data-Driven Forecasting of PISA Reading Literacy Performance of Türkiye

## Overview

This repository contains the datasets, Python scripts, forecasting outputs, and supplementary documentation used in the study:

**"Data-Driven Forecasting of PISA Reading Literacy Performance of Türkiye: Artificial Intelligence Based Time Series Estimation"**

The study investigates Türkiye’s future PISA Reading Literacy performance using both classical statistical forecasting methods and deep learning approaches.

Forecasting models include:

- ARIMA
- ARIMAX
- Prophet
- LSTM
- GRU

Independent variables were sequentially incorporated into the forecasting framework according to their estimated contribution to reading literacy performance:

- Highest Parental Education (PARED)
- Cultural Possessions (CULTPOS)
- Highest Parental Occupational Status (HISEI)
- Sense of Belonging at School (BELONG)

---

## Repository Structure

```text
Dataset/
Code/
Figures/
Documentation/
```

---

## Dataset

The Dataset folder contains:

| File | Description |
|--------|--------|
| 01_reading_scores_raw.xlsx | Original reading literacy scores |
| 02_independent_variables_raw.xlsx | Original independent variables |
| 03_reading_scores_processed.xlsx | Completed reading literacy dataset |
| 04_independent_variables_processed.xlsx | Completed independent variables dataset |
| 05_final_turkiye_model_dataset.xlsx | Final modelling dataset |
| 06_standardized_model_dataset.xlsx | Z-score standardized dataset |
| forecast_results_all_models.xlsx | Forecast outputs for all forecasting models |

---

## Forecasting Models

### Statistical Models

- ARIMA
- ARIMAX
- Prophet

### Deep Learning Models

- LSTM
- GRU

---

## Data Preprocessing

The following preprocessing procedures were applied:

- Missing value interpolation
- Missing value extrapolation
- Z-score standardization
- Sliding window transformation

Detailed descriptions are provided in the supplementary documentation.

---

## Forecasting Strategy

The study employed a recursive multi-step forecasting approach.

Forecasts were generated sequentially:

- 2025 forecast using data up to 2022
- 2028 forecast using data up to 2025
- 2031 forecast using data up to 2028

---

## Evaluation Metrics

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Dynamic Time Warping (DTW)

---

## Reproducibility

All datasets, scripts, forecasting outputs, and supplementary materials necessary to reproduce the analyses are provided in this repository.

Required software packages are listed in:

```text
requirements.txt
```

---

## Data Source

Original PISA data are publicly available from:

OECD Programme for International Student Assessment (PISA)

https://www.oecd.org/pisa/

---

## Citation

If you use these materials, please cite the associated publication.

---

## Contact

For questions regarding the repository, data package, or study, please contact the corresponding author.

---

## License

This repository is provided for academic and research purposes. Please provide appropriate citation when using the datasets, code, or supplementary materials.
