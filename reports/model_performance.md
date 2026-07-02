# Price Prediction Model Performance Report

This report summarizes the performance evaluation and final optimization of the PropertyIQ price prediction model.

## 1. Dataset Statistics

- **Total Properties**: 3554
- **Total Features**: 12
- **Numerical Features**: 5 (bedRoom, bathroom, built_up_area, servant room, store room)
- **Categorical Features**: 7 (property_type, sector, balcony, agePossession, furnishing_type, luxury_category, floor_category)
- **Unique Sectors**: 104
- **Missing Values**: 0
- **Duplicate Records**: 34
- **Train/Test Split**: 80% / 20%
- **Property Type Distribution**:
  - Flat: 2804
  - House: 750
- **Furnishing Distribution**:
  - Unfurnished: 2349
  - Semifurnished: 1018
  - Furnished: 187

---

## 2. Model Comparison Table

| Algorithm | Train R² | Test R² | Adjusted R² | MAE (Cr) | RMSE (Cr) | CV Mean R² | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Reg | 0.8704 | 0.8704 | 0.8704 | 0.6492 | 1.6321 | 0.8470 | 0.34 MB |
| Ridge Reg | 0.8701 | 0.8701 | 0.8701 | 0.6531 | 1.6537 | 0.8470 | 0.34 MB |
| Lasso Reg | 0.7347 | 0.7347 | 0.7347 | 0.9600 | 2.1935 | 0.8470 | 0.34 MB |
| Decision Tree | 0.7959 | 0.7959 | 0.7959 | 0.6902 | 1.8207 | 0.8470 | 0.34 MB |
| Default RF | 0.9082 | 0.9082 | 0.9082 | 0.4931 | 1.1225 | 0.8470 | 22.58 MB |
| Default GBR | 0.8893 | 0.8893 | 0.8893 | 0.5694 | 1.1121 | 0.8470 | 0.34 MB |
| Tuned GBR | 0.9083 | 0.9083 | 0.9083 | 0.5061 | 1.1573 | 0.8908 | 0.34 MB |
| **Tuned RF (Selected)** | **0.9627** | **0.9085** | **0.9069** | **0.5034** | **1.1127** | **0.8780** | **21.49 MB** |

---

## 3. Selected Model Profile

- **Selected Algorithm**: RandomForestRegressor
- **Final Model Size on Disk**: 21.49 MB *(Previously 146.60 MB)*
- **Model Size Reduction**: 85.3%

### Final Performance Metrics

- **Train R²**: 0.9627
- **Test R²**: 0.9085
- **Adjusted R²**: 0.9069
- **Mean Absolute Error (MAE)**: 0.5034 Cr
- **Root Mean Squared Error (RMSE)**: 1.1127 Cr
- **Mean Squared Error (MSE)**: 1.2382 Cr²
- **Mean Absolute Percentage Error (MAPE)**: 19.87%
- **Median Absolute Error**: 0.1826 Cr
- **5-Fold Cross Validation Mean R²**: 0.8780 (Std: 0.0137)
- **Training Time**: 2.93 seconds
- **Single Prediction Latency**: 20.98 ms
- **Average Inference Time (100 runs)**: 21.88 ms

---

## 4. Feature Importance

### Top 20 Most Important Features

| Rank | Feature Name | Importance Score | Percentage Contribution |
| --- | --- | --- | --- |
| 1 | built_up_area | 0.503178 | 50.32% |
| 2 | bedRoom | 0.110106 | 11.01% |
| 3 | bathroom | 0.106303 | 10.63% |
| 4 | property_type | 0.081998 | 8.20% |
| 5 | sector | 0.079563 | 7.96% |
| 6 | servant room | 0.019454 | 1.95% |
| 7 | balcony | 0.016001 | 1.60% |
| 8 | sector_sector 26 | 0.007269 | 0.73% |
| 9 | furnishing_type | 0.006720 | 0.67% |
| 10 | agePossession | 0.005983 | 0.60% |
| 11 | floor_category | 0.005444 | 0.54% |
| 12 | sector_sector 65 | 0.004801 | 0.48% |
| 13 | luxury_category | 0.004412 | 0.44% |
| 14 | store room | 0.003150 | 0.31% |
| 15 | sector_sector 25 | 0.002705 | 0.27% |
| 16 | sector_sector 54 | 0.002506 | 0.25% |
| 17 | agePossession_Relatively New | 0.002368 | 0.24% |
| 18 | sector_sector 43 | 0.002315 | 0.23% |
| 19 | agePossession_Old Property | 0.002217 | 0.22% |
| 20 | sector_manesar | 0.001976 | 0.20% |

### Feature Importance Chart
See the generated plot at [reports/feature_importance.png](file:///c:/Users/HP/Desktop/propertyiq/reports/feature_importance.png).
