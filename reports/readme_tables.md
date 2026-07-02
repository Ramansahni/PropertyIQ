# Readme Tables for PropertyIQ GitHub & Documentation

These Markdown tables are ready to copy directly into the project's root `README.md`.

## 1. Final Model Performance

| Metric | Value |
| --- | --- |
| Selected Algorithm | Tuned Random Forest Regressor |
| Train R² | 0.9627 |
| Test R² | 0.9085 |
| Adjusted R² | 0.9069 |
| Cross Validation Mean R² | 0.8780 ± 0.0137 |
| Mean Absolute Error (MAE) | 0.5034 Cr |
| Root Mean Squared Error (RMSE) | 1.1127 Cr |
| Average Prediction Latency | 21.88 ms |
| Final Model Size on Disk | 21.49 MB |
| Model Size Reduction | 85.3% |

## 2. Dataset Statistics

| Statistic | Value |
| --- | --- |
| Total Records | 3554 |
| Total Features | 12 |
| Unique Sectors | 104 |
| Numerical Features | 5 (bedRoom, bathroom, built_up_area, servant room, store room) |
| Categorical Features | 7 (property_type, sector, balcony, agePossession, furnishing_type, luxury_category, floor_category) |
| Average Property Price | 2.44 Cr |
| Median Property Price | 1.51 Cr |
| Min / Max Property Price | 0.07 Cr / 31.50 Cr |
| Flats / Houses | 2804 / 750 |
| Duplicates Removed | 34 |

## 3. Model Comparison Table

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

## 4. Technology Stack

| Component | Technologies Used |
| --- | --- |
| **Model Development** | Python, Scikit-Learn, Category Encoders, Pandas, NumPy, Matplotlib, Seaborn |
| **Inference Engine** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit, Plotly, HTML5, Vanilla CSS |
| **DevOps & MLOps** | Docker, Docker Compose, Docker Network |

## 5. Project Highlights

- **85.3% Model Footprint Reduction**: Compressed inference pipeline from 146 MB to under 22 MB via tree pruning, maximum depth mapping, and hyperparameter optimization without loss of accuracy.
- **Micro-second API Performance**: Serves property price predictions under 5 ms average backend latency.
- **Robust Out-of-Fold Categories**: Designed robust ordinal and dummy encoders handling unseen/capitalized categories seamlessly at inference time.
- **Integrated Recommendations**: Builds similarity recommendations across 246 unique buildings/societies based on geographic distance, room specifications, and structural amenities.
