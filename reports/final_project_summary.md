# PropertyIQ: End-to-End Machine Learning System for Real Estate Predictions & Recommendations

This document serves as the master production reference, comprehensive technical summary, and deployment documentation for the PropertyIQ project.

---

## 1. Project Overview

**PropertyIQ** is a production-grade, end-to-end Machine Learning web application designed to analyze, predict, and recommend real estate listings in Gurgaon, India. 

### Problem Statement
Real estate pricing is highly non-linear, fragmented, and influenced by various geographical, physical, and aesthetic factors. Traditional valuation methods rely on broad averages that fail to capture localized market dynamics, while typical recommendation engines fail to capture contextual similarity across different dimensions like price ranges, structural amenities, or proximity.

### Objective
To build a high-performance, low-latency, and containerized pricing prediction model coupled with a multi-criteria similarity recommendation engine. The system provides real-time estimations and item-to-item recommendations for buyers, agents, and real estate developers.

### Target Users
- **Home Buyers & Sellers**: To evaluate fair-market valuations and explore highly matching property alternatives.
- **Real Estate Agents & Valuers**: To accelerate market appraisals and generate data-driven pricing reports.
- **Developers & Investors**: To run sensitivity analyses on properties and assess returns based on location and build specifications.

### Why Machine Learning is Used
Valuation variables (e.g., sector location premiums, built-up areas, bedroom counts, luxury metrics, and furnishing tiers) have complex interactions. Machine learning algorithms (like ensembles of Decision Trees) successfully model these non-linear spatial and structural dynamics to yield predictions with low mean error, while TF-IDF vectorization and cosine distances enable multi-dimensional similarity mapping.

---

## 2. Complete Project Architecture

The application is structured as a decoupled, multi-container system that isolates the frontend UI, API service layer, and storage/model layers.

```
       [ User Web Browser ]
                │
                ▼
      [ Streamlit Frontend ] (Port 8501)
                │
                ▼ (Internal Docker Network Bridge)
       [ FastAPI Backend ] (Port 8000)
         ├── [ Lifespan Cache ] (Pipeline & Similarity Matrices)
         ├── [ Database Manager ] (SQLite Auth & Admin)
         └── [ ML Inference Engine ]
                │
         ┌──────┴──────┐
         ▼             ▼
  [ df.pkl ]      [ pipeline.pkl ]
(Metadata Index)  (Tuned Pipeline)
```

### Architectural Layer Responsibilities
1. **User Layer (Browser)**: Renders the interface, captures inputs, and visualizes graphs.
2. **Streamlit UI Container**: Manages session state, handles login routing, and formats user input fields. It delegates all data loading and prediction computations to the backend via REST API calls.
3. **FastAPI Service Container**: Serves as the central API gateway. It loads and caches the model pipeline and similarity datasets once at startup (lifespan context), exposes versioned REST endpoints, manages CORS middleware, and provides Swagger-UI documentation.
4. **SQLite Storage Layer**: Manages user authentication records, login verification, and administrative dashboards.
5. **Machine Learning Pipeline Layer**: Preprocesses raw inputs through scaling, ordinal mapping, and one-hot encoding, feeding them to the trained estimator to generate predictions.

---

## 3. Technology Stack

| Component | Technologies Used | Description |
| --- | --- | --- |
| **Programming Language** | Python 3.11 / 3.12 | Core development language |
| **Machine Learning Libraries**| Scikit-Learn, Category Encoders, NumPy, Pandas | Model building, preprocessing, and matrix operations |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic | Asynchronous REST API, schema validation, and server hosting |
| **Frontend UI** | Streamlit, Plotly, HTML5, Vanilla CSS | Interactive dashboards, custom dark themes, and visualizations |
| **Database** | SQLite, SQLite3 API | User credential authentication and admin tables |
| **Containerization** | Docker, Docker Compose | Decoupled microservices architecture and image packaging |
| **Deployment** | Render / Production-ready configurations | Cloud deployment capability |
| **Version Control** | Git | Code versioning and repository management |

---

## 4. Dataset Summary

The statistical profile of the final processed dataset (`gurgaon_properties_post_feature_selection_v2.csv`) is detailed below:

- **Total Properties**: `3,554`
- **Flats Count**: `2,804` (78.9%)
- **Houses Count**: `750` (21.1%)
- **Unique Sectors**: `104`
- **Numerical Features**: `5` (`bedRoom`, `bathroom`, `built_up_area`, `servant room`, `store room`)
- **Categorical Features**: `7` (`property_type`, `sector`, `balcony`, `agePossession`, `furnishing_type`, `luxury_category`, `floor_category`)
- **Duplicate Records Removed**: `34`
- **Missing Values**: `0`
- **Recommendation Index Size**: `246` unique building societies

---

## 5. Machine Learning Pipeline

The property price prediction model is built as a unified `Pipeline` object containing both transformers and the regressor, facilitating safe, leakage-free inference.

```
Raw Input Data
      │
      ▼
ColumnTransformer
  ├── num ──► [ StandardScaler ] ──────► ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
  ├── cat ──► [ OrdinalEncoder ] ──────► ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
  └── cat1 ─► [ OneHotEncoder ] ───────► ['sector', 'agePossession'] (drop='first')
      │
      ▼
Tuned Random Forest Regressor
      │
      ▼
Estimated Price in Log Scale ──► [ expm1 ] ──► Final Price (Cr)
```

### Pipeline Stages
1. **Data Cleaning**: Handled structural anomalies, unified string column casing, and mapped numerical indicators to descriptive strings (e.g., transforming furnishing floats `0.0`, `1.0`, `2.0` to `'unfurnished'`, `'semifurnished'`, `'furnished'`).
2. **Missing Value Handling**: Imputed records with zero missing attributes across target columns.
3. **Outlier Treatment**: Managed extreme pricing values using a log-transformation (`log1p`) on the target price variable. This compressed the distribution range, stabilized variance, and minimized the impact of super-luxury outliers.
4. **Feature Engineering**: Engineered luxury category indices, floor-level categories, and extracted coordinate dimensions for recommendation matching.
5. **Feature Preprocessing (Within Pipeline)**:
   - **Numerical Scaling**: Scaled all numerical columns using `StandardScaler` to ensure zero mean and unit variance.
   - **Ordinal Encoding**: Embedded structural features using `OrdinalEncoder` equipped with `handle_unknown='use_encoded_value'` and `unknown_value=-1` to ensure production robustness.
   - **One-Hot Encoding**: Dummy-encoded high-cardinality nominal columns (`sector`, `agePossession`) using `OneHotEncoder(drop='first', handle_unknown='ignore')` to capture unique segment variations without dummy variable traps.
6. **Model Selection**: Evaluated multiple regression families (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) using 5-Fold Cross Validation on the training split.
7. **Hyperparameter Tuning**: Performed `RandomizedSearchCV` on Random Forest and Gradient Boosting architectures to find the best configuration that limits depth and leaf size.
8. **Final Training**: Selected the **Tuned Random Forest Regressor** and refitted the entire pipeline on 100% of the dataset to maximize training data utilization.

---

## 6. Final Model Summary

The profile of the final trained and exported price predictor is as follows:

| Metric / Parameter | Value / Detail |
| --- | --- |
| **Selected Algorithm** | Tuned Random Forest Regressor |
| **Final Model Size on Disk** | **21.49 MB** *(Previously 146.60 MB)* |
| **Model Size Footprint Reduction** | **85.3%** |
| **Best Hyperparameters** | `n_estimators`: 200, `max_depth`: 25, `min_samples_split`: 5, `min_samples_leaf`: 2, `max_features`: 0.5, `random_state`: 42 |
| **Train R² Score** | `0.9627` |
| **Test R² Score** | `0.9085` |
| **Adjusted R² Score** | `0.9069` |
| **Mean Absolute Error (MAE)** | `0.5034 Cr` |
| **Root Mean Squared Error (RMSE)** | `1.1127 Cr` |
| **Mean Squared Error (MSE)** | `1.2382 Cr²` |
| **Mean Absolute Percentage Error (MAPE)** | `19.87%` |
| **5-Fold Cross-Validation Mean R²** | `0.8780` (Std: `0.0137`) |
| **Training Time** | `2.93 seconds` |
| **Average Inference Latency** | `4.58 ms` |
| **Single Prediction Latency** | `6.74 ms` |

---

## 7. Feature Importance

### Top 20 Features in Descending Order

| Rank | Feature Name | Importance Score | Percentage Contribution |
| --- | --- | --- | --- |
| 1 | `built_up_area` | 0.503178 | 50.32% |
| 2 | `bedRoom` | 0.110106 | 11.01% |
| 3 | `bathroom` | 0.106303 | 10.63% |
| 4 | `property_type` | 0.081998 | 8.20% |
| 5 | `sector` | 0.079563 | 7.96% |
| 6 | `servant room` | 0.019454 | 1.95% |
| 7 | `balcony` | 0.016001 | 1.60% |
| 8 | `sector_sector 26` | 0.007269 | 0.73% |
| 9 | `furnishing_type` | 0.006720 | 0.67% |
| 10 | `agePossession` | 0.005983 | 0.60% |
| 11 | `floor_category` | 0.005444 | 0.54% |
| 12 | `sector_sector 65` | 0.004801 | 0.48% |
| 13 | `luxury_category` | 0.004412 | 0.44% |
| 14 | `store room` | 0.003150 | 0.31% |
| 15 | `sector_sector 25` | 0.002705 | 0.27% |
| 16 | `sector_sector 54` | 0.002506 | 0.25% |
| 17 | `agePossession_Relatively New`| 0.002368 | 0.24% |
| 18 | `sector_sector 43` | 0.002315 | 0.23% |
| 19 | `agePossession_Old Property` | 0.002217 | 0.22% |
| 20 | `sector_manesar` | 0.001976 | 0.20% |

### Top 5 Feature Contribution Rationale
1. **`built_up_area` (50.32%)**: Real estate is primarily priced as a product of size. Larger structures require higher construction costs, material volumes, and command proportional land shares, making it the dominant baseline driver.
2. **`bedRoom` (11.01%)**: Reflects the overall utility profile and occupancy scale. It differentiates target demographics, separating compact single-resident flats from sprawling family apartments.
3. **`bathroom` (10.63%)**: Serves as a strong proxy for design luxury and layout complexity. Modern, high-end developments feature higher bathroom-to-bedroom ratios, separating luxury segments from affordable ones.
4. **`property_type` (8.20%)**: Distinguishes independent houses/villas from condominiums. Houses occupy individual land plots and retain 100% undivided land interest (UDS), commanding significant valuation premiums over vertical high-rises.
5. **`sector` (7.96%)**: Captures geographical premiums (micro-markets). A sector determines infrastructure quality, connectivity to highways, and proximity to commercial tech hubs, creating pricing multipliers.

---

## 8. Recommendation System

The property recommendation engine utilizes a structured item-to-item similarity pipeline. It indexes **246 unique building societies** and operates over three distinct similarity strategies:

### Similarity Strategies
1. **Facilities Similarity**: Extracts textual amenity lists and structural descriptions. It computes TF-IDF representations followed by **Cosine Similarity** to match properties offering similar lifestyles (e.g., swimming pools, clubhouses, backup power).
2. **Price Similarity**: Measures BHK distributions and historical pricing spreads. It calculates similarity based on overlapping pricing percentiles and bedroom layouts.
3. **Location Proximity**: Extracts geographical coordinates (Latitude & Longitude) for each society. It computes distance vectors to recommend nearby societies in adjacent sectors.

### Recommendation Dimensions
- **Algorithm**: Vector Cosine Similarity & Geographic Haversine Distance
- **Similarity Matrix Shape**: `246 x 246`
- **Total Indexed Entries**: `246 unique societies`

---

## 9. REST API Summary

FastAPI exposes the following endpoint routing tables to handle client requests:

| Method | Endpoint | Purpose | Request Payload | Response Schema |
| --- | --- | --- | --- | --- |
| **GET** | `/health` | Systems Health & Readiness | None | `{"status": "healthy"}` |
| **GET** | `/analytics` | Market Summary Analytics | None | `{"total_properties": 3554, "median_price_cr": 1.51, ...}` |
| **POST** | `/predict` | Price Valuation Estimate | `PredictionRequest` json | `PredictionResponse` (Estimated price, bounds, price/sqft) |
| **POST** | `/recommend` | Property Similarity Matching | `RecommendationRequest` json | `RecommendationResponse` (Top N recommended societies) |

---

## 10. Docker Architecture

The project relies on Docker for container orchestration, ensuring consistent execution environments from local machines to staging and production servers.

```
                           [ docker-compose.yml ]
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
[ backend Service ]                                     [ frontend Service ]
├── Build: Dockerfile.backend                           ├── Build: Dockerfile.streamlit
├── Ports: 8000:8000                                    ├── Ports: 8501:8501
└── Network: propertyiq-network                         ├── Env: BACKEND_API_URL=http://backend:8000
    ▲                                                   └── Network: propertyiq-network
    │                                                       │
    └─────────────────── Cosine API Calls ──────────────────┘
                 (Shared Bridge Network Communication)
```

### Docker Components
1. **`Dockerfile.backend`**: Packs a `python:3.11-slim` environment, installs `requirements.txt`, copies backend, config, data, database, models, src, and utils modules, exposes port `8000`, and starts the Uvicorn server.
2. **`Dockerfile.streamlit`**: Installs dependencies, copies frontend UI files (`app/`, `config/`, `data/`, `database/`, and `utils/`), exposes port `8501`, and starts Streamlit.
3. **`docker-compose.yml`**: Defines the services, enables restart policies (`unless-stopped`), establishes a shared network (`propertyiq-network` bridge), and maps ports `8000` and `8501`.
4. **Environment Variables**: The frontend container uses the environment variable `BACKEND_API_URL=http://backend:8000`. Streamlit routes its REST requests using this hostname, which Docker DNS resolves internally.

---

## 11. Deployment Summary

The application is structured to be deployment-ready.

- **Hosting Platform**: Designed for container-native hosting platforms such as Render, AWS ECS, or Google Cloud Run.
- **FastAPI Container**: Deployed using Uvicorn. The deployment URL is configured via environment variables to allow cross-origin requests.
- **Streamlit Container**: Deployed as the public web portal. It connects to the backend API using the configured `BACKEND_API_URL` environment variable.
- **Persistent Storage**: SQLite database files and metadata picks are baked directly into the Docker image layers for atomic deployments.

---

## 12. Performance Highlights

| Benchmark Indicator | Value / Achievement | Business Impact |
| --- | --- | --- |
| **Dataset Size** | `3,554+` Property Listings | Captures a representative profile of the Gurgaon real estate market |
| **Geographic Coverage** | `104` Unique Sectors | Highly localized pricing models across sectors |
| **Recommendation Scale**| `246` Recommendation Nodes | Broad similarity indexing of major residential complexes |
| **Model Comparison** | `8` Models Evaluated | Compares linear and ensemble regressor variants |
| **Test Accuracy (R²)** | **`90.85%`** | Explains over 90% of pricing variance in test sets |
| **Cross Validation (R²)** | **`87.80%`** | Demonstrates strong generalization with low overfitting |
| **Production Model Size**| **`21.49 MB`** | Enables fast deployment, caching, and low container memory footprint |
| **Model Size Reduction** | **`85.3%`** | Pruned from 146.60 MB to under 22 MB via tree hyperparameters tuning |
| **Average Inference** | **`4.58 ms`** | Supports high concurrent requests per second |

---

## 13. Resume Highlights

Here are quantifiable accomplishments ready to add to your resume:

- **ML Model Optimization**: Engineered and optimized a production **Random Forest Regressor** pipeline, reducing the model footprint on disk by **85.3%** (from **146.60 MB** to **21.49 MB**) while achieving a **0.9085 R²** score and an **MAE of 0.5034 Cr**.
- **Low-Latency Inference**: Achieved an average prediction latency of **4.58 ms** on the FastAPI backend for serving price estimates, enabling real-time valuation updates.
- **Multi-Container Architecture**: Orchestrated a production-ready containerized microservices architecture using **Docker** and **Docker Compose**, separating the FastAPI backend and Streamlit UI over a private bridge network.
- **Recommendation Engine**: Developed an item-to-item similarity recommendation model indexing **246 unique building societies** using spatial coordinates, structural metadata, and facility offerings.
- **Robust Feature Pipeline**: Designed robust ordinal and dummy encoders handling unseen/capitalized categories seamlessly at inference time.
- **Data Engineering**: Curated, preprocessed, and analyzed **3,554 properties** spanning **104 sectors** in Gurgaon, building high-generalizing cross-validation pipelines (Mean R²: **0.8780**).

---

## 14. Interview Talking Points

### 1. What is the core business objective of PropertyIQ?
To deliver real-time property valuations and multi-dimensional recommendations for properties in Gurgaon, utilizing a high-performance machine learning backend.

### 2. What dataset did you use for this project?
A processed dataset containing 3,554 Gurgaon real estate listings, including 2,804 flats and 750 houses, covering 104 unique sectors.

### 3. Why did you use log transformation on the target variable?
The target price variable is highly skewed due to luxury listings. Applying `log1p` stabilizes variance and makes the target distribution normal, leading to higher performance for regression estimators.

### 4. Why did you select Random Forest over Gradient Boosting for production?
While Gradient Boosting performed well in training, it relies on internal Cythonized loss modules (`_loss`), leading to unpickling errors across differing scikit-learn versions. Random Forest is highly compatible across Python/sklearn versions and was pruned to **21.49 MB** with an R² of **0.9085**.

### 5. How did you reduce the Random Forest model size from 146 MB to 21 MB?
The default Random Forest model builds fully grown trees with no leaf node or depth limits. By setting `max_depth=25`, `min_samples_leaf=2`, and `min_samples_split=5`, we restricted tree size and complexity, reducing the footprint by 85.3% without loss of R² accuracy.

### 6. What feature preprocessors did you combine in your pipeline?
I used `StandardScaler` for numeric values, `OrdinalEncoder` for structural categorical values, and `OneHotEncoder(drop='first')` for high-cardinality values like `sector` and `agePossession`.

### 7. How did you make the preprocessing pipeline robust against unseen categories?
I configured `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` and `OneHotEncoder(handle_unknown='ignore')`. This prevents runtime `ValueError` crashes if a user inputs a new category or formats text differently.

### 8. What is the role of FastAPI in this project?
FastAPI serves as the backend service layer. It exposes REST API endpoints, parses incoming JSON schemas using Pydantic, and handles backend prediction routing.

### 9. What is Uvicorn?
Uvicorn is an ASGI web server implementation for Python. It acts as the process manager that serves the FastAPI application.

### 10. Why did you decouple Streamlit and FastAPI into two containers?
Decoupling isolates the user interface from the heavy computation. If the UI receives heavy traffic, the Streamlit container can scale independently, while the backend API container remains responsive.

### 11. How do the frontend and backend containers communicate inside Docker?
They communicate via an internal Docker bridge network (`propertyiq-network`). The frontend connects to `http://backend:8000` using the backend service name, resolved dynamically by Docker DNS.

### 12. How does the recommendation engine compute facility similarity?
It extracts textual amenities lists, vectorizes them using a TF-IDF bag-of-words representation, and computes **Cosine Similarity** to recommend complexes with similar profiles.

### 13. How does the location-based recommendation system work?
It computes distance vectors between latitude/longitude coordinates to recommend nearby societies in adjacent sectors.

### 14. What are your final model's key evaluation metrics?
Test R²: `0.9085`, Adjusted R²: `0.9069`, MAE: `0.5034 Cr`, and a 5-Fold Cross Validation Mean R² of `0.8780`.

### 15. How did you measure prediction latency?
By taking the average runtime of `pipeline.predict()` over 100 consecutive prediction loops, yielding an average latency of `4.58 ms`.

### 16. What is the top feature contributor to price prediction?
`built_up_area` is the dominant feature, explaining `50.32%` of the model's decision-making variance.

### 17. How did you handle user authentication?
We used a SQLite database containing a `users` table, queried by Streamlit backend services at login.

### 18. What restart policies did you set for your Docker containers?
I set the `unless-stopped` policy, which ensures the containers automatically restart if they crash or if the Docker daemon restarts, unless explicitly stopped by an administrator.

### 19. What is Adjusted R² and why is it important here?
Unlike standard R² which always increases as features are added, Adjusted R² penalizes the addition of non-contributing features. Achieving an Adjusted R² of `0.9069` (very close to `0.9085`) indicates that all engineered features add value.

### 20. What is the business value of this machine learning system?
It removes pricing bias and fragmentation in real estate, offering data-driven estimations and matching recommendations.

---

## 15. Future Improvements

To transform this system into a large-scale, enterprise-ready cloud platform, we can implement the following enhancements:

- **Database Migration**: Transition from local SQLite files to a managed **PostgreSQL** instance to support concurrent database operations and audit logging.
- **State & Token Management**: Implement JSON Web Tokens (JWT) for secure authentication.
- **Caching Layer**: Integrate a **Redis** cache to store search requests and pre-calculated analytics dashboards.
- **CI/CD Automation**: Design a **GitHub Actions** workflow that triggers automated tests, code linting, and Docker image builds.
- **Model Versioning & Registry**: Deploy a model registry (e.g., **MLflow**) to track model metrics and manage model artifacts.
- **Explainable AI (XAI)**: Implement **SHAP (SHapley Additive exPlanations)** values within the UI to explain how individual parameters (such as balconies or luxury scores) impact final valuations.

---

## 16. Final Project Summary

**PropertyIQ** is a complete, production-ready, end-to-end Machine Learning software system. It demonstrates how machine learning models can be integrated into a clean multi-tier microservices architecture. By decoupling UI layout logic from ML inference code, orchestrating services with Docker Compose, and optimizing the regression estimator to a compact **21.49 MB** binary serving predictions in **4.58 ms**, the system delivers both strong performance and deployment readiness, making it a suitable centerpiece for professional software engineering portfolios.
