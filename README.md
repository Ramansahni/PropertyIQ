# PropertyIQ — Gurgaon Real Estate Predictor & Recommender

PropertyIQ is a premium, end-to-end data science and machine learning application designed to analyze, predict, and recommend real estate properties in Gurgaon, India. 

The project contains a complete data science pipeline, ranging from raw data ingestion and exploratory data analysis (EDA) to feature engineering, model selection, recommender system engineering, and a polished, fully interactive Streamlit web application.

---

## 📂 Project Architecture & Directory Structure

Here is a map of the workspace structure:

```
propertyiq/
│
├── Home.py                    # Streamlit Landing Page & Login Portal
│
├── data/
│   └── processed/             # Cleaned datasets used by the Streamlit App
│       ├── gurgaon_properties_post_feature_selection_v2.csv
│       └── latlong.csv
│
├── database/
│   └── database.db            # SQLite database for User authentication/Admin roles
│
├── models/                    # Pickled ML models, pipelines, and similarity matrices
│   ├── pipeline.pkl           # Trained regression estimator pipeline
│   ├── df.pkl                 # Feature dataframe structure definition
│   ├── cosine_sim1.pkl        # Similarity matrix: Facilities-based recommendation
│   ├── cosine_sim2.pkl        # Similarity matrix: Price-based recommendation
│   ├── cosine_sim3.pkl        # Similarity matrix: Location-based recommendation
│   └── location_df_normalized.pkl
│
├── notebooks/                 # Jupyter Notebooks detailing the data science pipeline
│   ├── data-preprocessing-flats.ipynb
│   ├── data-preprocessing-houses.ipynb
│   ├── merge-flats-and-house.ipynb
│   ├── data-preprocessing-level-2.ipynb
│   ├── eda-univariate-analysis.ipynb
│   ├── eda-multivariate-analysis.ipynb
│   ├── eda-pandas-profiling.ipynb
│   ├── data-visualization.ipynb
│   ├── outlier-treatment.ipynb
│   ├── missing-value-imputation.ipynb
│   ├── feature-engineering.ipynb
│   ├── feature-selection.ipynb
│   ├── baseline model.ipynb
│   ├── model-selection.ipynb
│   ├── recommender-system.ipynb
│   └── insights-module.ipynb
│
├── pages/                     # Streamlit Multipage application scripts
│   ├── 1_Price_Prediction.py  # AI Price Estimator Page
│   ├── 2_Analytics.py         # Sector map and market metrics dashboard
│   ├── 3_Recommendations.py   # Similarity search engine
│   ├── 4_Admin_Dashboard.py   # Admin controls
│   └── 4_Profile.py           # User session details
│
├── scripts/                   # Production scripts
│   ├── export_models.py       # Serializes/Pickles models and matrices
│   ├── insights-module.py     # Analyzes features and builds analytics
│   ├── model-selection.py     # Executes model training and evaluation
│   └── recommender-system.py  # Builds recommender models and similarities
│
└── utils/                     # Backend helpers and services
    ├── auth.py                # Captcha, password hashing, and DB operations
    ├── data_loader.py         # File caching, loaders, and latlong parsing
    ├── model.py               # Model prediction wrappers
    ├── recommender.py         # Computes facility, location, and price similarity
    └── theme.py               # Glassmorphic CSS style injector and Plotly dark theme
```

---

## 🛠️ What We Have Done in This Project

We have built a comprehensive data pipeline from raw data to a fully realized, styled web application:

### 1. Data Processing & Exploration (Jupyter Notebooks)
* **Pre-processing**: Processed separate datasets for apartments (`flats.csv`) and independent houses (`houses.csv`), cleaned values, normalized fields, and merged them into a unified dataset.
* **Exploratory Data Analysis (EDA)**: Conducted univariate and multivariate analysis, generated a Pandas Profiling Report, and created scatter, box, and histogram representations to discover correlations between variables.
* **Outlier & Missing Value Treatment**: Reconciled empty values using imputation strategies and treated outliers in price, area, and bedroom metrics to prevent machine learning distortions.
* **Feature Engineering & Selection**: Encoded structural types, floor categories, age of possession, and created luxury indices. Selected top predictor columns using statistical selection criteria.

### 2. Machine Learning & Model Export
* Trained and compared regression estimators (Linear Regression, Random Forest, Gradient Boosting, LightGBM, XGBoost).
* Created a unified preprocessing and estimation `pipeline.pkl` that applies log transforms to prices to produce highly accurate predictive mappings.
* Formulated similarity matrices (using Cosine Similarity) across property facilities, pricing structures, and locations for the custom recommender engine.

### 3. Streamlit SaaS Portal Development
* **Modern CSS Design**: Injected a custom glassmorphic styling system (`utils/theme.py`) featuring dark mode designs, card shadows, responsive grids, and typography enhancements.
* **Robust Security Suite**: Developed SQLite authentication (`utils/auth.py`) that includes password hashing (`bcrypt`), admin roles, session memory management, and visual mathematical captchas to prevent spam attempts.
* **Interactive Tooling**:
    * **Price Predictor**: Users configure 12 variables in a sidebar to generate instant price calculations.
    * **Market Analytics**: Plots interactive Scatter Mapbox views mapping average sector costs alongside custom BHK distributions.
    * **Recommendations**: Users enter their target building and retrieve similar matching suggestions based on price, location, or facilities.
