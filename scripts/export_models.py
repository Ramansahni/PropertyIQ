import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import numpy as np
import pandas as pd
import ast
import re
import json
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import category_encoders as ce

from src.recommendation.recommender import refined_parse_modified_v2, distance_to_meters
from config.constants import (
    PROPERTIES_DATA_PATH,
    APPARTMENTS_DATA_PATH,
    PIPELINE_PATH,
    DF_PATH,
    COSINE_SIM1_PATH,
    COSINE_SIM2_PATH,
    COSINE_SIM3_PATH,
    LOCATION_DF_PATH,
    TRAINED_MODELS_DIR,
    SIMILARITY_MODELS_DIR,
    METADATA_MODELS_DIR
)
from config.settings import N_ESTIMATORS

os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(SIMILARITY_MODELS_DIR, exist_ok=True)
os.makedirs(METADATA_MODELS_DIR, exist_ok=True)

print("Starting model extraction...")

# --- PRICE PREDICTION PIPELINE ---
print("Training Random Forest Pipeline...")
df = pd.read_csv(PROPERTIES_DATA_PATH)

# Preprocessing
df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
X = df.drop(columns=['price'])
y = df['price']
y_transformed = np.log1p(y)

columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
    ], 
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=N_ESTIMATORS)) # Using parameters from config
])

pipeline.fit(X, y_transformed)

with open(PIPELINE_PATH, 'wb') as file:
    pickle.dump(pipeline, file)

with open(DF_PATH, 'wb') as file:
    pickle.dump(X, file)


# --- RECOMMENDER SYSTEM ---
print("Calculating Recommendation Matrices...")
df_app = pd.read_csv(APPARTMENTS_DATA_PATH).drop(22)

# 1. Facilities
def extract_list(s):
    return re.findall(r"'(.*?)'", s)

df_app['TopFacilities'] = df_app['TopFacilities'].apply(extract_list)
df_app['FacilitiesStr'] = df_app['TopFacilities'].apply(' '.join)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_app['FacilitiesStr'])
cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)

with open(COSINE_SIM1_PATH, 'wb') as file:
    pickle.dump(cosine_sim1, file)

# 2. Price Details
data_refined = []
for _, row in df_app.iterrows():
    features = refined_parse_modified_v2(row['PriceDetails'])
    new_row = {'PropertyName': row['PropertyName']}
    for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
        new_row[f'building type_{config}'] = features.get(f'building type_{config}')
        new_row[f'area low {config}'] = features.get(f'area low {config}')
        new_row[f'area high {config}'] = features.get(f'area high {config}')
        new_row[f'price low {config}'] = features.get(f'price low {config}')
        new_row[f'price high {config}'] = features.get(f'price high {config}')
    data_refined.append(new_row)

df_final_refined_v2 = pd.DataFrame(data_refined).set_index('PropertyName')
df_final_refined_v2['building type_Land'] = df_final_refined_v2['building type_Land'].replace({'':'Land'})

categorical_columns = df_final_refined_v2.select_dtypes(include=['object']).columns.tolist()
ohe_df = pd.get_dummies(df_final_refined_v2, columns=categorical_columns, drop_first=True)
ohe_df.fillna(0,inplace=True)

scaler = StandardScaler()
ohe_df_normalized = pd.DataFrame(scaler.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)
cosine_sim2 = cosine_similarity(ohe_df_normalized)

with open(COSINE_SIM2_PATH, 'wb') as file:
    pickle.dump(cosine_sim2, file)

# 3. Location Advantages
location_matrix = {}
for index, row in df_app.iterrows():
    distances = {}
    try:
        for location, distance in ast.literal_eval(row['LocationAdvantages']).items():
            distances[location] = distance_to_meters(distance)
    except:
        pass
    location_matrix[index] = distances

location_df = pd.DataFrame.from_dict(location_matrix, orient='index')
location_df.index = df_app.PropertyName
location_df.fillna(54000,inplace=True)

scaler = StandardScaler()
location_df_normalized = pd.DataFrame(scaler.fit_transform(location_df), columns=location_df.columns, index=location_df.index)
cosine_sim3 = cosine_similarity(location_df_normalized)

with open(COSINE_SIM3_PATH, 'wb') as file:
    pickle.dump(cosine_sim3, file)

with open(LOCATION_DF_PATH, 'wb') as file:
    pickle.dump(location_df_normalized, file)

print("Export completed successfully!")
