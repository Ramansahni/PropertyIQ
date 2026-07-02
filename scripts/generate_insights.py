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

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA


# In[599]:


from config.constants import PROPERTIES_DATA_PATH

df = pd.read_csv(PROPERTIES_DATA_PATH).drop(columns=['store room','floor_category','balcony'])


# In[601]:


df.head()


# In[554]:


# 0 -> unfurnished
# 1 -> semifurnished
# 2 -> furnished


# In[594]:


# Numerical = bedRoom, bathroom, built_up_area, servant room
# Ordinal = property_type, furnishing_type, luxury_category 
# OHE = sector, agePossession


# In[604]:


df['agePossession'] = df['agePossession'].replace(
    {
        'Relatively New':'new',
        'Moderately Old':'old',
        'New Property' : 'new',
        'Old Property' : 'old',
        'Under Construction' : 'under construction'
    }
)


# In[605]:


df.head()


# In[606]:


df['property_type'] = df['property_type'].replace({'flat':0,'house':1})


# In[607]:


df.head()


# In[608]:


df['luxury_category'] = df['luxury_category'].replace({'Low':0,'Medium':1,'High':2})


# In[609]:


df.head()


# In[611]:


new_df = pd.get_dummies(df,columns=['sector','agePossession'],drop_first=True)


# In[614]:


X = new_df.drop(columns=['price'])
y = new_df['price']


# In[617]:


y_log = np.log1p(y)


# In[618]:


y_log


# In[621]:


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[623]:


X_scaled = pd.DataFrame(X_scaled,columns=X.columns)


# In[624]:


X_scaled


# In[625]:


kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(LinearRegression(), X_scaled, y_log, cv=kfold, scoring='r2')


# In[626]:


scores.mean(),scores.std()


# In[675]:


lr = LinearRegression()
ridge = Ridge(alpha=0.0001)


# In[628]:


lr.fit(X_scaled,y_log)


# In[676]:


ridge.fit(X_scaled,y_log)


# In[677]:


coef_df = pd.DataFrame(ridge.coef_.reshape(1,112),columns=X.columns).stack().reset_index().drop(columns=['level_0']).rename(columns={'level_1':'feature',0:'coef'})


# In[678]:


coef_df


# In[640]:


# 1. Import necessary libraries
import statsmodels.api as sm

# 2. Add a constant to X
X_with_const = sm.add_constant(X_scaled)

# 3. Fit the model
model = sm.OLS(y_log, X_with_const).fit()

# 4. Obtain summary statistics
print(model.summary())


# In[641]:


y_log.std()


# In[682]:


X_scaled['bedRoom'].std()


# In[683]:


0.21 * (0.557/1)


# In[681]:


np.expm1(0.030)


# In[653]:


2.4726962617564903e-05 * 100


# In[ ]:




