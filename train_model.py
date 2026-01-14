
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

CSV_PATH = "traineddata_updated.csv"      # ← your updated data file
TARGET = "Price"
PIPE_OUT = "pipe.pkl"                     # ← Streamlit loads this

# ------------------ Load ------------------
df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, "Rows:", len(df))

# Make sure Price is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# ------------------ Basic Cleaning ------------------
# Convert RAM like '8GB'
if 'Ram' in df.columns:
    df['Ram'] = df['Ram'].astype(str).str.extract(r'(\d+)').astype(float)

# Convert Weight like '1.5kg'
if 'Weight' in df.columns:
    df['Weight'] = df['Weight'].astype(str).str.replace("kg","", regex=False)
    df['Weight'] = pd.to_numeric(df['Weight'], errors="coerce")

# CPU → First 3 words
if 'CPU_name' not in df.columns and 'Cpu' in df.columns:
    df['CPU_name'] = df['Cpu'].astype(str).apply(lambda x: " ".join(x.split()[:3]))
if 'Cpu' in df.columns:
    df = df.drop(columns=['Cpu'])

# GPU Brand
if 'Gpu brand' not in df.columns and 'Gpu' in df.columns:
    df['Gpu brand'] = df['Gpu'].astype(str).apply(lambda x: x.split()[0])
if 'Gpu' in df.columns:
    df = df.drop(columns=['Gpu'])

# OS group
def fix_os(x):
    x = str(x).lower()
    if "windows" in x:
        return "Windows"
    if "mac" in x:
        return "Mac"
    return "Other"

if 'OpSys' in df.columns:
    df['OpSys'] = df['OpSys'].apply(fix_os)

# ------------------ Train/Test Split ------------------
X = df.drop(columns=[TARGET])
y = np.log(df[TARGET])    # your notebook uses log(price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# ------------------ Identify Categorical + Numeric ------------------
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical:", cat_cols)
print("Numeric:", num_cols)

# ------------------ Preprocessor ------------------
num_tf = SimpleImputer(strategy='median')

try:
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
except:
    ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ("ohe", ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
    ],
    remainder='drop'
)

# ------------------ Final Pipeline ------------------
pipe = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1
    ))
])

print("\nTraining model...")
pipe.fit(X_train, y_train)
print("Training complete.")

# ------------------ Evaluation ------------------
pred = pipe.predict(X_test)
print("R2:", metrics.r2_score(y_test, pred))
print("MAE:", metrics.mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pred)))

# ------------------ Save Pipeline ------------------
pickle.dump(pipe, open(PIPE_OUT, "wb"))
print("Saved new trained pipeline →", PIPE_OUT)
