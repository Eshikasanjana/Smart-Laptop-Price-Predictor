# -*- coding: utf-8 -*-

# laptop_price_predictor.py
# Fully updated for laptop_data_updated.csv
# Automatically trains model + saves pipe.pkl

# laptop_price_predictor.py
# Trains model using laptop_data_updated.csv and writes pipe.pkl
# Put this file in: /Users/eshikasanjana/Documents/Laptop-Price-Predictor-main/

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # hide sklearn encoder "unknown category" warnings

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

CSV_FILE = "laptop_data_updated.csv"   # expected in same folder as this script
OUTPUT_MODEL = "pipe.pkl"
RANDOM_STATE = 42

print("Loading CSV:", CSV_FILE)
df = pd.read_csv(CSV_FILE)
print("Rows:", len(df))
if "Price" not in df.columns:
    raise ValueError("Price column not found in CSV. Ensure laptop_data_updated.csv contains Price.")

# Ensure numeric Price and drop NaNs
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"]).reset_index(drop=True)

# Basic cleaning
if "Ram" in df.columns:
    df["Ram"] = df["Ram"].astype(str).str.extract(r"(\d+)").astype(float)

if "Weight" in df.columns:
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

# Screen resolution -> PPI (if available)
if "ScreenResolution" in df.columns and "Inches" in df.columns:
    try:
        split_res = df["ScreenResolution"].str.split("x", n=1, expand=True)
        df["X_res"] = split_res[0].str.replace(",", "").str.extract(r"(\d+\.?\d*)").astype(float)
        df["Y_res"] = split_res[1].str.extract(r"(\d+\.?\d*)").astype(float)
        df["PPI"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5) / df["Inches"]
        df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"], inplace=True, errors="ignore")
    except Exception:
        pass

# CPU -> CPU_name (first 3 words)
if "Cpu" in df.columns and "CPU_name" not in df.columns:
    df["CPU_name"] = df["Cpu"].astype(str).apply(lambda x: " ".join(str(x).split()[:3]))
    df.drop(columns=["Cpu"], inplace=True, errors="ignore")

# GPU brand
if "Gpu" in df.columns and "Gpu brand" not in df.columns:
    df["Gpu brand"] = df["Gpu"].astype(str).apply(lambda x: x.split()[0] if isinstance(x, str) and x.strip() != "" else "Unknown")
    df.drop(columns=["Gpu"], inplace=True, errors="ignore")

# Memory: extract HDD/SSD sizes (best-effort)
if "Memory" in df.columns:
    mem = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
    mem = mem.str.replace("GB", "", regex=False).str.replace("TB", "000", regex=False)
    parts = mem.str.split("+", n=1, expand=True).fillna("0")
    df["_m1"] = parts[0].str.strip()
    df["_m2"] = parts[1].str.strip()
    df["_m1_num"] = df["_m1"].str.extract(r"(\d+)").astype(float, errors="ignore").fillna(0)
    df["_m2_num"] = df["_m2"].str.extract(r"(\d+)").astype(float, errors="ignore").fillna(0)
    df["HDD"] = (df["Memory"].str.contains("HDD", na=False).astype(int) * df["_m1_num"]) + (df["Memory"].str.contains("HDD", na=False).astype(int) * df["_m2_num"])
    df["SSD"] = (df["Memory"].str.contains("SSD", na=False).astype(int) * df["_m1_num"]) + (df["Memory"].str.contains("SSD", na=False).astype(int) * df["_m2_num"])
    for c in ["_m1", "_m2", "_m1_num", "_m2_num"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    df.drop(columns=["Memory"], inplace=True, errors="ignore")

# Normalize OpSys
if "OpSys" in df.columns:
    def fix_os(x):
        x = str(x).lower()
        if "windows" in x: return "Windows"
        if "mac" in x: return "Mac"
        return "Other"
    df["OpSys"] = df["OpSys"].apply(fix_os)

# Drop index columns if present
for c in list(df.columns):
    if "Unnamed" in str(c):
        df.drop(columns=[c], inplace=True, errors="ignore")

# Prepare X, y
X = df.drop(columns=["Price"])
y = np.log(df["Price"])   # use log-price as notebook did

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)

# Detect categorical and numeric
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# Build transformers (OneHotEncoder handle_unknown='ignore')
num_tf = SimpleImputer(strategy="median")
try:
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

cat_tf = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", ohe)])

transformers = []
if num_cols:
    transformers.append(("num", num_tf, num_cols))
if cat_cols:
    transformers.append(("cat", cat_tf, cat_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

pipe = Pipeline([("pre", preprocessor), ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])

print("Fitting pipeline...")
pipe.fit(X_train, y_train)
print("Evaluating...")
y_pred = pipe.predict(X_test)
print("R²:", metrics.r2_score(y_test, y_pred))
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))

# Save pipeline
with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(pipe, f)

print("Model saved as:", OUTPUT_MODEL)
