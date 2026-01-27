import pickle
import pandas as pd
import joblib
from fastapi import FastAPI
import numpy as np

app = FastAPI(title="Credit Scoring API", version="1.0")

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

user_data = pd.read_csv("data/application_train_sample.csv").drop(columns=["TARGET"], errors='ignore')

print("Pré-calcul global des prédictions et importances...")

def clean_for_json(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def preprocess(df_input):
    df = df_input.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda s: s if s in le.classes_ else np.nan)
            
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def feature_engineering(df_input):
    df = df_input.copy()

    df['CREDIT_INCOME_PERCENT'] = np.where(
        df['AMT_INCOME_TOTAL'] != 0,
        df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
        0
    )

    df['ANNUITY_INCOME_PERCENT'] = np.where(
        df['AMT_INCOME_TOTAL'] != 0,
        df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        0
    )

    df['CREDIT_TERM'] = np.where(
        df['AMT_CREDIT'] != 0,
        df['AMT_ANNUITY'] / df['AMT_CREDIT'],
        0
    )

    df['DAYS_EMPLOYED_PERCENT'] = np.where(
        df['DAYS_BIRTH'] != 0,
        df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
        0
    )

    return df



X_eng = feature_engineering(user_data)
X_encoded = preprocess(X_eng)
X_imp = imputer.transform(X_encoded)
X_scaled = scaler.transform(X_imp)

# Calcul des prédictions globales
global_proba = model.predict_proba(X_scaled)[:, 1]
global_pred = (global_proba > 0.5).astype(int)

readable_df = X_eng.copy()
readable_df["SCORE_PROBA"] = global_proba
readable_df["PREDICTION"] = global_pred

numeric_cols = readable_df.select_dtypes(include=[np.number]).columns.tolist()
readable_df = clean_for_json(readable_df[numeric_cols])

# importance globale 
coefs = model.coef_[0]
global_local_importance = X_scaled * coefs
global_imp_mean = np.mean(np.abs(global_local_importance), axis=0)

global_feat_imp_dict = dict(zip(feature_columns, global_imp_mean))


@app.get("/predict")
def predict(user_id: int):
    if user_id not in user_data['SK_ID_CURR'].values:
        return {"error": "Client inconnu"}
        
    user_row = user_data[user_data['SK_ID_CURR'] == user_id].copy()
    
    # Pipeline de prédiction pour un user
    user_eng = feature_engineering(user_row)
    user_enc = preprocess(user_eng)
    
    user_matrix = imputer.transform(user_enc)
    user_scaled = scaler.transform(user_matrix)
    
    proba = model.predict_proba(user_scaled)[:, 1][0]
    prediction = int(proba > 0.5)
    
    # Importance Locale
    local_importances = coefs * user_scaled[0]
    importances_dict = dict(zip(feature_columns, local_importances))
    
    importances_dict = dict(sorted(importances_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "prediction": prediction,
        "probability": proba,
        "feature_importance": importances_dict
    }

@app.get("/global_importance")
def get_global_importance():
    sorted_imp = dict(sorted(global_feat_imp_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    return {"global_feature_importance": sorted_imp}

@app.get("/features_list")
def features_list():
    cols = [c for c in readable_df.columns if c not in ["SK_ID_CURR", "PREDICTION", "SCORE_PROBA"]]
    return {"features": cols}

@app.get("/plot_data")
def plot_data(user_id: int, feature1: str, feature2: str):

    if feature1 not in readable_df.columns or feature2 not in readable_df.columns:
        return {"error": "Feature inconnue"}

    sample_size = min(1000, len(readable_df))
    sample = readable_df.sample(n=sample_size, random_state=42)

    client_row = readable_df[readable_df['SK_ID_CURR'] == user_id]

    if client_row.empty:
        return {"error": "Client inconnu"}

    sample = clean_for_json(sample)
    client_row = clean_for_json(client_row)

    return {
        "data": sample[[feature1, feature2, "PREDICTION", "SCORE_PROBA"]].to_dict(orient="records"),
        "client": client_row[[feature1, feature2, "PREDICTION", "SCORE_PROBA"]].iloc[0].to_dict()
    }
