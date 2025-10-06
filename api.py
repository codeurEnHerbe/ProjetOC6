import pickle
import pandas as pd
import joblib
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
import os
import requests

app = FastAPI(title="Credit Scoring API test", version="1.0")

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

user_data = pd.read_csv("data/application_train_sample.csv").drop(columns=["TARGET"])


def preprocess(user):
    for col, le in encoders.items():
        if col in user.columns:
            user[cwol] = le.transform(user[col])
    user = pd.get_dummies(user)
    user = user.reindex(columns=feature_columns, fill_value=0)
    return user

def pipeline(user):
    user = preprocess(user)
    
    user['CREDIT_INCOME_PERCENT'] = user['AMT_CREDIT'] / user['AMT_INCOME_TOTAL']
    user['ANNUITY_INCOME_PERCENT'] = user['AMT_ANNUITY'] / user['AMT_INCOME_TOTAL']
    user['CREDIT_TERM'] = user['AMT_ANNUITY'] / user['AMT_CREDIT']
    user['DAYS_EMPLOYED_PERCENT'] = user['DAYS_EMPLOYED'] / user['DAYS_BIRTH']
    domain_features = imputer.transform(user)
    domain_features = scaler.transform(domain_features)
    proba = model.predict_proba(domain_features)[:, 1][0]
    prediction = int(proba > 0.5)
    return prediction, proba

@app.get("/predict")
def predict(user_id: int):
    user = user_data[user_data['SK_ID_CURR'] == user_id].copy()
    prediction, proba = pipeline(user)
    return {"prediction": prediction, "probability": proba}
