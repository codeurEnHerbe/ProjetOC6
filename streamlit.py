import streamlit as st
import requests

st.title("Credit Scoring App")

st.write("Entrez les données du client :")

# Exemple simplifié : tu ajoutes les champs de ton dataset
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Revenu", min_value=0, value=50000)

if st.button("Prédire"):
    input_data = {"data": {"AGE": age, "INCOME": income}}
    response = requests.post("http://localhost:8000/predict", json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.write(f"**Prédiction :** {result['prediction']}")
        st.write(f"**Probabilité de défaut :** {result['probability']:.2f}")
    else:
        st.error("Erreur API")