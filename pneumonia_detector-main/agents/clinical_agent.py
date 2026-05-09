import pandas as pd
import joblib


model = joblib.load("models/clinical_model.pkl")

def analyze_clinical_data(file_path: str) -> dict:
   
    df = pd.read_excel(file_path)

   
    X = df.drop("pneumonia", axis=1, errors="ignore")


    probabilities = model.predict_proba(X)

  
    pneumonia_risk = probabilities[0][1]*100

    result = {
        "agent": "Clinical Agent",
        "pneumonia_risk": round(float(pneumonia_risk), 2),
        "message": "Risk predicted using ML model"
    }

    return result
