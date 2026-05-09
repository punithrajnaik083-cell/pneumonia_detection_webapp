import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



data_path = "data/clinical/synthetic_pneumonia_clinical_data.xlsx"
df = pd.read_excel(data_path)

print("Dataset loaded successfully")
print(df.head())



X = df.drop("pneumonia", axis=1)   
y = df["pneumonia"]                



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))



model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)

print("Clinical model training completed")



y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")



joblib.dump(model, "models/clinical_model.pkl")

print("Clinical model saved as models/clinical_model.pkl")
