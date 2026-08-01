import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import uuid

mlflow.sklearn.autolog(log_models=False)

csv_path = os.environ.get("CSV_URL", "data_tagihan_bersih.csv")

df = pd.read_csv(csv_path)

X = df.drop(columns=['status_pembayaran'])
y = df['status_pembayaran']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Baseline_Model"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc}")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        pip_requirements="requirements.txt" 
    )