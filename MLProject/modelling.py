import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Mengaktifkan MLflow autolog
mlflow.sklearn.autolog()

# Memuat data bersih
df = pd.read_csv('data_tagihan_bersih.csv')

# Memisahkan fitur (X) dan target (y)
# Kolom 'status_pembayaran' adalah target yang ingin diprediksi
X = df.drop(columns=['status_pembayaran'])
y = df['status_pembayaran']

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Baseline_Model"):
    # Melatih model Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc}")