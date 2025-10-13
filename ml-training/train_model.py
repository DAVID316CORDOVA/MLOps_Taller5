import numpy as np 
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import palmerpenguins as pp
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import os
import time

# Esperar un poco para asegurar que MLflow esté completamente listo
time.sleep(10)

# ============================================================
# 1. Cargar y preparar los datos
# ============================================================
print("Cargando datos de penguins...")
df = pp.load_penguins()
df.dropna(inplace=True)

# Codificar variables categóricas
categorical_cols = ['sex', 'island']
encoder = OneHotEncoder(handle_unknown='ignore')
x = df.drop(columns=['species'])
y = df['species']
x_encoded = encoder.fit_transform(x[categorical_cols])
X_numeric = x.drop(columns=categorical_cols)
X_final = np.hstack((X_numeric.values, x_encoded.toarray()))

# Codificación simple con pandas (opcional, más legible)
df_encoded = pd.get_dummies(df, columns=['island', 'sex'])
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
df_encoded['species'] = df_encoded['species'].apply(lambda x:
                        1 if x == 'Adelie' else
                        2 if x == 'Chinstrap' else
                        3 if x == 'Gentoo' else None)

# ============================================================
# 2. Configurar MLflow
# ============================================================
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5009")
print(f"🔗 Conectando a MLflow en {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("experimento")

# ============================================================
# 3. Entrenar el modelo
# ============================================================
print("Entrenando modelo...")
df = df_encoded
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5009)
model.fit(X_train, y_train)

# ============================================================
# 4. Loguear y registrar el modelo en MLflow
# ============================================================
print("Registrando modelo en MLflow...")
with mlflow.start_run(run_name="logistic_regression_run") as run:
    # Registrar métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Log del modelo
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.head(1),  # ejemplo de entrada
        registered_model_name="reg_logistica"  # nombre del modelo en el registry
    )

    print(f"Modelo logueado con accuracy = {acc:.4f}")

# ============================================================
# 5. Promover el modelo a Producción
# ============================================================
print("Promoviendo modelo a Production...")
client = MlflowClient()

model_name = "reg_logistica"
model_version = 1  # la primera versión que acabas de crear

try:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True  # mueve versiones anteriores a Archived
    )
    print(f"Modelo {model_name} v{model_version} promovido a Production")
except Exception as e:
    print(f"Error al promover modelo: {e}")

print("¡Proceso completado exitosamente!")
