# MLOps Taller 4 - Despliegue de servicio de MLFlow #

**Grupo compuesto por Sebastian Rodríguez y David Córdova**

Este proyecto implementa múltiples servicios con el fin de lograr desplegar una instancia de Mlflow, utilizando los siguientes servicios:  MySQL, Jupyter, MiniIO, Postgres y FastAPI

## Características Principales

- **Arquitectura híbrida MLOps**: MLFlow server en host con infraestructura de soporte containerizada
- **Pipeline completo de ML**: Desde ingesta de datos hasta inferencia en producción con trazabilidad completa
- **Storage multi-capa especializado**:
  - Postgres para metadata de experimentos y modelos MLFlow
  - MySQL para datasets de entrenamiento y aplicación
  - MinIO como S3-compatible para artefactos (modelos, plots, logs)
- **Contenerización orquestada**: Docker Compose gestiona toda la infraestructura de servicios
- **Entorno integrado de desarrollo**: Jupyter con conectividad directa a todas las fuentes de datos
- **API de producción**: FastAPI consume modelos directamente desde MLFlow Registry
- **Tracking y versionado automático**: Experimentos, métricas y modelos registrados automáticamente
- **Configuración S3 local**: MinIO simula AWS S3 para desarrollo y testing

## Estructura del Proyecto

```
MLOps_Taller4/
├── fastapi/
│   ├── __pycache__/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── minio/
├── Notebooks/
├── Venv/
├── images/
├── work
├── docker-compose.yaml
└── README.md
└──Dockerfile
└──Requirement.txt

```

### Descripción de Componentes


**Estructura de servicios:**

- **fastapi/**:
  - **main.py**: API REST que consume modelos registrados en MLFlow para inferencia
  - **Dockerfile**: Containerización con dependencias ML y conexión a MLFlow
  - **requirements.txt**: Librerías específicas para servicio de predicciones

- **minio/**:
  - **Función**: Volume mount para almacenamiento persistente de artefactos MLFlow
  - **Contenido**: Modelos serializados, plots, logs y metadata de experimentos
  - **Acceso**: S3-compatible storage accesible desde Jupyter y MLFlow server

- **venv/**:
  - **Propósito**: Ambiente virtual Python para MLFlow server en host
  - **Librerías**: mlflow, awscli, boto3, psycopg2-binary para conexiones completas
  - **Función**: Aislamiento de dependencias del tracking server

- **work/**:
  - **Contenido**: Notebooks Jupyter con pipeline completo de ML
  - **Mount**: Volume compartido entre host y container Jupyter
  - **Acceso**: Persistencia de código y resultados de experimentación

- **images/**:
  - **Propósito**: Documentación visual del proyecto
  - **Contenido**: Screenshots de interfaces, evidencias de funcionamiento
  - **Uso**: Soporte para README y documentación técnica

**Configuración de orquestación:**

- **docker-compose.yaml**:
  - **Servicios gestionados**: MinIO, MySQL, Postgres, Jupyter, FastAPI
  - **Networking**: Red interna para comunicación inter-servicios
  - **Volúmenes persistentes**: mysql_data, postgres_data para persistencia
  - **Variables de entorno**: Configuración S3, credenciales y URIs de conexión
  - **Dependencias**: Orden de inicio optimizado para disponibilidad de servicios

**Servicios containerizados:**

- **MinIO Container**: S3-compatible storage (puertos 9000/9001)
- **MySQL Container**: Base de datos para datasets y aplicación (puerto 3306)
- **Postgres Container**: Backend store MLFlow metadata (puerto 5432)
- **Jupyter Container**: Entorno desarrollo ML con acceso completo a datos
- **FastAPI Container**: API producción conectada a MLFlow registry (puerto 8000)



## Configuración de Infraestructura

### ¿Por qué esta configuración?

**Problema original:**
- Por defecto toda la metadata de MLflow se almacena en SQLite, inadecuado para entornos de producción
- Se requiere una base de datos robusta para almacenar datasets de entrenamiento y tablas de aplicación
- Necesidad de storage escalable para artefactos de ML (modelos, plots, logs)
- FastAPI debe poder consumir modelos directamente desde MLflow Registry

**Solución implementada:**
- **Postgres** como backend store para metadata de MLflow (experimentos, runs, modelos)
- **MySQL** como base de datos principal para datasets y tablas de aplicación
- **MinIO** como S3-compatible storage para artefactos de MLflow
- **Configuración de red** que permite comunicación directa entre todos los servicios

### Componentes de Configuración

```bash
# Variables de entorno clave para la conexión de mlflow y fastapi
- MLFLOW_TRACKING_URI

**Función:** realiza la conexión entre FastAPI y Mlflow.
```

**Servicio de despligue Mlflow:**
```python -m mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflows3/artifacts \
  --host 0.0.0.0 --port 5005 --serve-artifacts
    "
```

### Docker Compose maneja automáticamente:
- Creación de redes internas
- Montaje de volúmenes persistentes
- Orden de dependencias entre servicios
- Variables de entorno para cada container


## Conexiones Configuradas

###  MySQL - Conexión entre Jupyter y MySQL mediante la librería MySQLdb
```yaml
MySQLdb.connect(
    host="mysql",          
    user="my_app_user",
    passwd="my_app_pass",  
    db="my_app_db",
    port=3306
````


### Conexión de FastAPI con Mlflow

```yaml
# Tracking server de MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5005"))
```
* Genera la conexión directa entre ambos servicios para poder generar la inferencia desde fastAPI tomando los modelos que vayamos desplegando en producción desde Mlflow
### Despliegue de Mlflow con backend en Postgres

```yaml
# Conexión Postgres
--backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db\
```



## Flujo del Pipeline

### Secuencia de Ejecución:

1. docker compose up
2. Servicios iniciando (Minio + mysql + Postgres + Jupyter + Fastapi )
3. Mlflow en entorno virtual
4. Correr el notebook principal en Jupyter
5. Validar experimentos y modelos en Mlflow
6. Realizar inferencia en Mlflow


## Explicación Notebook (ejecucion.ipynb)

Este notebook tiene todo el flujo correspondiente a la ingesta de información, entrenamiento, experimentos y paso a producción del modelo

1. **Preparación de la base de datos**
   - Elimina tablas previas (`penguins_raw` y `penguins_clean`) si existen.
   - Crea las tablas necesarias para datos crudos y limpios.

2. **Carga y limpieza de datos**
   - Inserta datos de pingüinos en la tabla `penguins_raw`.
   - Limpia y transforma los datos (One-Hot Encoding, manejo de NaN) y los inserta en `penguins_clean`.

3. **generación de experimentos**
   - Usa los datos limpios para generar múltiples experimentos con diferentes experimentos
   - Guarda todos los logs en Mlflow junto con los modelos 

4. **Paso a producción del modelo seleccionado**
   - Se ejecuta un comando para que Mlflow ponga el modelo seleccionado en producción y pueda ser consumido por la API


### Resumen del flujo

```
delete_table + delete_table_clean
         ↓
  create_table_raw
         ↓
 create_table_clean
         ↓
   insert_data
         ↓
    read_data
         ↓
   Experiments
         ↓
    production_model

```


**Resultado final:**  
Se obtiene un modelo de clasificación entrenado y validado automáticamente, listo para ser consumido desde FastAPI.


## Instrucciones de Ejecución

### Preparación Inicial

```bash
# Clonar el repositorio
git clone (https://github.com/spartron12/MLOps_Taller4)
cd MLOps_Taller4

# Limpiar entorno previo (si existe)
docker compose down -v
docker system prune -f
pkill -f "mlflow server" #Limpia todas las instancias de mflow creadas
```

### Ejecución 

```bash
# Después de la preparación inicial, simplemente:
docker compose up
```
```bash
# Generar el entorno virtual junto con las librerías necesarias para ejecutar Mlflow:
python3 -m venv venv
pip install mlflow awscli boto3 psycopg2-binary
```
```bash
# Generar el entorno virtual junto con las librerías necesarias para ejecutar Mlflow:
python3 -m venv venv
pip install mlflow awscli boto3 psycopg2-binary
python -m mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflows3/artifacts \
  --host 0.0.0.0 --port 5005 --serve-artifacts
    "
```


**Qué sucede**
- Se crean todos los contenedores necesarios
- Se despliega una instancia en Jupyter en donde debemos ejecutar el notebook 
- Se guardan los modelos en Mlflow
- La API consume los modelos para hacer la inferencia



## Acceso a Servicios

| Servicio | URL | Credenciales | Descripción |
|----------|-----|--------------|-------------|
| **Mlflow Web** | http://localhost:5005 | admin/admin | Dashboard del pipeline |
| **FastAPI Docs** | http://localhost:8000/docs | - | API de predicciones |
| **MySQL** | localhost:3306 | my_app_user/my_app_pass | Base de datos |
| **Jupyter** | http://localhost:8888 | - | Jupyter Notebook |
| **Postgres** | http://localhost:5432 | mlflow_user:mlflow_password | Postgres|
| **Minio** | http://localhost:9000 | admin:supersecret | Postgres|


## Ejecución del Proyecto

### 1. Creación del entorno virtual
![Inicio del sistema](./Imagenes/imagen1.png)

### 2. Despliegue entorno virtual
![Inicio del sistema](./Imagenes/imagen2.png)

### 3. Instalación de librerías necesarias para Mlflow
![Inicio del sistema](./Imagenes/imagen3.png)

## 4. Levantamiento de docker compose
![Inicio del sistema](./Imagenes/imagen5.png)

## 5. Levantamiento Mlflow 
![Inicio del sistema](./Imagenes/imagen6.png)


## 6. Ejecución de notebook en jupyter
![Inicio del sistema](./Imagenes/imagen8.png)

## 7. Validación de experimentos creados 
![Inicio del sistema](./Imagenes/experimentos.png)

## 8. Revisión de accuraccy
![Inicio del sistema](./Imagenes/Accuracy.png)

## 9. inferencia en FastAPI una vez creado el modelo
![Inicio del sistema](./Imagenes/Inferencia_fastapi.png)

### funciones notebook - Lógica del Pipeline

```python

#Crea las tablas en MySQL

import MySQLdb   # mysqlclient se importa como MySQLdb

connection = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

cursor = connection.cursor()

# Ejecutar tus queries
cursor.execute("DROP TABLE IF EXISTS penguins_raw;")
cursor.execute("""
CREATE TABLE penguins_raw (
    species VARCHAR(50) NULL,
    island VARCHAR(50) NULL,
    bill_length_mm DOUBLE NULL,
    bill_depth_mm DOUBLE NULL,
    flipper_length_mm DOUBLE NULL,
    body_mass_g DOUBLE NULL,
    sex VARCHAR(10) NULL,
    year INT NULL
);
""")

cursor.execute("DROP TABLE IF EXISTS penguins_clean;")
cursor.execute("""
CREATE TABLE penguins_clean (
    species INT NULL,
    bill_length_mm DOUBLE NULL,
    bill_depth_mm DOUBLE NULL,
    flipper_length_mm DOUBLE NULL,
    body_mass_g DOUBLE NULL,
    year INT NULL,
    island_Biscoe INT NULL,
    island_Dream INT NULL,
    island_Torgersen INT NULL,
    sex_female INT NULL,
    sex_male INT NULL
);
""")


connection.commit()
connection.close()

#Inserta la información cruda 

import palmerpenguins as pp
from palmerpenguins import load_penguins
import pandas as pd
"""
Función para insertar datos en la tabla MySQL
"""
TABLE_NAME = "penguins_raw"
df = pp.load_penguins()


connection = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

cursor = connection.cursor()

for _, row in df.iterrows():
    sql = f"""
    INSERT INTO {TABLE_NAME} 
    (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        row["species"],
        row["island"],
        None if pd.isna(row["bill_length_mm"]) else float(row["bill_length_mm"]),
        None if pd.isna(row["bill_depth_mm"]) else float(row["bill_depth_mm"]),
        None if pd.isna(row["flipper_length_mm"]) else float(row["flipper_length_mm"]),
        None if pd.isna(row["body_mass_g"]) else float(row["body_mass_g"]),
        row["sex"] if pd.notna(row["sex"]) else None,
        int(row["year"]),
    )
    
    cursor.execute(sql, values)

connection.commit()
connection.close()


#Trae la información desde MySQL
conn = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

query = "SELECT * FROM penguins_raw"
df = pd.read_sql(query, conn)

#Limpia los datos 

from sklearn.preprocessing import OneHotEncoder
import numpy as np

df[df.isna().any(axis=1)]
df.dropna(inplace=True)
categorical_cols = ['sex','island']
encoder = OneHotEncoder(handle_unknown='ignore')
x = df.drop(columns=['species'])
y = df['species']
x_encoded = encoder.fit_transform(x[categorical_cols])
X_numeric = x.drop(columns=categorical_cols)
X_final = np.hstack((X_numeric.values, x_encoded.toarray()))

df_encoded = pd.get_dummies(df, columns=['island','sex'])
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
df_encoded.head()
df_encoded['species'] = df_encoded['species'].apply(lambda x: 
                        1 if x == 'Adelie' else 
                        2 if x == 'Chinstrap' else 
                        3 if x == 'Gentoo' else 
                        None)
# Inserta la información limpia en MySQL
TABLE_NAME = "penguins_clean"

connection = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

cursor = connection.cursor()

insert_sql = """
    INSERT INTO penguins_clean (
        species, bill_length_mm, bill_depth_mm, flipper_length_mm, 
        body_mass_g, year, island_Biscoe, island_Dream, island_Torgersen, 
        sex_female, sex_male
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Genera los valores a insertar
values = [
    (
        int(row["species"]),
        None if pd.isna(row["bill_length_mm"]) else float(row["bill_length_mm"]),
        None if pd.isna(row["bill_depth_mm"]) else float(row["bill_depth_mm"]),
        None if pd.isna(row["flipper_length_mm"]) else float(row["flipper_length_mm"]),
        None if pd.isna(row["body_mass_g"]) else float(row["body_mass_g"]),
        int(row["year"]),
        int(row["island_Biscoe"]),
        int(row["island_Dream"]),
        int(row["island_Torgersen"]),
        int(row["sex_female"]),
        int(row["sex_male"]),
    )
    for _, row in df_encoded.iterrows()
]

# Inserta múltiples filas
cursor.executemany(insert_sql, values)

# Confirma los cambios
connection.commit()

# Cierra conexión
cursor.close()
connection.close()

#Trae la data limpia para entrenar el modelo
conn = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

query = "SELECT * FROM penguins_clean"
df_limpio = pd.read_sql(query, conn)

#Trae la data limpia para entrenar el modelo
conn = MySQLdb.connect(
    host="mysql",          # nombre del servicio en docker-compose
    user="my_app_user",
    passwd="my_app_pass",  # OJO: aquí se usa 'passwd' en lugar de 'password'
    db="my_app_db",
    port=3306
)

query = "SELECT * FROM penguins_clean"
df_limpio = pd.read_sql(query, conn)

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 
# ==============================
# 1. Conectar con tu MLflow server
# ==============================
mlflow.set_tracking_uri("http://10.43.100.83:5005")  # Ajusta según tu servidor MLflow
 
# ==============================
# 2. Preparar dataset
# ==============================
df = df_limpio  # tu DataFrame limpio
X = df.drop("species", axis=1)
y = df["species"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# ==============================
# 3. Definir modelos y sus hiperparámetros
# ==============================
model_params = {
    "logistic_regression": {
        "model": LogisticRegression,
        "params": [
            {"C": 0.1, "max_iter": 5000, "solver": "lbfgs"},
            {"C": 1, "max_iter": 5000, "solver": "lbfgs"},
            {"C": 10, "max_iter": 5000, "solver": "lbfgs"}
        ]
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": [
            {"n_estimators": 50, "max_depth": 5, "random_state": 42},
            {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            {"n_estimators": 200, "max_depth": None, "random_state": 42},
        ]
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier,
        "params": [
            {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 150, "learning_rate": 0.01, "max_depth": 4},
        ]
    },
    "adaboost": {
        "model": AdaBoostClassifier,
        "params": [
            {"n_estimators": 50, "learning_rate": 0.5},
            {"n_estimators": 100, "learning_rate": 1.0},
            {"n_estimators": 150, "learning_rate": 1.5},
        ]
    },
    "svc": {
        "model": SVC,
        "params": [
            {"C": 0.1, "kernel": "linear", "probability": True},
            {"C": 1, "kernel": "rbf", "probability": True},
            {"C": 10, "kernel": "poly", "probability": True},
        ]
    },
    "knn": {
        "model": KNeighborsClassifier,
        "params": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 7, "weights": "uniform"},
        ]
    },
    "decision_tree": {
        "model": DecisionTreeClassifier,
        "params": [
            {"max_depth": 3, "min_samples_split": 2},
            {"max_depth": 5, "min_samples_split": 4},
            {"max_depth": None, "min_samples_split": 2},
        ]
    }
}
 
# ==============================
# 4. Entrenar y loggear cada experimento por separado
# ==============================
for model_name, mp in model_params.items():
    ModelClass = mp["model"]
    for i, params in enumerate(mp["params"]):
        # Crear un experimento único por combinación
        experiment_name = f"{model_name}_exp_{i+1}"
        mlflow.set_experiment(experiment_name)  # crea o selecciona el experimento
 
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            # Entrenar modelo
            model = ModelClass(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
 
            # Log de métricas
            mlflow.log_metric("accuracy", acc)
 
            # Log de hiperparámetros
            for k, v in params.items():
                mlflow.log_param(k, v)
 
            # Log del modelo como artifact
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
 
            # Registrar en Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=f"{model_name}_model")
 
            print(f"{experiment_name} entrenado y logueado con accuracy: {acc}")

#Pasa el modelo a producción

from mlflow.tracking import MlflowClient

# Conexión al server
mlflow.set_tracking_uri("http://10.43.100.98:5005")
client = MlflowClient()

# Actualizar a producción
model_name = "reg_logistica"
model_version = 1  # la primera versión que acabas de crear

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production",
    archive_existing_versions=True  # mueve versiones anteriores a Archived
)

print(f"Modelo {model_name} v{model_version} promovido a Production")

```



## Conclusiones

Conclusiones

La implementación realizada demuestra cómo integrar de manera efectiva distintos servicios (MySQL, Postgres, MinIO, Jupyter, FastAPI y MLflow) en un entorno contenerizado con Docker Compose, logrando un flujo completo de MLOps.

Se logró establecer un pipeline automatizado que cubre todas las etapas: ingesta de datos, limpieza, entrenamiento, experimentación, registro de modelos, despliegue y consumo en producción.

El uso de MLflow con backend en Postgres y artefactos en MinIO garantiza trazabilidad, persistencia y escalabilidad en comparación con la configuración por defecto en SQLite.

La conexión entre FastAPI y MLflow permite consumir modelos directamente desde el registro, habilitando predicciones en tiempo real y demostrando un caso de uso cercano a un escenario de producción.

El notebook desarrollado constituye un ejemplo reproducible de todo el flujo de trabajo, desde la preparación de la base de datos hasta la promoción de un modelo en producción, lo que refuerza la importancia de la automatización y reproducibilidad en proyectos de ciencia de datos.

En general, este proyecto evidencia la viabilidad y relevancia de aplicar prácticas de MLOps en entornos académicos y profesionales, sirviendo como base para escalar a soluciones más complejas en el futuro.

---

**Desarrollado por:**
- Sebastian Rodríguez  
- David Córdova

**Proyecto:** MLOps Taller 4 - Mlflow
**Fecha:** Septiembre 2025
