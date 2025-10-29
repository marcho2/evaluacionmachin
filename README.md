# MLProject - Pipeline de Machine Learning End-to-End

Pipeline completo de Machine Learning para ingesta, validación, entrenamiento supervisado con tuning, clustering K-means y generación de reportes automatizados.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Uso](#uso)
- [Dataset: Bank Marketing](#dataset-bank-marketing)
- [Ejemplos](#ejemplos)
- [Salidas Generadas](#salidas-generadas)

---

## 🎯 Características

### Funcionalidades Principales

✅ **Ingesta y Validación de Datos**
- Carga de CSV con configuración flexible
- Validación de esquema (columnas esperadas, tipos, nulos)
- Generación automática de reportes de calidad

✅ **Preprocesamiento con Pipelines**
- `ColumnTransformer` para numéricas y categóricas
- Imputación configurable (median, most_frequent)
- Escalado (StandardScaler, MinMaxScaler)
- One-Hot Encoding con manejo de categorías desconocidas
- PCA/TruncatedSVD opcional (sin data leakage)

✅ **Modelos Supervisados**
- Baseline (LogisticRegression, DecisionTree)
- RandomForest con class balancing
- Validación cruzada estratificada
- **Tuning en 2 etapas**:
  1. RandomizedSearchCV (exploración amplia)
  2. GridSearchCV (refinamiento automático)

✅ **Clustering K-means**
- Evaluación de múltiples k con 4 métricas:
  - Elbow (Inercia)
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- Selección automática del mejor k
- Análisis de centroides y características

✅ **Evaluación y Visualizaciones**
- **Supervisado**: Confusion Matrix, ROC, PR Curves, Feature Importance
- **No Supervisado**: Elbow curve, Silhouette, PCA 2D, Distribuciones
- Reportes en HTML/Markdown/CSV
- Figuras en alta resolución (300 DPI)

✅ **CLI (Command Line Interface)**
- Comandos intuitivos para todo el pipeline
- Progreso visible y logging automático
- Reproducibilidad con metadata tracking

---

## 📦 Requisitos

- Python 3.8+
- Ver `requirements.txt` para dependencias

---

## 🚀 Instalación

### 1. Clonar/Descargar el proyecto

```bash
# Descargar el proyecto y navegar al directorio
cd mlproject
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar el paquete

```bash
pip install -e .
```

### 5. Verificar instalación

```bash
python -m mlproject.cli --help
```

---

## 📁 Estructura del Proyecto

```
mlproject/
├── mlproject/                  # Código fuente
│   ├── __init__.py
│   ├── cli.py                  # CLI principal
│   ├── config.py               # Gestión de configuración
│   ├── data/                   # Carga y validación
│   │   ├── loader.py
│   │   └── validator.py
│   ├── preprocessing/          # Pipelines
│   │   └── pipelines.py
│   ├── models/                 # Modelos
│   │   ├── supervised.py
│   │   └── unsupervised.py
│   ├── evaluation/             # Métricas y visualizaciones
│   │   ├── metrics.py
│   │   └── clustering_viz.py
│   └── reporting/              # Generación de reportes
│       └── report_generator.py
├── configs/                    # Archivos de configuración
│   ├── default.yaml
│   ├── exp_rf.yaml
│   └── exp_km.yaml
├── data/                       # Datasets
│   └── raw/
│       └── bank-additional-full.csv
├── outputs/                    # Salidas generadas
│   ├── models/                 # Modelos guardados
│   ├── reports/                # Reportes
│   ├── figures/                # Visualizaciones
│   └── logs/                   # Logs de ejecución
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Configuración

Todo se controla desde archivos YAML en `configs/`. Ejemplo para Bank Marketing:

```yaml
# configs/default.yaml (simplificado)
project_name: "bank_marketing_ml"
random_seed: 42

data:
  raw_path: "data/raw/bank-additional-full.csv"
  separator: ";"
  target_column: "y"
  task_type: "classification"

preprocessing:
  numeric_features:
    - age
    - duration
    - campaign
    # ... más features
  
  categorical_features:
    - job
    - marital
    - education
    # ... más features

cross_validation:
  n_splits: 5

supervised_models:
  random_forest:
    randomized_search:
      n_iter: 20
      params:
        n_estimators: [50, 100, 200, 300]
        max_depth: [5, 10, 15, 20, null]
        # ... más parámetros

kmeans:
  k_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## 💻 Uso

### Comandos CLI

#### 1. Resumen y validación de datos

```bash
python -m mlproject.cli data-summary --config configs/default.yaml
```

**Salida:**
- Validación de esquema
- Estadísticas descriptivas
- Distribución del target
- Reporte JSON guardado

---

#### 2. Entrenar modelo baseline

```bash
python -m mlproject.cli train-supervised --config configs/default.yaml --model baseline
```

**Salida:**
- Métricas de CV (mean ± std)
- Evaluación en test
- Modelo guardado en `outputs/models/`
- Figuras en `outputs/figures/`

---

#### 3. Entrenar RandomForest sin tuning

```bash
python -m mlproject.cli train-supervised --config configs/default.yaml --model random_forest
```

---

#### 4. Tuning de hiperparámetros (2 etapas)

```bash
python -m mlproject.cli tune-supervised --config configs/exp_rf.yaml
```

**Proceso:**
1. **RandomizedSearchCV**: Explora 20 combinaciones
2. **GridSearchCV**: Refina alrededor del mejor modelo
3. Evalúa en test set
4. Guarda modelo tuneado

**Salida:**
- Mejores parámetros de cada etapa
- Mejora entre etapas
- Modelo final en `outputs/models/`
- Metadata en `outputs/run_TIMESTAMP.json`

---

#### 5. Clustering K-means

```bash
# Usando Silhouette Score
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method silhouette

# Usando método del codo
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method elbow

# Otros métodos: calinski_harabasz, davies_bouldin
```

**Salida:**
- Tabla de métricas por k
- Mejor k seleccionado
- Centroides (escalados y originales)
- Análisis de características
- 6 visualizaciones generadas
- Modelo guardado

---

#### 6. Pipeline completo

```bash
python -m mlproject.cli full-pipeline --config configs/default.yaml
```

**Ejecuta en secuencia:**
1. Data summary
2. Train baseline
3. Tune RandomForest
4. K-means clustering

---

## 🏦 Dataset: Bank Marketing

### Descripción

Dataset de campañas de marketing bancario (UCI Machine Learning Repository).

**Objetivo:** Predecir si un cliente suscribirá un depósito a plazo (`y`: yes/no).

### Características

- **Samples:** ~41,188
- **Features:** 20 (numéricas y categóricas)
- **Target:** `y` (binary classification)
- **Desbalanceo:** ~11% de clase positiva

### Features Principales

**Numéricas:**
- `age`: Edad del cliente
- `duration`: Duración de la última llamada (segundos)
- `campaign`: Número de contactos en esta campaña
- `pdays`: Días desde el último contacto de campaña anterior
- `previous`: Número de contactos antes de esta campaña
- `emp.var.rate`: Tasa de variación del empleo
- `cons.price.idx`: Índice de precios al consumidor
- `cons.conf.idx`: Índice de confianza del consumidor
- `euribor3m`: Tasa Euribor a 3 meses
- `nr.employed`: Número de empleados

**Categóricas:**
- `job`: Tipo de trabajo
- `marital`: Estado civil
- `education`: Nivel educativo
- `default`: ¿Tiene crédito en default?
- `housing`: ¿Tiene préstamo hipotecario?
- `loan`: ¿Tiene préstamo personal?
- `contact`: Tipo de comunicación
- `month`: Mes del último contacto
- `day_of_week`: Día de la semana
- `poutcome`: Resultado de campaña anterior

### Descarga

```bash
# Crear directorio
mkdir -p data/raw

# Descargar dataset
# Opción 1: Desde UCI
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
unzip bank-additional.zip
mv bank-additional/bank-additional-full.csv data/raw/

# Opción 2: Descarga manual
# https://archive.ics.uci.edu/ml/datasets/bank+marketing
# Guardar bank-additional-full.csv en data/raw/
```

---

## 📊 Ejemplos

### Ejemplo Completo: Bank Marketing

```bash
# 1. Validar datos
python -m mlproject.cli data-summary --config configs/default.yaml

# 2. Entrenar baseline (LogisticRegression)
python -m mlproject.cli train-supervised --config configs/default.yaml --model baseline

# 3. Tunear RandomForest
python -m mlproject.cli tune-supervised --config configs/exp_rf.yaml

# 4. Clustering
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method silhouette

# 5. O ejecutar todo de una vez
python -m mlproject.cli full-pipeline --config configs/default.yaml
```

### Resultados Esperados

**Supervisado (Clasificación):**
- ROC AUC: ~0.79-0.80
- Balanced Accuracy: ~0.75
- F1 Score: ~0.60-0.65

**Clustering:**
- Mejor k: 3-4 clusters (depende del método)
- Clusters interpretables por características económicas y comportamiento

---

## 📤 Salidas Generadas

### Modelos
```
outputs/models/
├── baseline_20251028_120345.joblib
├── random_forest_tuned_20251028_130456.joblib
└── kmeans_k3_20251028_140512.joblib
```

### Figuras (Supervisado)
```
outputs/figures/
├── confusion_matrix_baseline.png
├── roc_curve_baseline.png
├── pr_curve_baseline.png
├── confusion_matrix_random_forest_tuned.png
├── roc_curve_random_forest_tuned.png
├── pr_curve_random_forest_tuned.png
└── feature_importance_random_forest_tuned.png
```

### Figuras (Clustering)
```
outputs/figures/
├── kmeans_elbow_curve.png
├── kmeans_silhouette_scores.png
├── kmeans_all_metrics.png
├── kmeans_pca_k3.png
├── kmeans_cluster_sizes_k3.png
└── kmeans_distributions_k3.png
```

### Reportes
```
outputs/
├── validation_report_20251028_120345.json
├── run_2025-10-28-130456.json  # Metadata del tuning
├── kmeans_metrics_20251028_140512.csv
└── kmeans_centroids_k3_20251028_140512.csv
```

### Logs
```
outputs/logs/
├── run_20251028_120345.log
├── run_20251028_130456.log
└── run_20251028_140512.log
```

---

## 📝 Notas Importantes

### Data Leakage Prevention
- ✅ Preprocesamiento (scaling, PCA) dentro del pipeline
- ✅ Transformaciones solo fiteadas en train
- ✅ Test set evaluado una sola vez al final

### Reproducibilidad
- ✅ Random seed fijado en config
- ✅ Metadata tracking (dataset hash, timestamp, params)
- ✅ Logs completos de cada run

### Métricas para Clasificación Desbalanceada
- ✅ ROC AUC (métrica principal)
- ✅ Average Precision
- ✅ Balanced Accuracy
- ✅ F1 Score weighted

---

## 🐛 Troubleshooting

### Error: "Dataset not found"
```bash
# Verificar ruta en config
cat configs/default.yaml | grep raw_path

# Verificar que el archivo existe
ls -lh data/raw/bank-additional-full.csv
```

### Error: "Missing columns"
```bash
# Verificar separador (Bank Marketing usa ";")
head -n 1 data/raw/bank-additional-full.csv
```

### Warning: "High null percentage"
```bash
# Revisar reporte de validación
cat outputs/validation_report_*.json
```

---

## 📚 Referencias

- **Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Scikit-learn:** https://scikit-learn.org/
- **Click (CLI):** https://click.palletsprojects.com/

---

## 👨‍💻 Autor

**Tu Nombre**
- Evaluación Regular 2 (30%)
- Curso: Machine Learning
- Fecha: Octubre 2025

---

## 📄 Licencia

Este proyecto es para propósitos educativos.