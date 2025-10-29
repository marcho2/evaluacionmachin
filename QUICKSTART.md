# 🚀 Quick Start Guide

Guía rápida para ejecutar el proyecto en 5 minutos.

---

## 📥 Paso 1: Preparar el entorno

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar el paquete
pip install -e .
```

---

## 📊 Paso 2: Descargar el dataset

### Opción A: Descarga manual
1. Ir a: https://archive.ics.uci.edu/ml/datasets/bank+marketing
2. Descargar `bank-additional.zip`
3. Extraer `bank-additional-full.csv`
4. Copiar a `data/raw/bank-additional-full.csv`

### Opción B: wget (Linux/Mac)
```bash
mkdir -p data/raw
cd data/raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
unzip bank-additional.zip
mv bank-additional/bank-additional-full.csv .
cd ../..
```

### Verificar
```bash
ls -lh data/raw/bank-additional-full.csv
# Debe mostrar ~4.5MB
```

---

## ⚡ Paso 3: Ejecutar el pipeline

### Opción 1: Pipeline completo (recomendado)
```bash
python -m mlproject.cli full-pipeline --config configs/default.yaml
```

Esto ejecuta en secuencia:
1. ✅ Validación de datos
2. ✅ Baseline model
3. ✅ RandomForest tuning
4. ✅ K-means clustering

**Tiempo estimado:** 10-15 minutos

---

### Opción 2: Paso a paso

#### A. Validar datos
```bash
python -m mlproject.cli data-summary --config configs/default.yaml
```

**Salida:**
- ✅ Esquema validado
- ✅ Reporte JSON en `outputs/`

---

#### B. Entrenar baseline
```bash
python -m mlproject.cli train-supervised --config configs/default.yaml --model baseline
```

**Salida:**
- ✅ Modelo guardado en `outputs/models/`
- ✅ Métricas CV
- ✅ Confusion matrix en `outputs/figures/`

**Tiempo:** ~1 min

---

#### C. Tunear RandomForest
```bash
python -m mlproject.cli tune-supervised --config configs/exp_rf.yaml
```

**Salida:**
- ✅ RandomizedSearch (30 iteraciones)
- ✅ GridSearch (refinado)
- ✅ Modelo tuneado guardado
- ✅ ROC, PR curves en `outputs/figures/`

**Tiempo:** ~8-10 min

---

#### D. Clustering K-means
```bash
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method silhouette
```

**Salida:**
- ✅ Mejor k seleccionado
- ✅ Centroides guardados
- ✅ 6 visualizaciones en `outputs/figures/`

**Tiempo:** ~2 min

---

## 📁 Paso 4: Revisar resultados

### Modelos entrenados
```bash
ls outputs/models/
```

Verás:
- `baseline_*.joblib`
- `random_forest_tuned_*.joblib`
- `kmeans_k*_*.joblib`

### Visualizaciones
```bash
ls outputs/figures/
```

Verás:
- Confusion matrices
- ROC curves
- PR curves
- Feature importance
- Elbow curve
- Silhouette scores
- PCA 2D clusters
- Y más...

### Métricas y reportes
```bash
ls outputs/
```

Verás:
- `run_*.json` (metadata)
- `validation_report_*.json`
- `kmeans_metrics_*.csv`
- `kmeans_centroids_*.csv`

### Logs
```bash
tail -f outputs/logs/run_*.log
```

---

## 📊 Resultados esperados

### Supervisado (Bank Marketing)
```
Baseline (Logistic Regression):
  ROC AUC: ~0.77-0.78
  Balanced Accuracy: ~0.72

RandomForest Tuned:
  ROC AUC: ~0.79-0.80
  Balanced Accuracy: ~0.75
  F1 Score: ~0.60-0.65
```

### Clustering
```
Best k: 3-4 clusters
Silhouette Score: ~0.25-0.35

Clusters interpretables por:
  - Variables económicas (emp.var.rate, euribor3m)
  - Comportamiento de campaña (duration, campaign)
  - Características demográficas (age)
```

---

## 🐛 Problemas comunes

### Error: "Dataset not found"
```bash
# Verificar que el archivo existe
ls data/raw/bank-additional-full.csv

# Verificar ruta en config
cat configs/default.yaml | grep raw_path
```

### Error: "Module not found"
```bash
# Reinstalar en modo editable
pip install -e .
```

### Error: "Permission denied"
```bash
# Crear directorios manualmente
mkdir -p outputs/models outputs/reports outputs/figures outputs/logs
```

### Warning: "High null percentage"
```bash
# Normal en este dataset, los nulos se imputan automáticamente
# Ver detalles en: outputs/validation_report_*.json
```

---

## 🎓 Siguientes pasos

### 1. Experimentar con hiperparámetros
Editar `configs/exp_rf.yaml`:
```yaml
randomized_search:
  n_iter: 50  # Aumentar iteraciones
  params:
    n_estimators: [100, 200, 300, 500, 1000]  # Más opciones
```

### 2. Probar otros métodos de clustering
```bash
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method elbow
python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method calinski_harabasz
```

### 3. Usar desde Python
```bash
python example_usage.py
```

### 4. Análisis personalizado
Crear un notebook Jupyter:
```python
from mlproject import Config, DataLoader
import joblib

# Cargar modelo guardado
model = joblib.load('outputs/models/random_forest_tuned_*.joblib')

# Hacer predicciones personalizadas
predictions = model.predict(X_new)
```

---

## 📚 Documentación completa

Ver `README.md` para:
- Descripción detallada de cada módulo
- Arquitectura del proyecto
- Ejemplos avanzados
- Troubleshooting completo

---

## ✅ Checklist de verificación

- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas
- [ ] Dataset descargado en `data/raw/`
- [ ] `data-summary` ejecutado correctamente
- [ ] Al menos un modelo entrenado
- [ ] Visualizaciones generadas en `outputs/figures/`
- [ ] Modelos guardados en `outputs/models/`

---

## 🎉 ¡Listo!

Si completaste todos los pasos, tienes:
- ✅ Pipeline completo funcionando
- ✅ Modelos entrenados y guardados
- ✅ Visualizaciones generadas
- ✅ Reportes y métricas listos para el informe

**Tiempo total estimado:** 15-20 minutos

Para el informe técnico, usa:
- Figuras de `outputs/figures/`
- Métricas de los logs y reportes JSON
- Tablas de hiperparámetros de los archivos de metadata

---

¿Dudas? Revisa `README.md` o los logs en `outputs/logs/`