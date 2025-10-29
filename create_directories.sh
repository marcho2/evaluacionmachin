#!/bin/bash
# Script para crear la estructura de directorios del proyecto

echo "Creating project directory structure..."

# Crear directorios principales
mkdir -p mlproject/data
mkdir -p mlproject/preprocessing
mkdir -p mlproject/models
mkdir -p mlproject/evaluation
mkdir -p mlproject/reporting

# Crear directorios de configuración
mkdir -p configs

# Crear directorios de datos
mkdir -p data/raw
mkdir -p data/processed

# Crear directorios de salida
mkdir -p outputs/models
mkdir -p outputs/reports
mkdir -p outputs/figures
mkdir -p outputs/logs

# Crear archivos .gitkeep para mantener directorios vacíos en git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch outputs/models/.gitkeep
touch outputs/reports/.gitkeep
touch outputs/figures/.gitkeep
touch outputs/logs/.gitkeep

echo "✓ Directory structure created successfully!"
echo ""
echo "Structure:"
echo "mlproject/"
echo "├── mlproject/           # Source code"
echo "├── configs/             # YAML configurations"
echo "├── data/"
echo "│   ├── raw/            # Raw datasets"
echo "│   └── processed/      # Processed data"
echo "└── outputs/"
echo "    ├── models/         # Trained models"
echo "    ├── reports/        # Reports (JSON, CSV)"
echo "    ├── figures/        # Visualizations"
echo "    └── logs/           # Execution logs"