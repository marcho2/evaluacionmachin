"""
Módulo para validar el esquema y calidad de datos.
"""
import pandas as pd
import logging
from typing import Dict, List, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class DataValidator:
    """Clase para validar esquema y calidad de datos."""
    
    def __init__(self, config):
        """
        Inicializa el validador.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.validation_report = {}
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Valida que el DataFrame tenga las columnas esperadas.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            True si el esquema es válido
        """
        expected_cols = self.config.get('schema.expected_columns', [])
        
        if not expected_cols:
            logger.warning("No expected columns defined in config")
            return True
        
        actual_cols = set(df.columns)
        expected_set = set(expected_cols)
        
        missing_cols = expected_set - actual_cols
        extra_cols = actual_cols - expected_set
        
        self.validation_report['schema'] = {
            'valid': len(missing_cols) == 0,
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols)
        }
        
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        logger.info("Schema validation passed")
        return True
    
    def validate_nulls(self, df: pd.DataFrame) -> bool:
        """
        Valida el porcentaje de valores nulos.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            True si los nulos están dentro del límite
        """
        max_null_pct = self.config.get('schema.max_null_percentage', 0.05)
        
        null_percentages = df.isnull().sum() / len(df)
        problematic_cols = null_percentages[null_percentages > max_null_pct]
        
        self.validation_report['nulls'] = {
            'valid': len(problematic_cols) == 0,
            'max_allowed_percentage': max_null_pct,
            'problematic_columns': problematic_cols.to_dict()
        }
        
        if len(problematic_cols) > 0:
            logger.warning(f"Columns with high null percentage: {problematic_cols.to_dict()}")
            return False
        
        logger.info("Null validation passed")
        return True
    
    def generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un resumen estadístico del dataset.
        
        Args:
            df: DataFrame a resumir
            
        Returns:
            Diccionario con el resumen
        """
        summary = {
            'shape': {
                'rows': df.shape[0],
                'columns': df.shape[1]
            },
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        }
        
        # Estadísticas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Distribución de categóricas (top 5 valores)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_distribution'] = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(5).to_dict()
                summary['categorical_distribution'][col] = value_counts
        
        # Información del target
        target_col = self.config.get('data.target_column')
        if target_col and target_col in df.columns:
            summary['target'] = {
                'name': target_col,
                'type': str(df[target_col].dtype),
                'unique_values': int(df[target_col].nunique()),
                'distribution': df[target_col].value_counts().to_dict()
            }
        
        self.validation_report['data_summary'] = summary
        logger.info("Data summary generated")
        
        return summary
    
    def save_report(self, output_dir: str = None):
        """
        Guarda el reporte de validación en un archivo.
        
        Args:
            output_dir: Directorio donde guardar el reporte
        """
        if output_dir is None:
            output_dir = self.config.get('output.reports_dir', 'outputs/reports')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        return report_file