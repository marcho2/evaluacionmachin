"""
Módulo para cargar y validar datos.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class DataLoader:
    """Clase para cargar y dividir datos."""
    
    def __init__(self, config):
        """
        Inicializa el data loader.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset desde la ruta especificada.
        
        Returns:
            DataFrame con los datos cargados
        """
        data_path = Path(self.config.get('data.raw_path'))
        separator = self.config.get('data.separator', ',')
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
        
        logger.info(f"Cargando dataset desde: {data_path}")
        
        df = pd.read_csv(data_path, sep=separator)
        
        logger.info(f"Dataset cargado: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en train y test.
        
        Args:
            df: DataFrame completo
            
        Returns:
            Tupla (train_df, test_df)
        """
        test_size = self.config.get('split.test_size', 0.2)
        random_seed = self.config.get('random_seed', 42)
        stratify = self.config.get('split.stratify', False)
        shuffle = self.config.get('split.shuffle', True)
        target_col = self.config.get('data.target_column')
        
        stratify_col = None
        if stratify and target_col in df.columns:
            task_type = self.config.get('data.task_type')
            if task_type == 'classification':
                stratify_col = df[target_col]
                logger.info("Using stratified split for classification task")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            shuffle=shuffle,
            stratify=stratify_col
        )
        
        logger.info(f"Data split: train={train_df.shape[0]}, test={test_df.shape[0]}")
        
        return train_df, test_df
    
    def prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features y target.
        
        Args:
            df: DataFrame con todas las columnas
            
        Returns:
            Tupla (X, y)
        """
        target_col = self.config.get('data.target_column')
        
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Para clasificación, convertir a numérico si es necesario
        task_type = self.config.get('data.task_type')
        if task_type == 'classification' and y.dtype == 'object':
            logger.info(f"Convirtiendo objetivo a numerico. Classes: {y.unique()}")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
            
            # Guardar el encoder para referencia
            self._label_encoder = le
            logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return X, y