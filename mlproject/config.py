"""
Módulo de configuración para cargar y validar archivos YAML.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import hashlib
import json
from datetime import datetime


class Config:
    """Clase para manejar la configuración del proyecto."""
    
    def __init__(self, config_path: str):
        """
        Inicializa la configuración desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga el archivo de configuración YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuracion no encontrado: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        log_dir = Path(self.config['output']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Configuracion cargada desde: {self.config_path}")
    
    def _setup_directories(self):
        """Crea los directorios de salida si no existen."""
        for key in ['base_dir', 'models_dir', 'reports_dir', 'figures_dir', 'logs_dir']:
            path = Path(self.config['output'][key])
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """
        Obtiene un valor de la configuración.
        
        Args:
            key: Clave en formato 'section.subsection.key'
            default: Valor por defecto si no existe
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_run_metadata(self, dataset_path: str = None) -> Dict[str, Any]:
        """
        Genera metadata del run para reproducibilidad.
        
        Args:
            dataset_path: Ruta al dataset para calcular hash
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_file': str(self.config_path),
            'random_seed': self.config.get('random_seed', 42),
            'project_name': self.config.get('project_name', 'ml_project'),
        }
        
        if dataset_path and Path(dataset_path).exists():
            metadata['dataset_hash'] = self._compute_file_hash(dataset_path)
            metadata['dataset_path'] = dataset_path
        
        return metadata
    
    @staticmethod
    def _compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
        """Calcula el hash SHA256 de un archivo."""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def save_run_metadata(self, metadata: Dict[str, Any], run_id: str = None):
        """Guarda la metadata del run en un archivo JSON."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        
        output_dir = Path(self.config['output']['base_dir'])
        metadata_file = output_dir / f"run_{run_id}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"metadata guardado en: {metadata_file}")
        return run_id