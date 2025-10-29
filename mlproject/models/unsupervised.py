"""
Módulo para clustering con K-means y selección de k.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import joblib
from pathlib import Path


logger = logging.getLogger(__name__)


class KMeansAnalyzer:
    """Clase para análisis de clustering con K-means."""
    
    def __init__(self, config, kmeans_pipeline_builder):
        """
        Inicializa el analizador de K-means.
        
        Args:
            config: Objeto de configuración
            kmeans_pipeline_builder: Constructor de pipelines de K-means
        """
        self.config = config
        self.kmeans_pipeline_builder = kmeans_pipeline_builder
        self.results = {}
        self.best_k = None
        self.fitted_pipelines = {}
    
    def evaluate_k_range(self, X_train, k_range: List[int] = None) -> pd.DataFrame:
        """
        Evalúa múltiples valores de k y calcula métricas.
        
        Args:
            X_train: Features de entrenamiento
            k_range: Lista de valores de k a evaluar
            
        Returns:
            DataFrame con métricas por cada k
        """
        if k_range is None:
            k_range = self.config.get('kmeans.k_range', [2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        logger.info(f"Evaluating K-means for k in {k_range}")
        
        metrics_list = []
        
        for k in k_range:
            logger.info(f"Training K-means with k={k}...")
            
            # Construir y entrenar pipeline
            pipeline = self.kmeans_pipeline_builder.build_kmeans_pipeline(k)
            pipeline.fit(X_train)
            
            # Obtener datos transformados y labels
            X_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
            labels = pipeline.named_steps['kmeans'].labels_
            
            # Calcular métricas
            metrics = self._calculate_metrics(X_transformed, labels, k)
            metrics_list.append(metrics)
            
            # Guardar pipeline
            self.fitted_pipelines[k] = pipeline
            
            logger.info(f"k={k}: inertia={metrics['inertia']:.2f}, "
                       f"silhouette={metrics['silhouette']:.4f}, "
                       f"CH={metrics['calinski_harabasz']:.2f}, "
                       f"DB={metrics['davies_bouldin']:.4f}")
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame(metrics_list)
        self.results['metrics_by_k'] = results_df
        
        logger.info("K-means evaluation completed")
        return results_df
    
    def _calculate_metrics(self, X_transformed, labels, k: int) -> Dict:
        """
        Calcula todas las métricas para un valor de k.
        
        Args:
            X_transformed: Datos transformados
            labels: Etiquetas de cluster
            k: Número de clusters
            
        Returns:
            Diccionario con métricas
        """
        # Obtener el modelo KMeans del pipeline
        kmeans = self.fitted_pipelines[k].named_steps['kmeans']
        
        metrics = {
            'k': k,
            'inertia': float(kmeans.inertia_),
            'silhouette': float(silhouette_score(X_transformed, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(X_transformed, labels)),
            'davies_bouldin': float(davies_bouldin_score(X_transformed, labels)),
            'n_samples_per_cluster': self._get_cluster_sizes(labels)
        }
        
        return metrics
    
    def _get_cluster_sizes(self, labels) -> Dict:
        """Obtiene el tamaño de cada cluster."""
        unique, counts = np.unique(labels, return_counts=True)
        return {f'cluster_{i}': int(count) for i, count in zip(unique, counts)}
    
    def select_best_k(self, method: str = 'silhouette') -> int:
        """
        Selecciona el mejor k según el método especificado.
        
        Args:
            method: 'silhouette', 'calinski_harabasz', 'davies_bouldin', o 'elbow'
            
        Returns:
            Mejor valor de k
        """
        if 'metrics_by_k' not in self.results:
            raise ValueError("Must run evaluate_k_range() first")
        
        df = self.results['metrics_by_k']
        
        if method == 'silhouette':
            # Mayor es mejor
            best_k = int(df.loc[df['silhouette'].idxmax(), 'k'])
            logger.info(f"Best k by silhouette: {best_k}")
        
        elif method == 'calinski_harabasz':
            # Mayor es mejor
            best_k = int(df.loc[df['calinski_harabasz'].idxmax(), 'k'])
            logger.info(f"Best k by Calinski-Harabasz: {best_k}")
        
        elif method == 'davies_bouldin':
            # Menor es mejor
            best_k = int(df.loc[df['davies_bouldin'].idxmin(), 'k'])
            logger.info(f"Best k by Davies-Bouldin: {best_k}")
        
        elif method == 'elbow':
            # Usar método del codo (detectar punto de inflexión)
            best_k = self._detect_elbow(df['k'].values, df['inertia'].values)
            logger.info(f"Best k by elbow method: {best_k}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.best_k = best_k
        return best_k
    
    def _detect_elbow(self, k_values, inertias) -> int:
        """
        Detecta el codo en la curva de inercia usando el método de la distancia.
        
        Args:
            k_values: Valores de k
            inertias: Valores de inercia
            
        Returns:
            Valor de k en el codo
        """
        # Normalizar valores
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
        inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
        
        # Calcular distancia de cada punto a la línea que une el primero y último punto
        p1 = np.array([k_norm[0], inertia_norm[0]])
        p2 = np.array([k_norm[-1], inertia_norm[-1]])
        
        distances = []
        for i in range(len(k_norm)):
            p = np.array([k_norm[i], inertia_norm[i]])
            distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        # El codo es donde la distancia es máxima
        elbow_idx = np.argmax(distances)
        
        return int(k_values[elbow_idx])
    
    def get_centroids(self, k: int = None, descale: bool = True) -> pd.DataFrame:
        """
        Obtiene los centroides del modelo K-means.
        
        Args:
            k: Número de clusters (usa best_k si no se especifica)
            descale: Si True, des-escala los centroides a valores originales
            
        Returns:
            DataFrame con los centroides
        """
        if k is None:
            k = self.best_k
            if k is None:
                raise ValueError("Must specify k or run select_best_k() first")
        
        if k not in self.fitted_pipelines:
            raise ValueError(f"K={k} not fitted. Run evaluate_k_range() first")
        
        pipeline = self.fitted_pipelines[k]
        kmeans = pipeline.named_steps['kmeans']
        centroids = kmeans.cluster_centers_
        
        # Crear DataFrame con los centroides
        feature_names = self.kmeans_pipeline_builder.numeric_features
        centroids_df = pd.DataFrame(
            centroids,
            columns=feature_names,
            index=[f'cluster_{i}' for i in range(k)]
        )
        
        # Des-escalar si se solicita
        if descale:
            scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
            centroids_descaled = scaler.inverse_transform(centroids)
            centroids_df = pd.DataFrame(
                centroids_descaled,
                columns=feature_names,
                index=[f'cluster_{i}' for i in range(k)]
            )
            logger.info(f"Centroids descaled to original values")
        else:
            logger.info(f"Centroids in z-score (standardized)")
        
        self.results[f'centroids_k{k}'] = centroids_df
        return centroids_df
    
    def predict_clusters(self, X, k: int = None) -> np.ndarray:
        """
        Predice los clusters para nuevos datos.
        
        Args:
            X: Datos a predecir
            k: Número de clusters (usa best_k si no se especifica)
            
        Returns:
            Array con las etiquetas de cluster
        """
        if k is None:
            k = self.best_k
            if k is None:
                raise ValueError("Must specify k or run select_best_k() first")
        
        if k not in self.fitted_pipelines:
            raise ValueError(f"K={k} not fitted")
        
        pipeline = self.fitted_pipelines[k]
        labels = pipeline.predict(X)
        
        return labels
    
    def analyze_cluster_characteristics(self, X_train, k: int = None) -> Dict:
        """
        Analiza las características de cada cluster.
        
        Args:
            X_train: Datos de entrenamiento
            k: Número de clusters (usa best_k si no se especifica)
            
        Returns:
            Diccionario con análisis por cluster
        """
        if k is None:
            k = self.best_k
            if k is None:
                raise ValueError("Must specify k or run select_best_k() first")
        
        # Predecir clusters
        labels = self.predict_clusters(X_train, k)
        
        # Obtener solo features numéricas
        numeric_features = self.kmeans_pipeline_builder.numeric_features
        X_numeric = X_train[numeric_features]
        
        # Análisis por cluster
        analysis = {}
        
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_data = X_numeric[cluster_mask]
            
            analysis[f'cluster_{cluster_id}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(labels) * 100),
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
                'min': cluster_data.min().to_dict(),
                'max': cluster_data.max().to_dict()
            }
        
        self.results[f'cluster_analysis_k{k}'] = analysis
        logger.info(f"Cluster characteristics analyzed for k={k}")
        
        return analysis
    
    def get_cluster_differences(self, k: int = None, top_n: int = 5) -> pd.DataFrame:
        """
        Identifica las variables que más diferencian los clusters.
        
        Args:
            k: Número de clusters (usa best_k si no se especifica)
            top_n: Número de features más importantes a retornar
            
        Returns:
            DataFrame con las diferencias por feature
        """
        if k is None:
            k = self.best_k
        
        centroids_df = self.get_centroids(k, descale=False)  # Usar z-scores
        
        # Calcular varianza entre centroides para cada feature
        variances = centroids_df.var(axis=0).sort_values(ascending=False)
        
        # Crear DataFrame con info
        differences = pd.DataFrame({
            'feature': variances.index,
            'variance_between_clusters': variances.values,
            'mean_centroid_value': centroids_df.mean(axis=0)[variances.index].values
        }).head(top_n)
        
        logger.info(f"Top {top_n} features differentiating clusters:")
        for _, row in differences.iterrows():
            logger.info(f"  {row['feature']}: variance={row['variance_between_clusters']:.4f}")
        
        return differences
    
    def save_model(self, k: int = None, run_id: str = None):
        """
        Guarda el modelo de K-means.
        
        Args:
            k: Número de clusters (usa best_k si no se especifica)
            run_id: ID del run
        """
        if k is None:
            k = self.best_k
            if k is None:
                raise ValueError("Must specify k or run select_best_k() first")
        
        if k not in self.fitted_pipelines:
            raise ValueError(f"K={k} not fitted")
        
        models_dir = Path(self.config.get('output.models_dir', 'outputs/models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id:
            filename = models_dir / f"kmeans_k{k}_{run_id}.joblib"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = models_dir / f"kmeans_k{k}_{timestamp}.joblib"
        
        joblib.dump(self.fitted_pipelines[k], filename)
        logger.info(f"K-means model (k={k}) saved to: {filename}")
        
        return filename
    
    def save_results(self, output_dir: str = None):
        """
        Guarda todos los resultados del análisis.
        
        Args:
            output_dir: Directorio donde guardar
        """
        if output_dir is None:
            output_dir = self.config.get('output.base_dir', 'outputs')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar métricas
        if 'metrics_by_k' in self.results:
            metrics_file = output_path / f"kmeans_metrics_{timestamp}.csv"
            self.results['metrics_by_k'].to_csv(metrics_file, index=False)
            logger.info(f"Metrics saved to: {metrics_file}")
        
        # Guardar centroides
        for k in self.fitted_pipelines.keys():
            if f'centroids_k{k}' in self.results:
                centroids_file = output_path / f"kmeans_centroids_k{k}_{timestamp}.csv"
                self.results[f'centroids_k{k}'].to_csv(centroids_file)
                logger.info(f"Centroids (k={k}) saved to: {centroids_file}")
        
        # Guardar análisis de clusters
        for key, value in self.results.items():
            if key.startswith('cluster_analysis_'):
                analysis_file = output_path / f"{key}_{timestamp}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(value, f, indent=2)
                logger.info(f"Cluster analysis saved to: {analysis_file}")