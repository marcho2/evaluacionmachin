"""
Módulo para visualizaciones de clustering.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ClusteringVisualizer:
    """Clase para generar visualizaciones de clustering."""
    
    def __init__(self, config):
        """
        Inicializa el visualizador.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.figures_dir = Path(config.get('output.figures_dir', 'outputs/figures'))
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_elbow_curve(self, metrics_df, save: bool = True) -> str:
        """
        Genera la curva del codo (inercia vs k).
        
        Args:
            metrics_df: DataFrame con métricas por k
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['k'], metrics_df['inertia'], marker='o', 
                linewidth=2, markersize=8)
        plt.xlabel('Numero de Clusters (k)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title('Metodo del codo para k optimo', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(metrics_df['k'])
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / "kmeans_elbow_curve.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Elbow curve saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_silhouette_scores(self, metrics_df, save: bool = True) -> str:
        """
        Genera gráfico de Silhouette Score vs k.
        
        Args:
            metrics_df: DataFrame con métricas por k
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['k'], metrics_df['silhouette'], marker='o',
                linewidth=2, markersize=8, color='darkorange')
        plt.xlabel('Numbero de Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score vs Numbero de Clusters', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(metrics_df['k'])
        
        # Marcar el máximo
        max_idx = metrics_df['silhouette'].idxmax()
        max_k = metrics_df.loc[max_idx, 'k']
        max_score = metrics_df.loc[max_idx, 'silhouette']
        plt.axvline(x=max_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max at k={int(max_k)}')
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / "kmeans_silhouette_scores.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Silhouette scores saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_all_metrics(self, metrics_df, save: bool = True) -> str:
        """
        Genera gráfico combinado con todas las métricas.
        
        Args:
            metrics_df: DataFrame con métricas por k
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Elbow (Inertia)
        axes[0, 0].plot(metrics_df['k'], metrics_df['inertia'], 
                       marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Numbero de Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Metodo del Codo', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(metrics_df['k'])
        
        # Silhouette Score
        axes[0, 1].plot(metrics_df['k'], metrics_df['silhouette'], 
                       marker='o', linewidth=2, markersize=8, color='darkorange')
        axes[0, 1].set_xlabel('Numbero de Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(metrics_df['k'])
        max_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        axes[0, 1].axvline(x=max_k, color='red', linestyle='--', alpha=0.7)
        
        # Calinski-Harabasz Index
        axes[1, 0].plot(metrics_df['k'], metrics_df['calinski_harabasz'], 
                       marker='o', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Numbero de Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz')
        axes[1, 0].set_title('Calinski-Harabasz', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(metrics_df['k'])
        max_k = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
        axes[1, 0].axvline(x=max_k, color='red', linestyle='--', alpha=0.7)
        
        # Davies-Bouldin Index
        axes[1, 1].plot(metrics_df['k'], metrics_df['davies_bouldin'], 
                       marker='o', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_xlabel('Numbero de Clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin')
        axes[1, 1].set_title('Davies-Bouldin', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(metrics_df['k'])
        min_k = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
        axes[1, 1].axvline(x=min_k, color='green', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / "kmeans_all_metrics.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Metricas guardadas en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_pca_clusters(self, X, labels, k: int, pca_pipeline=None, 
                         save: bool = True) -> str:
        """
        Genera visualización 2D de clusters usando PCA.
        
        Args:
            X: Datos originales
            labels: Etiquetas de cluster
            k: Número de clusters
            pca_pipeline: Pipeline con PCA para proyección
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        # Si hay pipeline PCA, usarlo para transformar
        if pca_pipeline is not None:
            X_pca = pca_pipeline.transform(X)
        else:
            # Si ya es 2D, usar directamente
            if X.shape[1] == 2:
                X_pca = X
            else:
                logger.warning("Pipeline no proporcionado, saltando visualicacion de PCA.")
                return None
        
        # Crear scatter plot
        plt.figure(figsize=(10, 8))
        
        # Usar colormap con colores distinguibles
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        
        for cluster_id in range(k):
            mask = labels == cluster_id
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[cluster_id]], 
                       label=f'Cluster {cluster_id}',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('Primer Componente Principal', fontsize=12)
        plt.ylabel('Segundo Componente Principal', fontsize=12)
        plt.title(f'K-means Clustering (k={k}) - PCA Projeccion', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"kmeans_pca_k{k}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"visualizacion de PCA cluster guardado en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_cluster_distributions(self, X, labels, feature_names, k: int,
                                  top_n_features: int = 6, save: bool = True) -> str:
        """
        Genera distribuciones de features por cluster.
        
        Args:
            X: Datos (DataFrame o array)
            labels: Etiquetas de cluster
            feature_names: Nombres de las features
            k: Número de clusters
            top_n_features: Número de features a visualizar
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        import pandas as pd
        
        # Convertir a DataFrame si es necesario
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        X_df['cluster'] = labels
        
        # Seleccionar top features (por varianza entre clusters)
        feature_variances = []
        for feat in feature_names[:min(len(feature_names), 20)]:
            cluster_means = X_df.groupby('cluster')[feat].mean()
            variance = cluster_means.var()
            feature_variances.append((feat, variance))
        
        feature_variances.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in feature_variances[:top_n_features]]
        
        # Crear subplots
        n_cols = 2
        n_rows = (top_n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        
        for idx, feat in enumerate(top_features):
            ax = axes[idx]
            
            for cluster_id in range(k):
                cluster_data = X_df[X_df['cluster'] == cluster_id][feat]
                ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster_id}',
                       bins=20, color=colors[cluster_id], edgecolor='black')
            
            ax.set_xlabel(feat)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribucion de {feat}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Ocultar axes extras
        for idx in range(top_n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"kmeans_distributions_k{k}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Distribucion de Cluster guardado en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_cluster_sizes(self, labels, k: int, save: bool = True) -> str:
        """
        Genera gráfico de barras con tamaños de clusters.
        
        Args:
            labels: Etiquetas de cluster
            k: Número de clusters
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color=plt.cm.tab10(np.linspace(0, 1, k)),
                      edgecolor='black', linewidth=1.5)
        
        # Añadir etiquetas de porcentaje
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Numero de Samples', fontsize=12)
        plt.title(f'Tamaño de Cluster (k={k})', fontsize=14, fontweight='bold')
        plt.xticks(unique)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"kmeans_cluster_sizes_k{k}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Tamaño de cluster guardado en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def generate_full_clustering_report(self, metrics_df, X, labels, k: int,
                                       feature_names, pca_pipeline=None) -> Dict[str, str]:
        """
        Genera reporte completo de clustering con todas las visualizaciones.
        
        Args:
            metrics_df: DataFrame con métricas por k
            X: Datos originales
            labels: Etiquetas de cluster del modelo seleccionado
            k: Número de clusters seleccionado
            feature_names: Nombres de las features
            pca_pipeline: Pipeline con PCA para visualización 2D
            
        Returns:
            Diccionario con rutas de todas las figuras
        """
        logger.info(f"Generando reporte de clustering para k={k}...")
        
        figures = {}
        
        # Métricas de selección de k
        figures['elbow_curve'] = self.plot_elbow_curve(metrics_df)
        figures['silhouette_scores'] = self.plot_silhouette_scores(metrics_df)
        figures['all_metrics'] = self.plot_all_metrics(metrics_df)
        
        # Visualización de clusters
        figures['pca_projection'] = self.plot_pca_clusters(X, labels, k, pca_pipeline)
        figures['cluster_sizes'] = self.plot_cluster_sizes(labels, k)
        figures['cluster_distributions'] = self.plot_cluster_distributions(
            X, labels, feature_names, k
        )
        
        logger.info(f"Reporte de clustering completado para k={k}")
        return figures