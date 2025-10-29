"""
Módulo para crear pipelines de preprocesamiento con ColumnTransformer.
"""
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Clase para construir pipelines de preprocesamiento."""
    
    def __init__(self, config):
        """
        Inicializa el constructor de pipelines.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.numeric_features = config.get('preprocessing.numeric_features', [])
        self.categorical_features = config.get('preprocessing.categorical_features', [])
        self.preprocessor = None
        self.pca_transformer = None
    
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Construye el ColumnTransformer con pipelines para numéricas y categóricas.
        
        Returns:
            ColumnTransformer configurado
        """
        # Pipeline para features numéricas
        numeric_transformer = self._build_numeric_pipeline()
        
        # Pipeline para features categóricas
        categorical_transformer = self._build_categorical_pipeline()
        
        # Combinar ambos pipelines
        transformers = []
        
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
            logger.info(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}...")
        
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
            logger.info(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features[:5]}...")
        
        if not transformers:
            raise ValueError("No features specified for preprocessing")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Elimina columnas no especificadas
        )
        
        logger.info("ColumnTransformer built successfully")
        return self.preprocessor
    
    def _build_numeric_pipeline(self) -> Pipeline:
        """
        Construye el pipeline para features numéricas.
        
        Returns:
            Pipeline con imputación y escalado
        """
        steps = []
        
        # Imputación
        imputation_strategy = self.config.get('preprocessing.imputation.numeric_strategy', 'median')
        steps.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
        
        # Escalado
        scaling_method = self.config.get('preprocessing.scaling.method', 'standard')
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        steps.append(('scaler', scaler))
        
        logger.info(f"Numeric pipeline: imputation={imputation_strategy}, scaling={scaling_method}")
        
        return Pipeline(steps)
    
    def _build_categorical_pipeline(self) -> Pipeline:
        """
        Construye el pipeline para features categóricas.
        
        Returns:
            Pipeline con imputación y one-hot encoding
        """
        steps = []
        
        # Imputación
        imputation_strategy = self.config.get('preprocessing.imputation.categorical_strategy', 'most_frequent')
        steps.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
        
        # One-Hot Encoding
        steps.append(('onehot', OneHotEncoder(
            handle_unknown='ignore',  # Ignorar categorías no vistas
            sparse_output=False,
            drop='first'  # Evitar multicolinealidad
        )))
        
        logger.info(f"Categorical pipeline: imputation={imputation_strategy}, encoding=one-hot")
        
        return Pipeline(steps)
    
    def build_full_pipeline(self, estimator) -> Pipeline:
        """
        Construye el pipeline completo incluyendo preprocesamiento, PCA opcional y modelo.
        
        Args:
            estimator: Modelo de scikit-learn
            
        Returns:
            Pipeline completo
        """
        steps = []
        
        # 1. Preprocesamiento
        if self.preprocessor is None:
            self.build_preprocessor()
        steps.append(('preprocessor', self.preprocessor))
        
        # 2. PCA opcional (después del preprocesamiento)
        pca_enabled = self.config.get('preprocessing.pca.enabled', False)
        if pca_enabled:
            variance_ratio = self.config.get('preprocessing.pca.variance_ratio', 0.95)
            
            # Usar TruncatedSVD si hay categorías one-hot (matriz sparse)
            if self.categorical_features:
                pca = TruncatedSVD(
                    n_components=min(50, len(self.numeric_features) + len(self.categorical_features) - 1),
                    random_state=self.config.get('random_seed', 42)
                )
                logger.info(f"Using TruncatedSVD (for sparse matrices)")
            else:
                pca = PCA(
                    n_components=variance_ratio,
                    random_state=self.config.get('random_seed', 42)
                )
                logger.info(f"Using PCA with variance_ratio={variance_ratio}")
            
            steps.append(('pca', pca))
            self.pca_transformer = pca
        
        # 3. Modelo
        steps.append(('model', estimator))
        
        pipeline = Pipeline(steps)
        logger.info(f"Full pipeline built with {len(steps)} steps")
        
        return pipeline
    
    def get_feature_names(self, fitted_preprocessor: ColumnTransformer = None) -> list:
        """
        Obtiene los nombres de las features después del preprocesamiento.
        
        Args:
            fitted_preprocessor: ColumnTransformer ya fiteado
            
        Returns:
            Lista de nombres de features
        """
        if fitted_preprocessor is None:
            fitted_preprocessor = self.preprocessor
        
        if fitted_preprocessor is None:
            raise ValueError("Preprocessor not built or fitted yet")
        
        feature_names = []
        
        # Features numéricas mantienen su nombre
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # Features categóricas se expanden con one-hot
        if self.categorical_features:
            try:
                # Obtener el encoder del pipeline categórico
                cat_pipeline = fitted_preprocessor.named_transformers_['cat']
                encoder = cat_pipeline.named_steps['onehot']
                
                cat_feature_names = encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_feature_names)
            except Exception as e:
                logger.warning(f"Could not extract categorical feature names: {e}")
                # Fallback: usar nombres genéricos
                feature_names.extend([f"cat_{i}" for i in range(len(self.categorical_features))])
        
        return feature_names
    
    def get_pca_info(self, fitted_pipeline: Pipeline) -> dict:
        """
        Extrae información del PCA después de fitear el pipeline.
        
        Args:
            fitted_pipeline: Pipeline ya fiteado
            
        Returns:
            Diccionario con info del PCA
        """
        if 'pca' not in fitted_pipeline.named_steps:
            return None
        
        pca = fitted_pipeline.named_steps['pca']
        
        info = {
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        }
        
        if hasattr(pca, 'explained_variance_'):
            info['explained_variance'] = pca.explained_variance_.tolist()
        
        logger.info(f"PCA reduced to {pca.n_components_} components")
        logger.info(f"Total variance explained: {info['cumulative_variance'][-1]:.4f}")
        
        return info


class KMeansPipeline:
    """Clase para construir pipelines de K-means."""
    
    def __init__(self, config):
        """
        Inicializa el constructor de pipelines para K-means.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.numeric_features = config.get('preprocessing.numeric_features', [])
        self.categorical_features = config.get('preprocessing.categorical_features', [])
    
    def build_kmeans_pipeline(self, k: int):
        """
        Construye el pipeline para K-means (sin PCA, solo estandarización).
        
        Args:
            k: Número de clusters
            
        Returns:
            Pipeline de preprocesamiento + KMeans
        """
        from sklearn.cluster import KMeans
        
        steps = []
        
        # 1. Preprocesamiento (solo numéricas, escaladas)
        # Para K-means usualmente solo se usan features numéricas
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features)
            ],
            remainder='drop'
        )
        
        steps.append(('preprocessor', preprocessor))
        
        # 2. KMeans
        kmeans = KMeans(
            n_clusters=k,
            n_init=self.config.get('kmeans.n_init', 10),
            max_iter=self.config.get('kmeans.max_iter', 300),
            random_state=self.config.get('random_seed', 42)
        )
        
        steps.append(('kmeans', kmeans))
        
        pipeline = Pipeline(steps)
        logger.info(f"K-means pipeline built with k={k}")
        
        return pipeline
    
    def build_pca_pipeline_for_visualization(self, n_components: int = 2):
        """
        Construye un pipeline de PCA para visualización 2D.
        
        Args:
            n_components: Número de componentes (default: 2 para 2D)
            
        Returns:
            Pipeline de preprocesamiento + PCA
        """
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features)
            ],
            remainder='drop'
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components, random_state=self.config.get('random_seed', 42)))
        ])
        
        logger.info(f"PCA visualization pipeline built with {n_components} components")
        
        return pipeline