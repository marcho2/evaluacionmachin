"""
Módulo para entrenar y hacer tuning de modelos supervisados.
"""
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    cross_validate, 
    StratifiedKFold, 
    KFold,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.metrics import make_scorer, get_scorer
import joblib
from pathlib import Path


logger = logging.getLogger(__name__)


class SupervisedModelTrainer:
    """Clase para entrenar modelos supervisados con tuning."""
    
    def __init__(self, config, preprocessing_pipeline):
        """
        Inicializa el trainer.
        
        Args:
            config: Objeto de configuración
            preprocessing_pipeline: Pipeline de preprocesamiento
        """
        self.config = config
        self.preprocessing_pipeline = preprocessing_pipeline
        self.task_type = config.get('data.task_type', 'classification')
        self.results = {}
    
    def get_baseline_model(self):
        """
        Obtiene el modelo baseline según la configuración.
        
        Returns:
            Modelo baseline instanciado
        """
        baseline_config = self.config.get('supervised_models.baseline', {})
        model_type = baseline_config.get('model_type', 'logistic_regression')
        params = baseline_config.get('params', {})
        
        if self.task_type == 'classification':
            if model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=self.config.get('random_seed', 42),
                    **params
                )
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(
                    random_state=self.config.get('random_seed', 42),
                    **params
                )
            else:
                raise ValueError(f"Unknown baseline model: {model_type}")
        else:
            if model_type == 'ridge':
                model = Ridge(random_state=self.config.get('random_seed', 42), **params)
            elif model_type == 'decision_tree':
                model = DecisionTreeRegressor(
                    random_state=self.config.get('random_seed', 42),
                    **params
                )
            else:
                raise ValueError(f"Unknown baseline model: {model_type}")
        
        logger.info(f"Baseline model created: {model_type}")
        return model
    
    def get_random_forest_model(self, **kwargs):
        """
        Obtiene un modelo RandomForest según la configuración.
        
        Returns:
            RandomForest instanciado
        """
        rf_config = self.config.get('supervised_models.random_forest', {})
        
        base_params = {
            'random_state': self.config.get('random_seed', 42),
            'n_jobs': -1
        }
        
        # Agregar class_weight si está configurado (solo clasificación)
        if self.task_type == 'classification':
            class_weight = rf_config.get('class_weight', None)
            if class_weight:
                base_params['class_weight'] = class_weight
        
        # Mezclar con parámetros adicionales
        base_params.update(kwargs)
        
        if self.task_type == 'classification':
            model = RandomForestClassifier(**base_params)
        else:
            model = RandomForestRegressor(**base_params)
        
        logger.info(f"RandomForest model created with params: {base_params}")
        return model
    
    def get_cv_strategy(self):
        """
        Obtiene la estrategia de validación cruzada.
        
        Returns:
            Objeto KFold o StratifiedKFold
        """
        cv_config = self.config.get('cross_validation', {})
        n_splits = cv_config.get('n_splits', 5)
        shuffle = cv_config.get('shuffle', True)
        random_seed = self.config.get('random_seed', 42)
        
        if self.task_type == 'classification':
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_seed
            )
            logger.info(f"Using StratifiedKFold with {n_splits} splits")
        else:
            cv = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_seed
            )
            logger.info(f"Using KFold with {n_splits} splits")
        
        return cv
    
    def get_scoring_metrics(self):
        """
        Obtiene las métricas de scoring según la tarea.
        
        Returns:
            Diccionario de métricas
        """
        if self.task_type == 'classification':
            metrics_config = self.config.get('metrics.classification', {})
            primary = metrics_config.get('primary', 'roc_auc')
            secondary = metrics_config.get('secondary', [])
            
            scoring = {primary: primary}
            for metric in secondary:
                scoring[metric] = metric
        else:
            metrics_config = self.config.get('metrics.regression', {})
            primary = metrics_config.get('primary', 'neg_root_mean_squared_error')
            secondary = metrics_config.get('secondary', [])
            
            scoring = {primary: primary}
            for metric in secondary:
                scoring[metric] = metric
        
        logger.info(f"Scoring metrics: primary={list(scoring.keys())[0]}, secondary={list(scoring.keys())[1:]}")
        return scoring, list(scoring.keys())[0]
    
    def train_with_cv(self, model, X_train, y_train, model_name: str = "model") -> Dict:
        """
        Entrena un modelo con validación cruzada.
        
        Args:
            model: Modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con resultados de CV
        """
        logger.info(f"Training {model_name} with cross-validation...")
        
        # Construir pipeline completo
        pipeline = self.preprocessing_pipeline.build_full_pipeline(model)
        
        # CV strategy y métricas
        cv = self.get_cv_strategy()
        scoring, primary_metric = self.get_scoring_metrics()
        
        # Entrenar con CV
        start_time = time.time()
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        training_time = time.time() - start_time
        
        # Procesar resultados
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'cv_results': {}
        }
        
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            results['cv_results'][metric] = {
                'mean': float(np.mean(test_scores)),
                'std': float(np.std(test_scores)),
                'scores': test_scores.tolist()
            }
        
        # Métrica primaria
        primary_mean = results['cv_results'][primary_metric]['mean']
        primary_std = results['cv_results'][primary_metric]['std']
        
        logger.info(f"{model_name} - {primary_metric}: {primary_mean:.4f} ± {primary_std:.4f}")
        logger.info(f"Training time: {training_time:.2f}s")
        
        # Fit final en todo el train set
        pipeline.fit(X_train, y_train)
        results['fitted_pipeline'] = pipeline
        
        return results
    
    def tune_randomized_search(self, X_train, y_train) -> Tuple[Dict, Any]:
        """
        Realiza tuning con RandomizedSearchCV (exploración amplia).
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Tupla (resultados, mejor_modelo)
        """
        logger.info("="*60)
        logger.info("STAGE 1: RandomizedSearchCV (broad exploration)")
        logger.info("="*60)
        
        rf_config = self.config.get('supervised_models.random_forest', {})
        rs_config = rf_config.get('randomized_search', {})
        
        n_iter = rs_config.get('n_iter', 20)
        param_distributions = rs_config.get('params', {})
        
        # Modelo base
        base_model = self.get_random_forest_model()
        
        # Pipeline
        pipeline = self.preprocessing_pipeline.build_full_pipeline(base_model)
        
        # Adaptar nombres de parámetros al pipeline
        param_dist_pipeline = {
            f'model__{k}': v for k, v in param_distributions.items()
        }
        
        # CV y métricas
        cv = self.get_cv_strategy()
        scoring, primary_metric = self.get_scoring_metrics()
        
        # RandomizedSearch
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist_pipeline,
            n_iter=n_iter,
            scoring=primary_metric,
            cv=cv,
            random_state=self.config.get('random_seed', 42),
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        logger.info(f"Running RandomizedSearchCV with {n_iter} iterations...")
        start_time = time.time()
        random_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Resultados
        results = {
            'best_score': float(random_search.best_score_),
            'best_params': random_search.best_params_,
            'training_time': training_time,
            'n_iterations': n_iter,
            'cv_results_df': pd.DataFrame(random_search.cv_results_)
        }
        
        logger.info(f"Best {primary_metric}: {results['best_score']:.4f}")
        logger.info(f"Best params: {results['best_params']}")
        logger.info(f"Time: {training_time:.2f}s")
        
        return results, random_search.best_estimator_
    
    def tune_grid_search(self, X_train, y_train, best_params_from_random: Dict) -> Tuple[Dict, Any]:
        """
        Realiza tuning refinado con GridSearchCV alrededor de los mejores parámetros.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            best_params_from_random: Mejores parámetros de RandomSearch
            
        Returns:
            Tupla (resultados, mejor_modelo)
        """
        logger.info("="*60)
        logger.info("STAGE 2: GridSearchCV (fine tuning)")
        logger.info("="*60)
        
        # Extraer parámetros del modelo (sin prefijo 'model__')
        model_params = {
            k.replace('model__', ''): v 
            for k, v in best_params_from_random.items() 
            if k.startswith('model__')
        }
        
        # Crear grid refinado alrededor de los mejores valores
        param_grid = self._create_refined_grid(model_params)
        
        # Modelo base
        base_model = self.get_random_forest_model()
        
        # Pipeline
        pipeline = self.preprocessing_pipeline.build_full_pipeline(base_model)
        
        # Adaptar nombres al pipeline
        param_grid_pipeline = {
            f'model__{k}': v for k, v in param_grid.items()
        }
        
        # CV y métricas
        cv = self.get_cv_strategy()
        scoring, primary_metric = self.get_scoring_metrics()
        
        # GridSearch
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid_pipeline,
            scoring=primary_metric,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        logger.info(f"Running GridSearchCV with refined grid...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Resultados
        results = {
            'best_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'training_time': training_time,
            'param_grid': param_grid_pipeline,
            'cv_results_df': pd.DataFrame(grid_search.cv_results_)
        }
        
        logger.info(f"Best {primary_metric}: {results['best_score']:.4f}")
        logger.info(f"Best params: {results['best_params']}")
        logger.info(f"Time: {training_time:.2f}s")
        
        return results, grid_search.best_estimator_
    
    def _create_refined_grid(self, best_params: Dict) -> Dict:
        """
        Crea un grid refinado alrededor de los mejores parámetros.
        
        Args:
            best_params: Mejores parámetros encontrados
            
        Returns:
            Grid refinado
        """
        refined_grid = {}
        
        for param, value in best_params.items():
            if param == 'n_estimators':
                # Buscar alrededor ±50
                refined_grid[param] = [
                    max(50, value - 50),
                    value,
                    value + 50
                ]
            
            elif param == 'max_depth':
                if value is None:
                    refined_grid[param] = [None, 20, 25, 30]
                else:
                    refined_grid[param] = [
                        max(5, value - 5),
                        value,
                        value + 5,
                        None
                    ]
            
            elif param == 'max_features':
                if isinstance(value, str):
                    refined_grid[param] = ['sqrt', 'log2']
                else:
                    refined_grid[param] = [
                        max(0.1, value - 0.1),
                        value,
                        min(1.0, value + 0.1)
                    ]
            
            elif param == 'min_samples_split':
                refined_grid[param] = [
                    max(2, value - 2),
                    value,
                    value + 2
                ]
            
            elif param == 'min_samples_leaf':
                refined_grid[param] = [
                    max(1, value - 1),
                    value,
                    value + 1
                ]
            
            else:
                # Mantener el valor encontrado
                refined_grid[param] = [value]
        
        logger.info(f"Refined grid: {refined_grid}")
        return refined_grid
    
    def save_model(self, pipeline, model_name: str, run_id: str = None):
        """
        Guarda el modelo entrenado.
        
        Args:
            pipeline: Pipeline a guardar
            model_name: Nombre del modelo
            run_id: ID del run
        """
        models_dir = Path(self.config.get('output.models_dir', 'outputs/models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id:
            filename = models_dir / f"{model_name}_{run_id}.joblib"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = models_dir / f"{model_name}_{timestamp}.joblib"
        
        joblib.dump(pipeline, filename)
        logger.info(f"Model saved to: {filename}")
        
        return filename