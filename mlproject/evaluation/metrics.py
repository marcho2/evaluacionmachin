"""
Módulo para evaluación de modelos y generación de métricas.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """Clase para evaluar modelos y generar métricas/visualizaciones."""
    
    def __init__(self, config):
        """
        Inicializa el evaluador.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.task_type = config.get('data.task_type', 'classification')
        self.figures_dir = Path(config.get('output.figures_dir', 'outputs/figures'))
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, 
                               model_name: str = "model") -> Dict[str, Any]:
        """
        Evalúa un modelo de clasificación.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            y_pred_proba: Probabilidades predichas (para ROC/PR)
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con todas las métricas
        """
        logger.info(f"Evaluando modelo de clasificacion: {model_name}")
        
        results = {
            'model_name': model_name,
            'task_type': 'classification'
        }
        
        # Métricas básicas
        results['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        results['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
        
        # ROC AUC y Average Precision (si hay probabilidades)
        if y_pred_proba is not None:
            # Para clasificación binaria
            if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
                if y_pred_proba.ndim == 2:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba
                
                results['roc_auc'] = float(roc_auc_score(y_true, y_scores))
                results['average_precision'] = float(average_precision_score(y_true, y_scores))
            else:
                # Multiclase: usar promedio
                results['roc_auc'] = float(roc_auc_score(
                    y_true, y_pred_proba, 
                    multi_class='ovr', 
                    average='weighted'
                ))
        
        # Classification report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        results['classification_report'] = report_dict
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
        if 'roc_auc' in results:
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def evaluate_regression(self, y_true, y_pred, model_name: str = "model") -> Dict[str, Any]:
        """
        Evalúa un modelo de regresión.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones del modelo
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con todas las métricas
        """
        logger.info(f"Evaluando modelo de regresion: {model_name}")
        
        results = {
            'model_name': model_name,
            'task_type': 'regression'
        }
        
        # Métricas de regresión
        results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        results['mae'] = float(mean_absolute_error(y_true, y_pred))
        results['r2'] = float(r2_score(y_true, y_pred))
        results['mse'] = float(mean_squared_error(y_true, y_pred))
        
        # Calcular MAPE si no hay ceros en y_true
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            results['mape'] = float(mape)
        
        logger.info(f"RMSE: {results['rmse']:.4f}")
        logger.info(f"MAE: {results['mae']:.4f}")
        logger.info(f"R2: {results['r2']:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "model",
                             save: bool = True) -> str:
        """
        Genera y guarda la matriz de confusión.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            model_name: Nombre del modelo
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"confusion_matrix_{model_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de Confusion guardado en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name: str = "model",
                      save: bool = True) -> str:
        """
        Genera y guarda la curva ROC.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            model_name: Nombre del modelo
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        # Para clasificación binaria
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"roc_curve_{model_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Curva ROC guardada en: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                    model_name: str = "model",
                                    save: bool = True) -> str:
        """
        Genera y guarda la curva Precision-Recall.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            model_name: Nombre del modelo
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        # Para clasificación binaria
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"pr_curve_{model_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_feature_importance(self, feature_names, importances, 
                               model_name: str = "model", top_n: int = 20,
                               save: bool = True) -> str:
        """
        Genera y guarda el gráfico de feature importance.
        
        Args:
            feature_names: Nombres de las features
            importances: Importancias de las features
            model_name: Nombre del modelo
            top_n: Número de features a mostrar
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        # Ordenar features por importancia
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_importances)), top_importances, align='center')
        plt.yticks(range(len(top_importances)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"feature_importance_{model_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_learning_curve(self, train_scores, val_scores, param_name: str,
                           param_values, model_name: str = "model",
                           save: bool = True) -> str:
        """
        Genera y guarda curva de aprendizaje para tuning.
        
        Args:
            train_scores: Scores de entrenamiento
            val_scores: Scores de validación
            param_name: Nombre del parámetro
            param_values: Valores del parámetro
            model_name: Nombre del modelo
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, train_scores, label='Train score', marker='o')
        plt.plot(param_values, val_scores, label='Validation score', marker='o')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"learning_curve_{model_name}_{param_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def plot_residuals(self, y_true, y_pred, model_name: str = "model",
                      save: bool = True) -> str:
        """
        Genera y guarda gráfico de residuales (para regresión).
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones del modelo
            model_name: Nombre del modelo
            save: Si guardar la figura
            
        Returns:
            Ruta del archivo guardado
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.figures_dir / f"residuals_{model_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to: {filename}")
            plt.close()
            return str(filename)
        else:
            plt.show()
            return None
    
    def generate_full_evaluation_report(self, model, X_test, y_test, 
                                       model_name: str = "model") -> Dict[str, Any]:
        """
        Genera reporte completo de evaluación con todas las métricas y gráficos.
        
        Args:
            model: Modelo entrenado (pipeline)
            X_test: Features de test
            y_test: Target de test
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con todos los resultados y rutas de figuras
        """
        logger.info(f"Generando informe de evaluacion para {model_name}...")
        
        report = {
            'model_name': model_name,
            'figures': {}
        }
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            # Probabilidades si el modelo las soporta
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
            
            # Métricas
            metrics = self.evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
            report.update(metrics)
            
            # Gráficos
            report['figures']['confusion_matrix'] = self.plot_confusion_matrix(
                y_test, y_pred, model_name
            )
            
            if y_pred_proba is not None:
                report['figures']['roc_curve'] = self.plot_roc_curve(
                    y_test, y_pred_proba, model_name
                )
                report['figures']['pr_curve'] = self.plot_precision_recall_curve(
                    y_test, y_pred_proba, model_name
                )
        
        else:  # regression
            # Métricas
            metrics = self.evaluate_regression(y_test, y_pred, model_name)
            report.update(metrics)
            
            # Gráficos
            report['figures']['residuals'] = self.plot_residuals(
                y_test, y_pred, model_name
            )
        
        # Feature importance si está disponible
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            
            # Obtener nombres de features
            try:
                from mlproject.preprocessing import PreprocessingPipeline
                preprocessor = model.named_steps['preprocessor']
                # Esto requeriría pasar el PreprocessingPipeline, simplificamos
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            except:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            report['figures']['feature_importance'] = self.plot_feature_importance(
                feature_names, importances, model_name
            )
            report['feature_importances'] = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        
        logger.info(f"Informe de evaluación completo para{model_name}")
        return report