"""
CLI (Command Line Interface) para ejecutar el pipeline de ML.
"""
import click
import logging
from pathlib import Path
import sys
import os

# Configurar encoding para Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Importar módulos del proyecto
from mlproject.config import Config
from mlproject.data import DataLoader, DataValidator
from mlproject.preprocessing import PreprocessingPipeline, KMeansPipeline
from mlproject.models import SupervisedModelTrainer, KMeansAnalyzer
from mlproject.evaluation import ModelEvaluator, ClusteringVisualizer


logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MLProject - Pipeline de Machine Learning end-to-end."""
    pass


@cli.command()
@click.option('--config', required=True, help='Path to config YAML file')
def data_summary(config):
    """
    Genera un resumen del dataset y valida el esquema.
    
    Ejemplo:
        python -m mlproject.cli data-summary --config configs/default.yaml
    """
    click.echo("="*60)
    click.echo("DATA SUMMARY & VALIDATION")
    click.echo("="*60)
    
    try:
        # Cargar configuración
        cfg = Config(config)
        
        # Cargar datos
        loader = DataLoader(cfg)
        df = loader.load_data()
        
        click.echo(f"\n[OK] Dataset cargado: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validar esquema
        validator = DataValidator(cfg)
        schema_valid = validator.validate_schema(df)
        nulls_valid = validator.validate_nulls(df)
        
        if schema_valid:
            click.echo("[OK] Schema validation passed")
        else:
            click.echo("[ERROR] Schema validation failed", err=True)
        
        if nulls_valid:
            click.echo("[OK] Null values validation passed")
        else:
            click.echo("[WARN] Some columns have high null percentage")
        
        # Generar resumen
        summary = validator.generate_data_summary(df)
        
        click.echo(f"\nData Types:")
        for col, dtype in list(summary['dtypes'].items())[:10]:
            click.echo(f"  {col}: {dtype}")
        if len(summary['dtypes']) > 10:
            click.echo(f"  ... and {len(summary['dtypes']) - 10} more columns")
        
        click.echo(f"\nMissing Values:")
        missing = {k: v for k, v in summary['missing_values'].items() if v > 0}
        if missing:
            for col, count in list(missing.items())[:5]:
                pct = summary['missing_percentage'][col]
                click.echo(f"  {col}: {count} ({pct:.2f}%)")
        else:
            click.echo("  No missing values found")
        
        # Información del target
        if 'target' in summary:
            click.echo(f"\nTarget Variable: {summary['target']['name']}")
            click.echo(f"  Type: {summary['target']['type']}")
            click.echo(f"  Unique values: {summary['target']['unique_values']}")
            click.echo(f"  Distribution:")
            for value, count in summary['target']['distribution'].items():
                click.echo(f"    {value}: {count}")
        
        # Guardar reporte
        report_file = validator.save_report()
        click.echo(f"\n[OK] Validation report saved to: {report_file}")
        
        click.echo("\n" + "="*60)
        click.echo("DATA SUMMARY COMPLETED")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.exception("Error in data-summary")
        sys.exit(1)


@cli.command()
@click.option('--config', required=True, help='Path to config YAML file')
@click.option('--model', default='baseline', help='Model to train: baseline, random_forest')
def train_supervised(config, model):
    """
    Entrena un modelo supervisado con validación cruzada.
    
    Ejemplo:
        python -m mlproject.cli train-supervised --config configs/default.yaml --model baseline
    """
    click.echo("="*60)
    click.echo(f"TRAINING SUPERVISED MODEL: {model.upper()}")
    click.echo("="*60)
    
    try:
        cfg = Config(config)
        loader = DataLoader(cfg)
        df = loader.load_data()
        train_df, test_df = loader.split_data(df)
        X_train, y_train = loader.prepare_xy(train_df)
        X_test, y_test = loader.prepare_xy(test_df)
        
        click.echo(f"\n[OK] Data loaded and split")
        click.echo(f"  Train: {len(X_train)} samples")
        click.echo(f"  Test: {len(X_test)} samples")
        
        prep_pipeline = PreprocessingPipeline(cfg)
        trainer = SupervisedModelTrainer(cfg, prep_pipeline)
        
        if model == 'baseline':
            click.echo(f"\n[TRAINING] Baseline model...")
            model_obj = trainer.get_baseline_model()
            results = trainer.train_with_cv(model_obj, X_train, y_train, "baseline")
        elif model == 'random_forest':
            click.echo(f"\n[TRAINING] RandomForest with default params...")
            model_obj = trainer.get_random_forest_model()
            results = trainer.train_with_cv(model_obj, X_train, y_train, "random_forest")
        else:
            raise ValueError(f"Unknown model: {model}")
        
        click.echo(f"\n[OK] Cross-validation completed in {results['training_time']:.2f}s")
        click.echo(f"\nCV Metrics:")
        for metric, values in results['cv_results'].items():
            click.echo(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
        
        evaluator = ModelEvaluator(cfg)
        test_report = evaluator.generate_full_evaluation_report(
            results['fitted_pipeline'],
            X_test,
            y_test,
            model_name=model
        )
        
        click.echo(f"\n[OK] Test set evaluation completed")
        click.echo(f"\nTest Metrics:")
        if cfg.get('data.task_type') == 'classification':
            click.echo(f"  Balanced Accuracy: {test_report['balanced_accuracy']:.4f}")
            click.echo(f"  F1 Score: {test_report['f1_score']:.4f}")
            if 'roc_auc' in test_report:
                click.echo(f"  ROC AUC: {test_report['roc_auc']:.4f}")
        else:
            click.echo(f"  RMSE: {test_report['rmse']:.4f}")
            click.echo(f"  R2: {test_report['r2']:.4f}")
        
        model_file = trainer.save_model(results['fitted_pipeline'], model)
        click.echo(f"\n[OK] Model saved to: {model_file}")
        
        click.echo(f"\n[OK] Figures generated:")
        for fig_name, fig_path in test_report['figures'].items():
            if fig_path:
                click.echo(f"  {fig_name}: {fig_path}")
        
        click.echo("\n" + "="*60)
        click.echo("TRAINING COMPLETED")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.exception("Error in train-supervised")
        sys.exit(1)


@cli.command()
@click.option('--config', required=True, help='Path to config YAML file')
def tune_supervised(config):
    """
    Realiza tuning de hiperparámetros en 2 etapas: RandomizedSearch -> GridSearch.
    
    Ejemplo:
        python -m mlproject.cli tune-supervised --config configs/exp_rf.yaml
    """
    click.echo("="*60)
    click.echo("HYPERPARAMETER TUNING - 2 STAGES")
    click.echo("="*60)
    
    try:
        cfg = Config(config)
        loader = DataLoader(cfg)
        df = loader.load_data()
        train_df, test_df = loader.split_data(df)
        X_train, y_train = loader.prepare_xy(train_df)
        X_test, y_test = loader.prepare_xy(test_df)
        
        click.echo(f"\n[OK] Data loaded: Train={len(X_train)}, Test={len(X_test)}")
        
        prep_pipeline = PreprocessingPipeline(cfg)
        trainer = SupervisedModelTrainer(cfg, prep_pipeline)
        
        click.echo(f"\n{'='*60}")
        click.echo("STAGE 1: RandomizedSearchCV")
        click.echo("="*60)
        
        rs_results, best_model_rs = trainer.tune_randomized_search(X_train, y_train)
        
        click.echo(f"\n[OK] RandomizedSearch completado en {rs_results['training_time']:.2f}s")
        click.echo(f"  Best CV score: {rs_results['best_score']:.4f}")
        click.echo(f"  Best params:")
        for param, value in rs_results['best_params'].items():
            click.echo(f"    {param}: {value}")
        
        grid_enabled = cfg.get('supervised_models.random_forest.grid_search.enabled', True)
        
        if grid_enabled:
            click.echo(f"\n{'='*60}")
            click.echo("STAGE 2: GridSearchCV")
            click.echo("="*60)
            
            gs_results, best_model_gs = trainer.tune_grid_search(
                X_train, y_train, rs_results['best_params']
            )
            
            click.echo(f"\n[OK] GridSearch completado en {gs_results['training_time']:.2f}s")
            click.echo(f"  Mejor CV score: {gs_results['best_score']:.4f}")
            click.echo(f"  Mejores parametros:")
            for param, value in gs_results['best_params'].items():
                click.echo(f"    {param}: {value}")
            
            improvement = gs_results['best_score'] - rs_results['best_score']
            click.echo(f"\n  Mejora sobre RandomSearch: {improvement:+.4f}")
            
            best_model = best_model_gs
        else:
            click.echo(f"\n[WARN] GridSearch desabilitado en config")
            best_model = best_model_rs
        
        click.echo(f"\n{'='*60}")
        click.echo("FINAL EVALUATION ON TEST SET")
        click.echo("="*60)
        
        evaluator = ModelEvaluator(cfg)
        test_report = evaluator.generate_full_evaluation_report(
            best_model,
            X_test,
            y_test,
            model_name="random_forest_tuned"
        )
        
        click.echo(f"\nTest Metrics:")
        if cfg.get('data.task_type') == 'classification':
            click.echo(f"  Accuracy: {test_report['balanced_accuracy']:.4f}")
            click.echo(f"  F1 Score: {test_report['f1_score']:.4f}")
            if 'roc_auc' in test_report:
                click.echo(f"  ROC AUC: {test_report['roc_auc']:.4f}")
        else:
            click.echo(f"  RMSE: {test_report['rmse']:.4f}")
            click.echo(f"  R2: {test_report['r2']:.4f}")
        
        model_file = trainer.save_model(best_model, "random_forest_tuned")
        click.echo(f"\n[OK] Tuned model saved to: {model_file}")
        
        metadata = cfg.get_run_metadata(cfg.get('data.raw_path'))
        metadata['tuning'] = {
            'randomized_search': {
                'best_score': rs_results['best_score'],
                'best_params': rs_results['best_params']
            }
        }
        if grid_enabled:
            metadata['tuning']['grid_search'] = {
                'best_score': gs_results['best_score'],
                'best_params': gs_results['best_params']
            }
        
        run_id = cfg.save_run_metadata(metadata)
        click.echo(f"[OK] Run metadata guardado con ID: {run_id}")
        
        click.echo("\n" + "="*60)
        click.echo("TUNING COMPLETED")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.exception("Error en tune-supervised")
        sys.exit(1)


@cli.command()
@click.option('--config', required=True, help='Path to config YAML file')
@click.option('--method', default='silhouette', 
              help='Metodo seleccionado con k: silhouette, calinski_harabasz, davies_bouldin, elbow')
def cluster_kmeans(config, method):
    """
    Ejecuta clustering con K-means y selecciona el mejor k.
    
    Ejemplo:
        python -m mlproject.cli cluster-kmeans --config configs/exp_km.yaml --method silhouette
    """
    click.echo("="*60)
    click.echo("K-MEANS CLUSTERING")
    click.echo("="*60)
    
    try:
        cfg = Config(config)
        loader = DataLoader(cfg)
        df = loader.load_data()
        train_df, _ = loader.split_data(df)
        X_train, _ = loader.prepare_xy(train_df)
        
        click.echo(f"\n[OK] Data loaded: {len(X_train)} samples")
        
        kmeans_builder = KMeansPipeline(cfg)
        analyzer = KMeansAnalyzer(cfg, kmeans_builder)
        
        k_range = cfg.get('kmeans.k_range', [2, 3, 4, 5, 6, 7, 8, 9, 10])
        click.echo(f"\n[TRAINING] Evaluando k-means con k en {k_range}...")
        
        metrics_df = analyzer.evaluate_k_range(X_train, k_range)
        
        click.echo(f"\n[OK] K-means evaluacion completada")
        click.echo(f"\nMetrics por k:")
        click.echo(metrics_df.to_string(index=False))
        
        click.echo(f"\n[SELECTING] Mejor k usando metodo: {method}")
        best_k = analyzer.select_best_k(method=method)
        
        click.echo(f"\n[OK] Mejor k seleccionado: {best_k}")
        
        click.echo(f"\n{'='*60}")
        click.echo(f"ANALISIS Con k={best_k}")
        click.echo("="*60)
        
        centroids = analyzer.get_centroids(k=best_k, descale=True)
        click.echo(f"\nCentroides:")
        click.echo(centroids.to_string())
        
        analysis = analyzer.analyze_cluster_characteristics(X_train, k=best_k)
        click.echo(f"\nCluster sizes:")
        for cluster_id in range(best_k):
            cluster_info = analysis[f'cluster_{cluster_id}']
            click.echo(f"  Cluster {cluster_id}: {cluster_info['size']} samples ({cluster_info['percentage']:.2f}%)")
        
        top_features = analyzer.get_cluster_differences(k=best_k, top_n=5)
        click.echo(f"\nTop features diferenciadores para clusters:")
        click.echo(top_features.to_string(index=False))
        
        click.echo(f"\n[GENERATING] Visualizations...")
        visualizer = ClusteringVisualizer(cfg)
        
        pca_pipeline = kmeans_builder.build_pca_pipeline_for_visualization(n_components=2)
        pca_pipeline.fit(X_train)
        
        labels = analyzer.predict_clusters(X_train, k=best_k)
        
        figures = visualizer.generate_full_clustering_report(
            metrics_df=metrics_df,
            X=X_train,
            labels=labels,
            k=best_k,
            feature_names=kmeans_builder.numeric_features,
            pca_pipeline=pca_pipeline
        )
        
        click.echo(f"\n[OK] Visualizaciones generadas:")
        for fig_name, fig_path in figures.items():
            if fig_path:
                click.echo(f"  {fig_name}: {fig_path}")
        
        model_file = analyzer.save_model(k=best_k)
        click.echo(f"\n[OK] K-means model guardado en: {model_file}")
        
        analyzer.save_results()
        click.echo(f"[OK] Resultados guardados en: {cfg.get('output.base_dir')}")
        
        metadata = cfg.get_run_metadata(cfg.get('data.raw_path'))
        metadata['clustering'] = {
            'best_k': int(best_k),
            'selection_method': method,
            'metrics': metrics_df[metrics_df['k'] == best_k].to_dict('records')[0]
        }
        
        run_id = cfg.save_run_metadata(metadata)
        click.echo(f"[OK] Run metadata guardado con ID: {run_id}")
        
        click.echo("\n" + "="*60)
        click.echo("CLUSTERING COMPLETED")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.exception("Error en cluster-kmeans")
        sys.exit(1)


@cli.command()
@click.option('--config', required=True, help='Path to config YAML file')
def full_pipeline(config):
    """
    Ejecuta el pipeline completo: validación -> supervisado -> clustering.
    
    Ejemplo:
        python -m mlproject.cli full-pipeline --config configs/default.yaml
    """
    click.echo("="*60)
    click.echo("FULL ML PIPELINE")
    click.echo("="*60)
    
    from click.testing import CliRunner
    runner = CliRunner()
    
    click.echo("\n" + "="*60)
    click.echo("STEP 1: DATA SUMMARY & VALIDATION")
    click.echo("="*60)
    result = runner.invoke(data_summary, ['--config', config])
    if result.exit_code != 0:
        click.echo("[ERROR] Data summary fallado", err=True)
        sys.exit(1)
    
    click.echo("\n" + "="*60)
    click.echo("STEP 2: TRAIN BASELINE MODEL")
    click.echo("="*60)
    result = runner.invoke(train_supervised, ['--config', config, '--model', 'baseline'])
    if result.exit_code != 0:
        click.echo("[ERROR] Baseline training fallado", err=True)
        sys.exit(1)
    
    click.echo("\n" + "="*60)
    click.echo("STEP 3: TUNE RANDOM FOREST")
    click.echo("="*60)
    result = runner.invoke(tune_supervised, ['--config', config])
    if result.exit_code != 0:
        click.echo("[ERROR] Tuning fallado", err=True)
        sys.exit(1)
    
    click.echo("\n" + "="*60)
    click.echo("STEP 4: K-MEANS CLUSTERING")
    click.echo("="*60)
    result = runner.invoke(cluster_kmeans, ['--config', config, '--method', 'silhouette'])
    if result.exit_code != 0:
        click.echo("[ERROR] Clustering fallado", err=True)
        sys.exit(1)
    
    click.echo("\n" + "="*60)
    click.echo("[OK] FULL PIPELINE COMPLETADO CORRECTAMENTE!")
    click.echo("="*60)


if __name__ == '__main__':
    cli()