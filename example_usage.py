"""
Script de ejemplo para usar MLProject desde Python (sin CLI).
"""
from mlproject.config import Config
from mlproject.data import DataLoader, DataValidator
from mlproject.preprocessing import PreprocessingPipeline, KMeansPipeline
from mlproject.models import SupervisedModelTrainer, KMeansAnalyzer
from mlproject.evaluation import ModelEvaluator, ClusteringVisualizer


def example_supervised_workflow():
    """Ejemplo de flujo supervisado completo."""
    print("="*60)
    print("SUPERVISED LEARNING WORKFLOW")
    print("="*60)
    
    # 1. Configuración
    config = Config('configs/exp_rf.yaml')
    
    # 2. Cargar datos
    loader = DataLoader(config)
    df = loader.load_data()
    train_df, test_df = loader.split_data(df)
    X_train, y_train = loader.prepare_xy(train_df)
    X_test, y_test = loader.prepare_xy(test_df)
    
    print(f"\n✓ Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    # 3. Validar datos
    validator = DataValidator(config)
    validator.validate_schema(df)
    validator.validate_nulls(df)
    summary = validator.generate_data_summary(df)
    validator.save_report()
    
    print(f"✓ Data validated")
    
    # 4. Crear pipeline de preprocesamiento
    prep_pipeline = PreprocessingPipeline(config)
    
    # 5. Entrenar baseline
    trainer = SupervisedModelTrainer(config, prep_pipeline)
    baseline_model = trainer.get_baseline_model()
    baseline_results = trainer.train_with_cv(baseline_model, X_train, y_train, "baseline")
    
    print(f"\n✓ Baseline trained")
    print(f"  ROC AUC: {baseline_results['cv_results']['roc_auc']['mean']:.4f}")
    
    # 6. Tuning RandomForest
    print(f"\n{'='*60}")
    print("TUNING RANDOM FOREST")
    print("="*60)
    
    # Etapa 1: RandomizedSearch
    rs_results, best_model_rs = trainer.tune_randomized_search(X_train, y_train)
    print(f"\n✓ RandomizedSearch: {rs_results['best_score']:.4f}")
    
    # Etapa 2: GridSearch
    gs_results, best_model_gs = trainer.tune_grid_search(
        X_train, y_train, 
        rs_results['best_params']
    )
    print(f"✓ GridSearch: {gs_results['best_score']:.4f}")
    print(f"  Improvement: {gs_results['best_score'] - rs_results['best_score']:+.4f}")
    
    # 7. Evaluar en test
    evaluator = ModelEvaluator(config)
    test_report = evaluator.generate_full_evaluation_report(
        best_model_gs,
        X_test,
        y_test,
        model_name="random_forest_tuned"
    )
    
    print(f"\n✓ Test evaluation:")
    print(f"  ROC AUC: {test_report['roc_auc']:.4f}")
    print(f"  Balanced Accuracy: {test_report['balanced_accuracy']:.4f}")
    print(f"  F1 Score: {test_report['f1_score']:.4f}")
    
    # 8. Guardar modelo
    trainer.save_model(best_model_gs, "random_forest_tuned")
    
    print(f"\n✓ Model saved")
    print(f"✓ Figures saved to: {config.get('output.figures_dir')}")


def example_clustering_workflow():
    """Ejemplo de flujo de clustering completo."""
    print("\n" + "="*60)
    print("CLUSTERING WORKFLOW")
    print("="*60)
    
    # 1. Configuración
    config = Config('configs/exp_km.yaml')
    
    # 2. Cargar datos (solo train para clustering)
    loader = DataLoader(config)
    df = loader.load_data()
    train_df, _ = loader.split_data(df)
    X_train, _ = loader.prepare_xy(train_df)
    
    print(f"\n✓ Data loaded: {len(X_train)} samples")
    
    # 3. Crear pipeline builder
    kmeans_builder = KMeansPipeline(config)
    
    # 4. Crear analizador
    analyzer = KMeansAnalyzer(config, kmeans_builder)
    
    # 5. Evaluar rango de k
    k_range = [2, 3, 4, 5, 6, 7, 8]
    metrics_df = analyzer.evaluate_k_range(X_train, k_range)
    
    print(f"\n✓ K-means evaluated for k in {k_range}")
    print(f"\nMetrics by k:")
    print(metrics_df[['k', 'inertia', 'silhouette', 'calinski_harabasz', 'davies_bouldin']])
    
    # 6. Seleccionar mejor k
    best_k_silhouette = analyzer.select_best_k(method='silhouette')
    best_k_elbow = analyzer.select_best_k(method='elbow')
    
    print(f"\n✓ Best k (silhouette): {best_k_silhouette}")
    print(f"✓ Best k (elbow): {best_k_elbow}")
    
    # Usar silhouette como criterio principal
    best_k = best_k_silhouette
    
    # 7. Analizar clusters
    centroids = analyzer.get_centroids(k=best_k, descale=True)
    print(f"\n✓ Centroids (k={best_k}):")
    print(centroids)
    
    analysis = analyzer.analyze_cluster_characteristics(X_train, k=best_k)
    print(f"\n✓ Cluster sizes:")
    for cluster_id in range(best_k):
        info = analysis[f'cluster_{cluster_id}']
        print(f"  Cluster {cluster_id}: {info['size']} samples ({info['percentage']:.2f}%)")
    
    top_features = analyzer.get_cluster_differences(k=best_k, top_n=5)
    print(f"\n✓ Top differentiating features:")
    print(top_features)
    
    # 8. Visualizaciones
    visualizer = ClusteringVisualizer(config)
    
    # Pipeline PCA para visualización
    pca_pipeline = kmeans_builder.build_pca_pipeline_for_visualization(n_components=2)
    pca_pipeline.fit(X_train)
    
    # Predecir labels
    labels = analyzer.predict_clusters(X_train, k=best_k)
    
    # Generar todas las visualizaciones
    figures = visualizer.generate_full_clustering_report(
        metrics_df=metrics_df,
        X=X_train,
        labels=labels,
        k=best_k,
        feature_names=kmeans_builder.numeric_features,
        pca_pipeline=pca_pipeline
    )
    
    print(f"\n✓ Visualizations generated:")
    for fig_name, fig_path in figures.items():
        if fig_path:
            print(f"  {fig_name}")
    
    # 9. Guardar modelo y resultados
    analyzer.save_model(k=best_k)
    analyzer.save_results()
    
    print(f"\n✓ Model and results saved")


def example_compare_models():
    """Ejemplo de comparación de múltiples modelos."""
    print("\n" + "="*60)
    print("COMPARING MULTIPLE MODELS")
    print("="*60)
    
    config = Config('configs/default.yaml')
    
    # Cargar datos
    loader = DataLoader(config)
    df = loader.load_data()
    train_df, test_df = loader.split_data(df)
    X_train, y_train = loader.prepare_xy(train_df)
    X_test, y_test = loader.prepare_xy(test_df)
    
    # Preprocesamiento
    prep_pipeline = PreprocessingPipeline(config)
    trainer = SupervisedModelTrainer(config, prep_pipeline)
    
    # Entrenar múltiples modelos
    models_results = {}
    
    print("\n🔄 Training multiple models...")
    
    # Baseline
    baseline = trainer.get_baseline_model()
    models_results['baseline'] = trainer.train_with_cv(baseline, X_train, y_train, "baseline")
    
    # RandomForest con diferentes configuraciones
    rf_50 = trainer.get_random_forest_model(n_estimators=50, max_depth=10)
    models_results['rf_50_d10'] = trainer.train_with_cv(rf_50, X_train, y_train, "rf_50_d10")
    
    rf_100 = trainer.get_random_forest_model(n_estimators=100, max_depth=15)
    models_results['rf_100_d15'] = trainer.train_with_cv(rf_100, X_train, y_train, "rf_100_d15")
    
    rf_200 = trainer.get_random_forest_model(n_estimators=200, max_depth=20)
    models_results['rf_200_d20'] = trainer.train_with_cv(rf_200, X_train, y_train, "rf_200_d20")
    
    # Comparar resultados
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS (CV)")
    print("="*60)
    
    import pandas as pd
    
    comparison = []
    for model_name, results in models_results.items():
        row = {
            'model': model_name,
            'roc_auc': results['cv_results']['roc_auc']['mean'],
            'roc_auc_std': results['cv_results']['roc_auc']['std'],
            'balanced_acc': results['cv_results']['balanced_accuracy']['mean'],
            'f1': results['cv_results']['f1']['mean'],
            'time': results['training_time']
        }
        comparison.append(row)
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Mejor modelo
    best_model_name = comparison_df.iloc[0]['model']
    print(f"\n✓ Best model: {best_model_name}")
    print(f"  ROC AUC: {comparison_df.iloc[0]['roc_auc']:.4f} ± {comparison_df.iloc[0]['roc_auc_std']:.4f}")


def example_custom_analysis():
    """Ejemplo de análisis personalizado con los resultados."""
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS")
    print("="*60)
    
    config = Config('configs/default.yaml')
    
    # Cargar datos
    loader = DataLoader(config)
    df = loader.load_data()
    train_df, test_df = loader.split_data(df)
    X_train, y_train = loader.prepare_xy(train_df)
    X_test, y_test = loader.prepare_xy(test_df)
    
    # Entrenar modelo rápido
    prep_pipeline = PreprocessingPipeline(config)
    trainer = SupervisedModelTrainer(config, prep_pipeline)
    
    rf_model = trainer.get_random_forest_model(n_estimators=100)
    results = trainer.train_with_cv(rf_model, X_train, y_train, "custom_rf")
    
    # Obtener el pipeline entrenado
    fitted_pipeline = results['fitted_pipeline']
    
    # Hacer predicciones
    y_pred = fitted_pipeline.predict(X_test)
    y_pred_proba = fitted_pipeline.predict_proba(X_test)
    
    # Análisis personalizado
    print(f"\n📊 Custom Analysis:")
    
    # 1. Distribución de predicciones
    import numpy as np
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"\nPrediction distribution:")
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} ({count/len(y_pred)*100:.2f}%)")
    
    # 2. Confianza de predicciones
    confidences = np.max(y_pred_proba, axis=1)
    print(f"\nPrediction confidence:")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Min: {confidences.min():.4f}")
    print(f"  Max: {confidences.max():.4f}")
    
    # 3. Feature importance (si está disponible)
    if hasattr(fitted_pipeline.named_steps['model'], 'feature_importances_'):
        importances = fitted_pipeline.named_steps['model'].feature_importances_
        
        # Top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\nTop 10 most important features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")
    
    # 4. Errores por segmento
    from sklearn.metrics import classification_report
    print(f"\nDetailed classification report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    example_supervised_workflow()
    
    example_clustering_workflow()
    
    example_compare_models()
    
    
    print("\n" + "="*60)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*60)