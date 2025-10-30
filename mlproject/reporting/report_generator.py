"""
Generador de reportes en HTML y Markdown.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Clase para generar reportes en HTML y Markdown."""
    
    def __init__(self, config):
        """
        Inicializa el generador de reportes.
        
        Args:
            config: Objeto de configuración
        """
        self.config = config
        self.reports_dir = Path(config.get('output.reports_dir', 'outputs/reports'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, 
                                 data_summary: Dict,
                                 supervised_results: Dict = None,
                                 clustering_results: Dict = None,
                                 save: bool = True) -> str:
        """
        Genera un reporte completo en Markdown.
        
        Args:
            data_summary: Resumen del dataset
            supervised_results: Resultados de modelos supervisados
            clustering_results: Resultados de clustering
            save: Si guardar el archivo
            
        Returns:
            String con el contenido del reporte
        """
        md = []
        
        # Header
        md.append(f"# Machine Learning Pipeline Report")
        md.append(f"\n**Project:** {self.config.get('project_name', 'ML Project')}")
        md.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\n**Random Seed:** {self.config.get('random_seed', 42)}")
        md.append("\n---\n")
        
        # Data Summary
        if data_summary:
            md.append("## 1. Data Summary\n")
            
            if 'shape' in data_summary:
                md.append(f"- **Samples:** {data_summary['shape']['rows']:,}")
                md.append(f"- **Features:** {data_summary['shape']['columns']}")
            
            if 'target' in data_summary:
                md.append(f"\n### Target Variable: `{data_summary['target']['name']}`\n")
                md.append(f"- **Type:** {data_summary['target']['type']}")
                md.append(f"- **Unique Values:** {data_summary['target']['unique_values']}")
                md.append(f"\n**Distribution:**\n")
                
                for value, count in data_summary['target']['distribution'].items():
                    pct = count / data_summary['shape']['rows'] * 100
                    md.append(f"- `{value}`: {count:,} ({pct:.2f}%)")
            
            if 'missing_values' in data_summary:
                missing_cols = {k: v for k, v in data_summary['missing_values'].items() if v > 0}
                if missing_cols:
                    md.append(f"\n### Missing Values\n")
                    md.append("| Column | Missing | Percentage |")
                    md.append("|--------|---------|------------|")
                    for col, count in list(missing_cols.items())[:10]:
                        pct = data_summary['missing_percentage'][col]
                        md.append(f"| {col} | {count} | {pct:.2f}% |")
                else:
                    md.append(f"\n✓ No missing values found")
        
        # Supervised Results
        if supervised_results:
            md.append("\n---\n")
            md.append("## 2. Supervised Learning Results\n")
            
            for model_name, results in supervised_results.items():
                md.append(f"\n### {model_name.replace('_', ' ').title()}\n")
                
                # CV Results
                if 'cv_results' in results:
                    md.append("**Cross-Validation Metrics:**\n")
                    md.append("| Metric | Mean | Std |")
                    md.append("|--------|------|-----|")
                    for metric, values in results['cv_results'].items():
                        md.append(f"| {metric} | {values['mean']:.4f} | {values['std']:.4f} |")
                
                # Test Results
                if 'roc_auc' in results:
                    md.append(f"\n**Test Set Metrics:**\n")
                    md.append(f"- **ROC AUC:** {results['roc_auc']:.4f}")
                    if 'balanced_accuracy' in results:
                        md.append(f"- **Balanced Accuracy:** {results['balanced_accuracy']:.4f}")
                    if 'f1_score' in results:
                        md.append(f"- **F1 Score:** {results['f1_score']:.4f}")
                
                # Tuning info
                if 'best_params' in results:
                    md.append(f"\n**Best Hyperparameters:**")
                    for param, value in results['best_params'].items():
                        clean_param = param.replace('model__', '')
                        md.append(f"- `{clean_param}`: {value}")
                
                # Figures
                if 'figures' in results:
                    md.append(f"\n**Generated Figures:**")
                    for fig_name, fig_path in results['figures'].items():
                        if fig_path:
                            md.append(f"- {fig_name}: `{fig_path}`")
        
        # Clustering Results
        if clustering_results:
            md.append("\n---\n")
            md.append("## 3. Resultados Clustering  (K-means)\n")
            
            if 'best_k' in clustering_results:
                md.append(f"\n** k seleccionado:** {clustering_results['best_k']}")
                md.append(f"**Selection method:** {clustering_results.get('method', 'unknown')}")
            
            # Metrics by k
            if 'metrics_by_k' in clustering_results:
                md.append(f"\n### Metricas por k\n")
                md.append("| k | Inertia | Silhouette | Calinski-Harabasz | Davies-Bouldin |")
                md.append("|---|---------|------------|-------------------|----------------|")
                
                df = clustering_results['metrics_by_k']
                for _, row in df.iterrows():
                    md.append(f"| {int(row['k'])} | {row['inertia']:.2f} | "
                             f"{row['silhouette']:.4f} | {row['calinski_harabasz']:.2f} | "
                             f"{row['davies_bouldin']:.4f} |")
            
            # Cluster sizes
            if 'cluster_sizes' in clustering_results:
                md.append(f"\n### Tamaño Cluster \n")
                for cluster_id, size in clustering_results['cluster_sizes'].items():
                    md.append(f"- **{cluster_id}:** {size} samples")
            
            # Centroids
            if 'centroids' in clustering_results:
                md.append(f"\n### Centroids\n")
                md.append("See `outputs/kmeans_centroids_*.csv` for detailed centroid values.")
            
            # Figures
            if 'figures' in clustering_results:
                md.append(f"\n**Generated Figures:**")
                for fig_name, fig_path in clustering_results['figures'].items():
                    if fig_path:
                        md.append(f"- {fig_name}: `{fig_path}`")
        
        # Footer
        md.append("\n---\n")
        md.append(f"*Report generated by MLProject*")
        
        # Join all lines
        report_content = "\n".join(md)
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.reports_dir / f"report_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Markdown report saved to: {filename}")
        
        return report_content
    
    def generate_html_report(self,
                            data_summary: Dict,
                            supervised_results: Dict = None,
                            clustering_results: Dict = None,
                            save: bool = True) -> str:
        """
        Genera un reporte completo en HTML.
        
        Args:
            data_summary: Resumen del dataset
            supervised_results: Resultados de modelos supervisados
            clustering_results: Resultados de clustering
            save: Si guardar el archivo
            
        Returns:
            String con el contenido HTML
        """
        html = []
        
        # HTML Header
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("  <title>ML Pipeline Report</title>")
        html.append("  <style>")
        html.append(self._get_css_styles())
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Title
        html.append("  <div class='container'>")
        html.append("    <h1>Machine Learning Pipeline Report</h1>")
        html.append(f"    <p class='meta'>Project: {self.config.get('project_name', 'ML Project')}</p>")
        html.append(f"    <p class='meta'>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"    <p class='meta'>Random Seed: {self.config.get('random_seed', 42)}</p>")
        
        # Data Summary
        if data_summary:
            html.append("    <h2>1. Data Summary</h2>")
            
            if 'shape' in data_summary:
                html.append("    <div class='card'>")
                html.append(f"      <p><strong>Samples:</strong> {data_summary['shape']['rows']:,}</p>")
                html.append(f"      <p><strong>Features:</strong> {data_summary['shape']['columns']}</p>")
                html.append("    </div>")
            
            if 'target' in data_summary:
                html.append(f"    <h3>Target Variable: {data_summary['target']['name']}</h3>")
                html.append("    <div class='card'>")
                html.append(f"      <p><strong>Type:</strong> {data_summary['target']['type']}</p>")
                html.append(f"      <p><strong>Unique Values:</strong> {data_summary['target']['unique_values']}</p>")
                html.append("      <p><strong>Distribution:</strong></p>")
                html.append("      <ul>")
                for value, count in data_summary['target']['distribution'].items():
                    pct = count / data_summary['shape']['rows'] * 100
                    html.append(f"        <li>{value}: {count:,} ({pct:.2f}%)</li>")
                html.append("      </ul>")
                html.append("    </div>")
        
        # Supervised Results
        if supervised_results:
            html.append("    <h2>2. Supervised Learning Results</h2>")
            
            for model_name, results in supervised_results.items():
                html.append(f"    <h3>{model_name.replace('_', ' ').title()}</h3>")
                html.append("    <div class='card'>")
                
                # Metrics table
                if 'cv_results' in results:
                    html.append("      <h4>Cross-Validation Metrics</h4>")
                    html.append("      <table>")
                    html.append("        <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>")
                    for metric, values in results['cv_results'].items():
                        html.append(f"        <tr><td>{metric}</td><td>{values['mean']:.4f}</td><td>{values['std']:.4f}</td></tr>")
                    html.append("      </table>")
                
                if 'roc_auc' in results:
                    html.append("      <h4>Test Set Metrics</h4>")
                    html.append(f"      <p><strong>ROC AUC:</strong> {results['roc_auc']:.4f}</p>")
                    if 'balanced_accuracy' in results:
                        html.append(f"      <p><strong>Balanced Accuracy:</strong> {results['balanced_accuracy']:.4f}</p>")
                
                html.append("    </div>")
        
        # Clustering Results
        if clustering_results:
            html.append("    <h2>3. Clustering Results</h2>")
            html.append("    <div class='card'>")
            
            if 'best_k' in clustering_results:
                html.append(f"      <p><strong>Best k:</strong> {clustering_results['best_k']}</p>")
            
            html.append("    </div>")
        
        # Footer
        html.append("    <hr>")
        html.append("    <p class='footer'>Report generated by MLProject</p>")
        html.append("  </div>")
        html.append("</body>")
        html.append("</html>")
        
        # Join
        report_content = "\n".join(html)
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.reports_dir / f"report_{timestamp}.html"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"HTML report saved to: {filename}")
        
        return report_content
    
    def _get_css_styles(self) -> str:
        """Retorna estilos CSS para el reporte HTML."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        h3 {
            color: #7f8c8d;
        }
        .meta {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0;
        }
        .card {
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            color: #95a5a6;
            font-style: italic;
            margin-top: 30px;
        }
        """
    
    def export_results_to_csv(self, results: Dict, filename: str):
        """
        Exporta resultados a CSV.
        
        Args:
            results: Diccionario con resultados
            filename: Nombre del archivo
        """
        import pandas as pd
        
        filepath = self.reports_dir / filename
        
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        elif isinstance(results, dict):
            df = pd.DataFrame([results])
            df.to_csv(filepath, index=False)
        
        logger.info(f"Results exported to CSV: {filepath}")
        return filepath