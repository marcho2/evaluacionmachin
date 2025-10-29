from setuptools import setup, find_packages

setup(
    name="mlproject",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "jinja2>=3.1.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "mlproject=mlproject.cli:cli",
        ],
    },
    python_requires=">=3.8",
)