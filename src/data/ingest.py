"""
Ingesta de datos — Data Analysis Component
Descarga el dataset y lo versiona con DVC
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(output_path: str = "data/raw/housing.csv") -> pd.DataFrame:
    """Descarga California Housing dataset"""
    logger.info("Descargando dataset...")
    
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    
    # Añadir columna target
    df['price'] = dataset.target
    
    # Guardar a CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Datos guardados: {len(df)} filas, {len(df.columns)} columnas")
    logger.info(f"Columnas: {list(df.columns)}")
    return df

def analyze_data(df: pd.DataFrame) -> dict:
    """Análisis exploratorio básico"""
    stats = {
        'shape': df.shape,
        'nulls': df.isnull().sum().to_dict(),
        'price_mean': df['price'].mean(),
        'price_std': df['price'].std(),
        'price_min': df['price'].min(),
        'price_max': df['price'].max(),
    }
    logger.info(f"Stats: media={stats['price_mean']:.2f}, std={stats['price_std']:.2f}")
    return stats

if __name__ == "__main__":
    df = download_data()
    stats = analyze_data(df)
    print("\n=== ANÁLISIS DEL DATASET ===")
    print(f"Filas: {stats['shape'][0]}, Columnas: {stats['shape'][1]}")
    print(f"Precio medio: ${stats['price_mean']:.2f}k")
    print(f"Nulls: {sum(stats['nulls'].values())}")