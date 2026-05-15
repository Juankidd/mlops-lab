"""
Feature Engineering — Feature Store Component
Crea y almacena features reutilizables
"""
import pandas as pd
import numpy as np
from pathlib import Path

def build_features(input_path: str, output_path: str):
    """Construye features desde datos raw"""
    df = pd.read_csv(input_path)
    
    # === FEATURE ENGINEERING ===
    
    # Feature 1: Ratio habitaciones/población
    df['rooms_per_person'] = df['AveRooms'] / (df['Population'] / df['AveOccup'] + 1)
    
    # Feature 2: Densidad del bloque
    df['block_density'] = df['Population'] / (df['AveRooms'] + 1)
    
    # Feature 3: Ingresos al cuadrado (relación no lineal)
    df['income_squared'] = df['MedInc'] ** 2
    
    # Feature 4: Distancia al centro (SF: 37.77, -122.41)
    df['dist_to_center'] = np.sqrt(
        (df['Latitude'] - 37.77) ** 2 + (df['Longitude'] - (-122.41)) ** 2
    )
    
    # Feature 5: Zona geográfica (norte/sur de SF)
    df['is_bay_area'] = (df['Latitude'].between(37, 38)).astype(int)
    
    # Guardar features procesadas
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Features creadas: {list(df.columns)}")
    print(f"📊 Shape: {df.shape}")
    return df

if __name__ == "__main__":
    build_features("data/raw/housing.csv", "data/processed/features.csv")