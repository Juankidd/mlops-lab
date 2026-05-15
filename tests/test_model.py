import pytest
import pandas as pd
import numpy as np
from src.features.build import build_features
from src.data.ingest import download_data, analyze_data

class TestDataIngestion:
    def test_download_creates_file(self, tmp_path):
        output = tmp_path / "test.csv"
        df = download_data(str(output))
        assert output.exists()
        assert len(df) > 0
        assert 'price' in df.columns

    def test_no_nulls(self, tmp_path):
        df = download_data(str(tmp_path / "test.csv"))
        assert df.isnull().sum().sum() == 0

class TestFeatureEngineering:
    def test_new_features_created(self, tmp_path):
        raw = tmp_path / "raw.csv"
        processed = tmp_path / "features.csv"
        download_data(str(raw))
        df = build_features(str(raw), str(processed))
        assert 'rooms_per_person' in df.columns
        assert 'dist_to_center' in df.columns

    def test_no_infinite_values(self, tmp_path):
        raw = tmp_path / "raw.csv"
        processed = tmp_path / "features.csv"
        download_data(str(raw))
        df = build_features(str(raw), str(processed))
        assert not df.isin([np.inf, -np.inf]).any().any()

# Ejecutar con: pytest tests/ -v