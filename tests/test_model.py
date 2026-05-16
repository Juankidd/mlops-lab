import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_basic():
    assert True


def test_download(tmp_path):
    from src.data.ingest import download_data
    output = str(tmp_path / "test.csv")
    df = download_data(output)
    assert Path(output).exists()
    assert len(df) > 0
    assert 'price' in df.columns


def test_no_nulls(tmp_path):
    from src.data.ingest import download_data
    df = download_data(str(tmp_path / "test.csv"))
    assert df.isnull().sum().sum() == 0


def test_features(tmp_path):
    from src.data.ingest import download_data
    from src.features.build import build_features
    raw = str(tmp_path / "raw.csv")
    processed = str(tmp_path / "features.csv")
    download_data(raw)
    df = build_features(raw, processed)
    assert 'rooms_per_person' in df.columns
    assert 'dist_to_center' in df.columns
    assert not df.isin([float('inf'), float('-inf')]).any().any()