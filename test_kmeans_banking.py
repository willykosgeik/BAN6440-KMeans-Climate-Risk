"""
BAN6440 Module 4: Unit Tests for K-Means Climate Risk Segmentation
Framework: pytest
Author: Willy Kangogo Kosgei
Date: August 2026

Run: python -m pytest test_kmeans_banking.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from kmeans_banking import (
    create_station_features, preprocess_data,
    find_optimal_k, run_kmeans, profile_clusters
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def mock_ghcn_data():
    """Create mock NOAA GHCN observation data matching the real format."""
    rows = []
    for station in ['USW00094728', 'USW00023174', 'USW00013874']:
        for day in range(1, 201):
            date = f"2023{day:04d}"
            rows.append({'station_id': station, 'date': date, 'element': 'TMAX', 'value': np.random.normal(250, 80)})
            rows.append({'station_id': station, 'date': date, 'element': 'TMIN', 'value': np.random.normal(150, 80)})
            rows.append({'station_id': station, 'date': date, 'element': 'PRCP', 'value': np.random.exponential(30)})
            rows.append({'station_id': station, 'date': date, 'element': 'SNOW', 'value': np.random.choice([0, 0, 0, 50, 100])})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_station_features():
    """Pre-built station-level features for testing."""
    return pd.DataFrame({
        'avg_max_temp': [30.5, 15.2, -5.0, 22.0, 35.1, 10.0, 28.3, 5.5],
        'avg_min_temp': [18.0, 5.0, -15.0, 12.0, 22.0, 0.0, 16.0, -5.0],
        'total_precip': [800, 1200, 500, 950, 200, 1500, 700, 900],
        'snow_days': [0, 20, 80, 5, 0, 40, 2, 60],
        'temp_range': [12.5, 10.2, 10.0, 10.0, 13.1, 10.0, 12.3, 10.5],
        'observation_count': [500, 600, 450, 550, 400, 650, 520, 480],
    })


@pytest.fixture
def synthetic_blob_data():
    """Well-separated synthetic clusters for correctness testing."""
    X, y = make_blobs(n_samples=300, n_features=4, centers=3,
                      cluster_std=0.5, random_state=42)
    return X, y


@pytest.fixture
def scaled_sample(sample_station_features):
    scaler = StandardScaler()
    return scaler.fit_transform(sample_station_features)


# ============================================================
# TEST: Feature Creation
# ============================================================

class TestFeatureCreation:
    def test_creates_station_level_data(self, mock_ghcn_data):
        result = create_station_features(mock_ghcn_data)
        assert len(result) == 3  # 3 stations

    def test_has_expected_columns(self, mock_ghcn_data):
        result = create_station_features(mock_ghcn_data)
        for col in ['avg_max_temp', 'avg_min_temp', 'total_precip', 'snow_days']:
            assert col in result.columns

    def test_temp_converted_from_tenths(self, mock_ghcn_data):
        """GHCN stores temps in tenths of C. Verify conversion."""
        result = create_station_features(mock_ghcn_data)
        # avg max temp should be around 25C (250 tenths / 10), not 250
        assert result['avg_max_temp'].mean() < 50

    def test_filters_low_observation_stations(self, mock_ghcn_data):
        """Stations with <100 observations should be excluded."""
        # add a station with only 10 observations
        sparse = pd.DataFrame({
            'station_id': ['SPARSE001'] * 40,
            'date': ['20230101'] * 40,
            'element': ['TMAX', 'TMIN', 'PRCP', 'SNOW'] * 10,
            'value': [200, 100, 50, 0] * 10,
        })
        combined = pd.concat([mock_ghcn_data, sparse], ignore_index=True)
        result = create_station_features(combined)
        assert 'SPARSE001' not in result['station_id'].values


# ============================================================
# TEST: Preprocessing
# ============================================================

class TestPreprocessing:
    def test_returns_four_elements(self, sample_station_features):
        result = preprocess_data(sample_station_features)
        assert len(result) == 4

    def test_scaled_zero_mean(self, sample_station_features):
        scaled, _, _, _ = preprocess_data(sample_station_features)
        for m in np.mean(scaled, axis=0):
            assert abs(m) < 1e-10

    def test_preserves_rows(self, sample_station_features):
        scaled, _, _, _ = preprocess_data(sample_station_features)
        assert scaled.shape[0] == len(sample_station_features)

    def test_handles_missing_values(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        scaled, _, _, _ = preprocess_data(df)
        assert scaled.shape[0] == 1

    def test_excludes_non_numeric(self):
        df = pd.DataFrame({'temp': [20, 30], 'name': ['A', 'B'], 'precip': [100, 200]})
        _, _, _, feat = preprocess_data(df)
        assert 'name' not in feat


# ============================================================
# TEST: Clustering Correctness (Synthetic Data)
# ============================================================

class TestClusteringCorrectness:
    def test_recovers_known_clusters(self, synthetic_blob_data):
        X, _ = synthetic_blob_data
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        assert sil > 0.5, f"Silhouette too low: {sil}"

    def test_correct_cluster_count(self, synthetic_blob_data):
        X, _ = synthetic_blob_data
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
        assert len(np.unique(labels)) == 3

    def test_all_points_assigned(self, synthetic_blob_data):
        X, _ = synthetic_blob_data
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
        assert len(labels) == len(X)

    def test_no_empty_clusters(self, synthetic_blob_data):
        X, _ = synthetic_blob_data
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
        for cid in range(3):
            assert np.sum(labels == cid) > 10

    def test_inertia_decreases(self, synthetic_blob_data):
        X, _ = synthetic_blob_data
        prev = float('inf')
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            assert km.inertia_ < prev
            prev = km.inertia_


# ============================================================
# TEST: Elbow Method
# ============================================================

class TestElbowMethod:
    def test_returns_valid_k(self, scaled_sample):
        k, _, _ = find_optimal_k(scaled_sample, k_range=range(2, 6))
        assert 2 <= k <= 5

    def test_correct_list_lengths(self, scaled_sample):
        _, inertias, sils = find_optimal_k(scaled_sample, k_range=range(2, 6))
        assert len(inertias) == 4 and len(sils) == 4

    def test_silhouette_bounded(self, scaled_sample):
        _, _, sils = find_optimal_k(scaled_sample, k_range=range(2, 6))
        for s in sils:
            assert -1 <= s <= 1


# ============================================================
# TEST: Run K-Means
# ============================================================

class TestRunKMeans:
    def test_returns_model_labels_score(self, scaled_sample):
        model, labels, score = run_kmeans(scaled_sample, 3)
        assert model is not None and len(labels) == 8 and score > 0

    def test_correct_unique_labels(self, scaled_sample):
        _, labels, _ = run_kmeans(scaled_sample, 3)
        assert len(np.unique(labels)) == 3


# ============================================================
# TEST: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_feature(self):
        X = np.array([[1], [2], [100], [101], [200], [201]])
        assert len(np.unique(KMeans(3, random_state=42, n_init=10).fit_predict(X))) == 3

    def test_two_points(self):
        X = np.array([[0, 0], [10, 10]])
        l = KMeans(2, random_state=42, n_init=10).fit_predict(X)
        assert l[0] != l[1]

    def test_identical_points(self):
        X = np.array([[1, 1]] * 3 + [[100, 100]] * 2)
        l = KMeans(2, random_state=42, n_init=10).fit_predict(X)
        assert l[0] == l[1] == l[2] and l[3] == l[4]

    def test_high_dimensional(self):
        X = np.random.RandomState(42).randn(100, 20)
        assert len(KMeans(3, random_state=42, n_init=10).fit_predict(X)) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
