"""
BAN6440 Module 4: K-Means Climate Risk Segmentation for Banking
Dataset: NOAA Global Historical Climatology Network Daily (GHCN-D)
         Registry of Open Data on AWS: https://registry.opendata.aws/noaa-ghcn/
         S3 Bucket: s3://noaa-ghcn-pds/csv/by_year/
Author: Willy Kangogo Kosgei
Date: August 2026

Banks and insurers are required to assess climate-related financial
risk under Basel III and TCFD frameworks. This application clusters
weather stations by climate severity profile, enabling a bank to
segment its mortgage portfolio by geographic climate risk exposure.

SETUP:
    pip install scikit-learn pandas matplotlib numpy seaborn pytest
    aws s3 cp s3://noaa-ghcn-pds/csv/by_year/2023.csv . --no-sign-request
    python kmeans_banking.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings('ignore')

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filepath="2023.csv"):
    """
    Load NOAA GHCN-D weather observation data downloaded from the
    Registry of Open Data on AWS (s3://noaa-ghcn-pds).

    The CSV has no header row. Columns are:
    ID, DATE, ELEMENT, DATA_VALUE, M_FLAG, Q_FLAG, S_FLAG, OBS_TIME

    ELEMENT codes used:
        TMAX = maximum temperature (tenths of degrees C)
        TMIN = minimum temperature (tenths of degrees C)
        PRCP = precipitation (tenths of mm)
        SNOW = snowfall (mm)
    """
    if not os.path.exists(filepath):
        print("=" * 60)
        print("DATASET NOT FOUND")
        print("=" * 60)
        print("Download from AWS Registry of Open Data:")
        print("  aws s3 cp s3://noaa-ghcn-pds/csv/by_year/2023.csv . --no-sign-request")
        print("=" * 60)
        raise FileNotFoundError(f"{filepath} not found. See instructions above.")

    print(f"Loading NOAA GHCN data from {filepath}...")
    print("(This may take a moment for large files)")

    # column names per GHCN-D documentation
    col_names = ['station_id', 'date', 'element', 'value',
                 'm_flag', 'q_flag', 's_flag', 'obs_time']

    # read only the columns we need; filter to key weather elements
    df = pd.read_csv(
        filepath, header=0, low_memory=False,
        dtype=str
    )
    # handle both named and unnamed column formats
    if 'DATA_VALUE' in df.columns:
        df = df.rename(columns={
            'ID': 'station_id', 'DATE': 'date',
            'ELEMENT': 'element', 'DATA_VALUE': 'value'
        })
    elif df.columns[0] != 'station_id':
        df.columns = ['station_id', 'date', 'element', 'value',
                       'm_flag', 'q_flag', 's_flag', 'obs_time']
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[['station_id', 'date', 'element', 'value']]

    # keep only the four elements we need for climate profiling
    elements = ['TMAX', 'TMIN', 'PRCP', 'SNOW']
    df = df[df['element'].isin(elements)]

    print(f"Loaded {len(df):,} observations across {df['station_id'].nunique():,} stations")
    return df


def create_station_features(df):
    """
    Aggregate daily observations into station-level climate features.
    Each row in the output represents one weather station with its
    annual climate profile.

    Features created:
        - avg_max_temp: mean daily maximum temperature (degrees C)
        - avg_min_temp: mean daily minimum temperature (degrees C)
        - temp_range: difference between avg max and avg min (volatility)
        - total_precip: total annual precipitation (mm)
        - snow_days: number of days with recorded snowfall
        - observation_count: total observations (data completeness proxy)
    """
    print("Creating station-level climate features...")

    # pivot elements into separate columns per station per date
    pivot = df.pivot_table(
        index='station_id', columns='element',
        values='value', aggfunc='mean'
    )

    # convert from tenths to actual units
    station_df = pd.DataFrame()
    station_df['station_id'] = pivot.index

    if 'TMAX' in pivot.columns:
        station_df['avg_max_temp'] = (pivot['TMAX'] / 10).values  # tenths C -> C
    if 'TMIN' in pivot.columns:
        station_df['avg_min_temp'] = (pivot['TMIN'] / 10).values
    if 'PRCP' in pivot.columns:
        station_df['total_precip'] = (pivot['PRCP'] / 10).values  # tenths mm -> mm
    if 'SNOW' in pivot.columns:
        station_df['snow_days'] = pivot['SNOW'].values

    # calculate temperature range (climate volatility)
    if 'avg_max_temp' in station_df.columns and 'avg_min_temp' in station_df.columns:
        station_df['temp_range'] = station_df['avg_max_temp'] - station_df['avg_min_temp']

    # count observations per station as data quality indicator
    obs_counts = df.groupby('station_id').size().reset_index(name='observation_count')
    station_df = station_df.merge(obs_counts, on='station_id', how='left')

    # drop stations with too few observations (unreliable data)
    station_df = station_df.dropna()
    station_df = station_df[station_df['observation_count'] >= 100].reset_index(drop=True)

    feature_cols = [c for c in station_df.columns if c != 'station_id']
    print(f"Created features for {len(station_df):,} stations (100+ observations)")
    print(f"Features: {feature_cols}")

    return station_df


def preprocess_data(df):
    """
    Clean and scale the dataset for K-Means clustering.
    K-Means uses Euclidean distance, so features must be on
    the same scale to prevent high-magnitude features from
    dominating the clustering.
    """
    original_len = len(df)
    df_clean = df.dropna()
    dropped = original_len - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values")

    # select numeric columns only (exclude station_id)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'station_id' in numeric_cols:
        numeric_cols.remove('station_id')
    df_numeric = df_clean[numeric_cols]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    print(f"Preprocessed {len(df_numeric):,} records, {len(numeric_cols)} features")
    return scaled_data, df_numeric, scaler, numeric_cols


def find_optimal_k(scaled_data, k_range=range(2, 11)):
    """
    Use the Elbow Method and Silhouette Score to determine the
    optimal number of clusters. Elbow plots inertia (WCSS) against k.
    Silhouette measures cluster separation quality.
    """
    inertias = []
    silhouette_scores = []

    print("\nElbow Method Analysis:")
    print(f"{'k':<5} {'Inertia':<15} {'Silhouette':<12}")
    print("-" * 32)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(scaled_data)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(sil_score)
        print(f"{k:<5} {kmeans.inertia_:<15.2f} {sil_score:<12.4f}")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method: Inertia vs. k', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(k_range), silhouette_scores, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score vs. k', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nElbow plot saved to {OUTPUT_DIR}/elbow_analysis.png")

    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"Recommended k = {optimal_k} (silhouette: {max(silhouette_scores):.4f})")

    return optimal_k, inertias, silhouette_scores


def run_kmeans(scaled_data, n_clusters):
    """Run K-Means with the chosen k and return model, labels, score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(scaled_data)
    sil_score = silhouette_score(scaled_data, labels)

    print(f"\nK-Means Results (k={n_clusters}):")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Inertia (WCSS):  {kmeans.inertia_:.2f}")
    print(f"  Iterations:      {kmeans.n_iter_}")

    return kmeans, labels, sil_score


def profile_clusters(df_original, labels, feature_names):
    """
    Profile each cluster by mean feature values. In the banking
    context, each cluster represents a climate risk zone that the
    bank can map to its mortgage portfolio exposure.
    """
    df_profiled = df_original.copy()
    df_profiled['Cluster'] = labels

    print("\n" + "=" * 70)
    print("CLUSTER PROFILES (Climate Risk Zones)")
    print("=" * 70)

    for cid in sorted(df_profiled['Cluster'].unique()):
        data = df_profiled[df_profiled['Cluster'] == cid]
        count = len(data)
        pct = count / len(df_profiled) * 100
        print(f"\n--- Cluster {cid} ({count:,} stations, {pct:.1f}%) ---")
        for feat in feature_names:
            print(f"  {feat:<25} {data[feat].mean():>12.2f}")

    summary = df_profiled.groupby('Cluster')[feature_names].mean().round(2)
    summary['station_count'] = df_profiled.groupby('Cluster').size()
    summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))
    print(f"\nProfiles saved to {OUTPUT_DIR}/cluster_profiles.csv")

    return df_profiled, summary


def visualize_clusters(df_profiled, feature_names):
    """Create scatter plots and heatmap of cluster results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ('avg_max_temp', 'total_precip'),
        ('avg_min_temp', 'snow_days'),
        ('temp_range', 'observation_count'),
    ]

    for ax, (x, y) in zip(axes, pairs):
        if x in feature_names and y in feature_names:
            ax.scatter(df_profiled[x], df_profiled[y],
                       c=df_profiled['Cluster'], cmap='tab10', alpha=0.5, s=10)
            ax.set_xlabel(x.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel(y.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{x.replace("_"," ").title()} vs {y.replace("_"," ").title()}', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # heatmap
    summary = df_profiled.groupby('Cluster')[feature_names].mean()
    norm = (summary - summary.min()) / (summary.max() - summary.min())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(norm, annot=summary.round(1).values, fmt='', cmap='YlOrRd', ax=ax,
                xticklabels=[f.replace('_', '\n') for f in feature_names],
                yticklabels=[f'Cluster {i}' for i in summary.index])
    ax.set_title('Climate Risk Zone Heatmap (actual means)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {OUTPUT_DIR}/")


def main():
    """
    Main flow:
    1. Load NOAA GHCN data from AWS Open Data Registry
    2. Create station-level climate features
    3. Preprocess and scale
    4. Elbow Method for optimal k
    5. Run K-Means
    6. Profile and visualize climate risk zones
    """
    print("=" * 70)
    print("BAN6440 - K-MEANS CLIMATE RISK SEGMENTATION")
    print("Dataset: NOAA GHCN-D (Registry of Open Data on AWS)")
    print("Use Case: Mortgage Portfolio Climate Risk Assessment")
    print("=" * 70)

    df_raw = load_data()
    station_df = create_station_features(df_raw)
    scaled_data, df_numeric, scaler, features = preprocess_data(station_df)
    optimal_k, inertias, sil_scores = find_optimal_k(scaled_data)
    model, labels, sil_score = run_kmeans(scaled_data, optimal_k)
    df_profiled, summary = profile_clusters(df_numeric, labels, features)
    visualize_clusters(df_profiled, features)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Raw Observations: {len(df_raw):,}")
    print(f"  Stations:         {len(station_df):,}")
    print(f"  Features:         {len(features)}")
    print(f"  Optimal Clusters: {optimal_k}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Output:           {OUTPUT_DIR}/")
    print("=" * 70)

    return model, labels, df_profiled, summary


if __name__ == "__main__":
    main()
