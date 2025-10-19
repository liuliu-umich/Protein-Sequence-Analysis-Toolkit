# Protein Sequence Analysis Toolkit

A comprehensive Python library for processing, analyzing, and visualizing protein sequence data with built-in support for feature extraction, dimensionality reduction, and clustering.

## Overview

This toolkit provides a complete workflow for protein sequence analysis, from data preprocessing to advanced machine learning techniques. It's designed to handle protein sequence data efficiently with automatic feature extraction, visualization, and clustering capabilities.

## Features

### 🧬 Protein Sequence Data Management (`ProteinSequenceData`)
- **Automatic Feature Detection**: Smart detection of existing one-hot encodings and PLM embeddings
- **Intelligent Feature Generation**: Generates missing features only when needed
- **Flexible Data Integration**: Concatenate new DataFrames with custom prefixes
- **Column Management**: Extract, list, and manage columns by prefix
- **Data Validation**: Sequence validation and filtering for non-standard amino acids
- **Export Capabilities**: Save processed data to CSV format

### 📊 Dimensionality Reduction
#### PCA Analyzer (`PCAAnalyzer`)
- Automated principal component analysis
- Variance explanation visualization
- Optimal component selection
- 2D projection plotting
- Model persistence

#### UMAP Analyzer (`UMAPAnalyzer`)
- Non-linear dimensionality reduction
- Hyperparameter grid search with silhouette scoring
- Customizable projection parameters
- Interactive visualization
- Model saving/loading

### 🎯 Clustering Analysis (`KMeansAnalyzer`)
- K-means clustering implementation
- Optimal cluster number detection (elbow method + silhouette scores)
- Cluster visualization
- Performance metrics

## Installation


Quick Start
Basic Protein Data Processing
python

import pandas as pd
from protein_sequence_toolkit import ProteinSequenceData

# Load your protein data
sample_data = pd.DataFrame({
    'protein_id': ['prot1', 'prot2', 'prot3'],
    'sequence': ['ACDEFGHIKLMNPQRSTVWY', 'MKTVRQERLKSIVRILERSK', 'GSDEDAFRLMNPQSTVWYCK'],
    'label': [1, 0, 1]
})

# Initialize processor
protein_data = ProteinSequenceData(
    dataframe=sample_data,
    sequence_column='sequence',
    label_column='label'
)

# Get processed data
processed_df = protein_data.get_data()
protein_data.info()

Adding Custom Features
python

# Add new features with prefix
new_features = pd.DataFrame({
    'structural_feature1': [0.1, 0.2, 0.3],
    'structural_feature2': [0.4, 0.5, 0.6]
})

protein_data.concat_dataframe_with_prefix(new_features, 'structure_')

Dimensionality Reduction with PCA
python

from protein_sequence_toolkit import PCAAnalyzer

# Extract features for PCA
features_df, features_array = protein_data.extract_columns_by_prefix('onehot_')

# Perform PCA
pca_analyzer = PCAAnalyzer(n_components=10, labels=protein_data.get_labels())
pca_result = pca_analyzer.fit_transform(features_array)

# Visualize results
pca_analyzer.plot_variance()
pca_analyzer.plot_2d_projection(pca_result)

Non-linear Visualization with UMAP
python

from protein_sequence_toolkit import UMAPAnalyzer

# UMAP analysis
umap_analyzer = UMAPAnalyzer(n_components=2, labels=protein_data.get_labels())
umap_embedding = umap_analyzer.fit(features_array)

# Plot results
umap_analyzer.plot_embedding()

Clustering Analysis
python

from protein_sequence_toolkit import KMeansAnalyzer

# Find optimal clusters and perform clustering
kmeans_analyzer = KMeansAnalyzer()
optimal_k, k_range, inertias, scores = kmeans_analyzer.find_optimal_clusters(features_array)

# Fit with optimal clusters
kmeans_analyzer.n_clusters = optimal_k
cluster_labels = kmeans_analyzer.fit(features_array)

# Visualize clusters
kmeans_analyzer.plot_clusters(features_array[:, :2])  # Use first two features for visualization

Advanced Usage
Hyperparameter Tuning with UMAP
python

# Grid search for optimal UMAP parameters
param_grid = {
    'n_components': range(2, 6),
    'n_neighbors': [5, 15, 30],
    'min_dist': [0.1, 0.5],
    'metric': ['euclidean', 'manhattan']
}

best_embedding, best_params, scores = umap_analyzer.hyperparameter_grid_search(
    features_array, 
    param_grid
)

Feature Extraction and Management
python

# Extract specific feature types
onehot_df, onehot_array = protein_data.extract_columns_by_prefix('onehot_')
plm_df, plm_array = protein_data.extract_columns_by_prefix('plm_')
structure_df, structure_array = protein_data.extract_columns_by_prefix('structure_')

# List available features
available_prefixes = protein_data.get_column_prefixes()
onehot_columns = protein_data.list_columns_by_prefix('onehot_')

Model Persistence
python

# Save models for later use
pca_analyzer.save_model('pca_model.joblib')
umap_analyzer.save_model('umap_model.joblib')
kmeans_analyzer.save_model('kmeans_model.joblib')

# Load saved models
new_pca = PCAAnalyzer()
new_pca.load_model('pca_model.joblib')

Complete Code

The complete implementation is available in the protein_sequence_toolkit.py file. Key components include:
ProteinSequenceData Class

    Smart Feature Handling: Avoids duplicate feature generation

    Prefix-based Organization: Clean column management

    Sequence Validation: Ensures data quality

    Flexible Input: Works with pre-computed features or generates them automatically

PCAAnalyzer Class

    Automated Variance Analysis: Helps select optimal components

    Visualization Tools: Built-in plotting capabilities

    Scalable: Handles large feature sets efficiently

UMAPAnalyzer Class

    Non-linear Patterns: Captures complex relationships

    Parameter Optimization: Automated hyperparameter tuning

    Quality Metrics: Uses silhouette scores for evaluation

KMeansAnalyzer Class

    Cluster Optimization: Automatic determination of optimal cluster count

    Multiple Metrics: Uses both inertia and silhouette scores

    Visual Diagnostics: Elbow plots and cluster visualization

Use Cases

    Protein Function Prediction: Feature extraction for ML models

    Sequence Similarity Analysis: Dimensionality reduction for visualization

    Protein Family Classification: Clustering and pattern discovery

    Data Quality Assessment: Outlier detection in sequence space

    Feature Engineering: Creating informative representations for downstream tasks

Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for new features and bug fixes.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Citation

If you use this toolkit in your research, please cite:
bibtex

@software{protein_sequence_toolkit,
  title = {Protein Sequence Analysis Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/protein-sequence-toolkit}
}

Support

For questions and support, please open an issue on GitHub or contact the maintainers.
text


You can copy this entire content and save it as `README.md` in your GitHub repository. This file includes all the documentation, usage examples, and structure needed for a comprehensive project README.



