import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings

class ProteinSequenceData:
    """
    Class for handling protein sequence data, supports extraction and generation of one-hot encodings and PLM embeddings
    """
    
    # Standard amino acids
    STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 sequence_column: str = 'sequence',
                 label_column: Optional[str] = None,
                 onehot_prefix: str = 'onehot_',
                 plm_prefix: str = 'plm_'):
        """
        Initialize ProteinSequenceData
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            DataFrame containing protein sequence data
        sequence_column : str
            Column name for sequences
        label_column : str, optional
            Column name for labels/targets
        onehot_prefix : str
            Prefix for one-hot encoding columns
        plm_prefix : str
            Prefix for PLM embedding columns
        """
        self.df = dataframe.copy()
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.onehot_prefix = onehot_prefix
        self.plm_prefix = plm_prefix
        
        # Validate sequence column exists
        if sequence_column not in self.df.columns:
            raise ValueError(f"Sequence column '{sequence_column}' not found in DataFrame")
        
        # Extract sequences
        self.sequences = self.df[sequence_column].tolist()
        
        # Extract labels if specified
        self.labels = None
        if label_column and label_column in self.df.columns:
            self.labels = self.df[label_column].values
            print(f"Found label column: '{label_column}'")
        elif label_column:
            warnings.warn(f"Label column '{label_column}' not found in DataFrame")
        
        # Check and extract existing embeddings
        self._extract_existing_embeddings()
        
        # Generate missing embeddings only if they don't exist
        self._generate_missing_embeddings()
    
    def _extract_existing_embeddings(self):
        """Extract existing one-hot and PLM embedding columns from DataFrame"""
        # Find one-hot encoding columns
        self.onehot_cols = [col for col in self.df.columns if col.startswith(self.onehot_prefix)]
        self.plm_cols = [col for col in self.df.columns if col.startswith(self.plm_prefix)]
        
        if self.onehot_cols:
            self.onehot_embeddings = self.df[self.onehot_cols].values
            self.has_onehot = True
            self.onehot_generated = False
            print(f"Found {len(self.onehot_cols)} one-hot encoding columns")
        else:
            self.onehot_embeddings = None
            self.has_onehot = False
            self.onehot_generated = False
            
        if self.plm_cols:
            self.plm_embeddings = self.df[self.plm_cols].values
            self.has_plm = True
            self.plm_generated = False
            print(f"Found {len(self.plm_cols)} PLM embedding columns")
        else:
            self.plm_embeddings = None
            self.has_plm = False
            self.plm_generated = False
    
    def _generate_missing_embeddings(self):
        """Generate missing embeddings only if they don't already exist"""
        # Generate one-hot encodings only if they don't exist
        if not self.has_onehot:
            print("Generating one-hot encodings...")
            self.onehot_embeddings = self._generate_onehot_embeddings()
            self.has_onehot = True
            self.onehot_generated = True
            
        # Generate PLM embeddings only if they don't exist
        if not self.has_plm:
            print("Generating PLM embeddings...")
            self.plm_embeddings = self._generate_plm_embeddings()
            self.has_plm = True
            self.plm_generated = True
            
        # Add generated embeddings to DataFrame only if they were generated
        self._add_embeddings_to_dataframe()
    
    def _generate_onehot_embeddings(self) -> np.ndarray:
        """
        Generate one-hot encodings for all sequences
        
        Returns:
        --------
        np.ndarray
            One-hot encoding array
        """
        embeddings = []
        for sequence in self.sequences:
            embedding = self._single_sequence_onehot(sequence)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _single_sequence_onehot(self, sequence: str) -> np.ndarray:
        """
        Generate one-hot encoding for a single sequence
        
        Parameters:
        -----------
        sequence : str
            Protein sequence
            
        Returns:
        --------
        np.ndarray
            One-hot encoding vector
        """
        # Filter non-standard amino acids
        filtered_seq = self._filter_sequence(sequence)
        length = len(filtered_seq)
        
        # Calculate amino acid frequencies (normalized counts)
        counts = np.zeros(len(self.STANDARD_AMINO_ACIDS), dtype=float)
        for aa in filtered_seq:
            if aa in self.STANDARD_AMINO_ACIDS:
                idx = self.STANDARD_AMINO_ACIDS.index(aa)
                counts[idx] += 1
        
        # Normalize
        if length > 0:
            embedding = counts / length
        else:
            embedding = counts
            
        return embedding
    
    def _generate_plm_embeddings(self, embedding_dim: int = 1280) -> np.ndarray:
        """
        Generate PLM embeddings (simplified version)
        
        Note: In practice, you should replace this with actual PLM models like ESM, ProtBERT, etc.
        
        Parameters:
        -----------
        embedding_dim : int
            Dimension of PLM embeddings
            
        Returns:
        --------
        np.ndarray
            PLM embedding array
        """
        warnings.warn(
            "Using randomly generated PLM embeddings. In practice, replace with actual PLM models.",
            UserWarning
        )
        
        embeddings = []
        for sequence in self.sequences:
            # Here you should call actual PLM models
            # Using random generation as placeholder
            embedding = np.random.normal(0, 1, embedding_dim)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _add_embeddings_to_dataframe(self):
        """Add generated embeddings to DataFrame only if they were newly generated"""
        # Add one-hot encodings only if they were generated
        if self.onehot_generated and self.onehot_embeddings is not None:
            onehot_cols = [f"{self.onehot_prefix}{aa}" for aa in self.STANDARD_AMINO_ACIDS]
            onehot_df = pd.DataFrame(self.onehot_embeddings, columns=onehot_cols)
            self.df = pd.concat([self.df, onehot_df], axis=1)
            print("Added generated one-hot encodings to DataFrame")
        
        # Add PLM embeddings only if they were generated
        if self.plm_generated and self.plm_embeddings is not None:
            plm_dim = self.plm_embeddings.shape[1]
            plm_cols = [f"{self.plm_prefix}{i}" for i in range(plm_dim)]
            plm_df = pd.DataFrame(self.plm_embeddings, columns=plm_cols)
            self.df = pd.concat([self.df, plm_df], axis=1)
            print("Added generated PLM embeddings to DataFrame")
        
        # If no embeddings were generated, just use the original DataFrame
        if not self.onehot_generated and not self.plm_generated:
            print("Using existing embeddings from input DataFrame")
    
    def concat_dataframe_with_prefix(self, new_df: pd.DataFrame, prefix: str) -> None:
        """
        Concatenate a new DataFrame to the current data with column names prefixed
        
        Parameters:
        -----------
        new_df : pd.DataFrame
            New DataFrame to concatenate
        prefix : str
            Prefix to add to all column names of the new DataFrame
            
        Returns:
        --------
        None
        """
        # Validate that the new DataFrame has the same number of rows
        if len(new_df) != len(self.df):
            raise ValueError(f"Row count mismatch: current DataFrame has {len(self.df)} rows, "
                           f"new DataFrame has {len(new_df)} rows")
        
        # Create a copy to avoid modifying the original
        new_df_prefixed = new_df.copy()
        
        # Add prefix to all column names
        new_df_prefixed.columns = [f"{prefix}{col}" for col in new_df_prefixed.columns]
        
        # Concatenate with the current DataFrame
        self.df = pd.concat([self.df, new_df_prefixed], axis=1)
        
        print(f"Successfully concatenated {new_df_prefixed.shape[1]} columns with prefix '{prefix}'")
    
    def extract_columns_by_prefix(self, prefix: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract columns from DataFrame that match the given prefix
        
        Parameters:
        -----------
        prefix : str
            Prefix to search for in column names
            
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            DataFrame with only the matching columns, and numpy array of the values
        """
        matching_cols = [col for col in self.df.columns if col.startswith(prefix)]
        
        if not matching_cols:
            warnings.warn(f"No columns found with prefix '{prefix}'")
            return pd.DataFrame(), np.array([])
        
        extracted_df = self.df[matching_cols]
        extracted_array = extracted_df.values
        
        print(f"Extracted {len(matching_cols)} columns with prefix '{prefix}'")
        return extracted_df, extracted_array
    
    def list_columns_by_prefix(self, prefix: str) -> List[str]:
        """
        List all column names that match the given prefix
        
        Parameters:
        -----------
        prefix : str
            Prefix to search for in column names
            
        Returns:
        --------
        List[str]
            List of column names matching the prefix
        """
        matching_cols = [col for col in self.df.columns if col.startswith(prefix)]
        return matching_cols
    
    def get_column_prefixes(self) -> Dict[str, List[str]]:
        """
        Get all unique prefixes in the DataFrame column names
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping prefixes to their column names
        """
        prefixes = {}
        for col in self.df.columns:
            # Find the prefix (everything before the last underscore)
            if '_' in col:
                prefix = col.split('_')[0] + '_'
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(col)
            else:
                # For columns without underscores, use the whole name as prefix
                if col not in prefixes:
                    prefixes[col] = []
                prefixes[col].append(col)
        
        return prefixes
    
    @staticmethod
    def _filter_sequence(sequence: str) -> str:
        """Filter non-standard amino acids, replace with 'A'"""
        return ''.join(c if c in ProteinSequenceData.STANDARD_AMINO_ACIDS else 'A' 
                      for c in sequence.upper())
    
    @staticmethod
    def validate_sequence(sequence: str) -> bool:
        """Validate if sequence contains valid amino acids"""
        return all(aa in ProteinSequenceData.STANDARD_AMINO_ACIDS 
                  for aa in sequence.upper())
    
    def get_data(self) -> pd.DataFrame:
        """Get processed complete data"""
        return self.df
    
    def get_sequences(self) -> List[str]:
        """Get list of sequences"""
        return self.sequences
    
    def get_labels(self) -> Optional[np.ndarray]:
        """Get labels if available"""
        return self.labels
    
    def get_onehot_embeddings(self) -> Optional[np.ndarray]:
        """Get one-hot encodings"""
        return self.onehot_embeddings
    
    def get_plm_embeddings(self) -> Optional[np.ndarray]:
        """Get PLM embeddings"""
        return self.plm_embeddings
    
    def to_csv(self, filepath: str, **kwargs):
        """
        Export data to CSV file
        
        Parameters:
        -----------
        filepath : str
            Output file path
        **kwargs
            Additional arguments passed to pandas.DataFrame.to_csv
        """
        self.df.to_csv(filepath, **kwargs)
        print(f"Data saved to: {filepath}")
    
    def info(self):
        """Display data information"""
        print(f"Number of sequences: {len(self.sequences)}")
        if self.labels is not None:
            print(f"Labels available: Yes (from column '{self.label_column}')")
            print(f"Label type: {type(self.labels)}")
            if hasattr(self.labels, 'shape'):
                print(f"Label shape: {self.labels.shape}")
        else:
            print("Labels available: No")
        print(f"Has one-hot encodings: {self.has_onehot}")
        if self.has_onehot and self.onehot_embeddings is not None:
            print(f"One-hot encoding dimensions: {self.onehot_embeddings.shape}")
            print(f"One-hot encodings source: {'Generated' if self.onehot_generated else 'From input DataFrame'}")
        print(f"Has PLM embeddings: {self.has_plm}")
        if self.has_plm and self.plm_embeddings is not None:
            print(f"PLM embedding dimensions: {self.plm_embeddings.shape}")
            print(f"PLM embeddings source: {'Generated' if self.plm_generated else 'From input DataFrame'}")
        print(f"DataFrame shape: {self.df.shape}")



import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAAnalyzer:
    def __init__(self, n_components=10, labels=None):
        self.n_components = n_components
        self.labels = labels
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.model_fitted = False
        self.cumulative_variance = None
        self.pca_cutoff_dim = None

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.model_fitted = True
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.pca_cutoff_dim = None
        return self

    def transform(self, X):
        if not self.model_fitted:
            raise RuntimeError("Fit model before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save_model(self, filepath):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before saving it")
        joblib.dump((self.scaler, self.pca), filepath)

    def load_model(self, filepath):
        self.scaler, self.pca = joblib.load(filepath)
        self.model_fitted = True
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

    def tune_components(self, X, component_list):
        X_scaled = self.scaler.fit_transform(X)
        max_var = 0
        best_n = component_list[0]
        for n in component_list:
            pca = PCA(n_components=n)
            pca.fit(X_scaled)
            cum_var = np.sum(pca.explained_variance_ratio_)
            if cum_var >= 0.95:
                best_n = n
                break
            if cum_var > max_var:
                max_var = cum_var
                best_n = n
        self.n_components = best_n
        self.pca = PCA(n_components=best_n)
        self.model_fitted = False
        return best_n

    def plot_variance(self):
        if not self.model_fitted or self.cumulative_variance is None:
            raise RuntimeError("Fit model before plotting variance")
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        if self.pca_cutoff_dim is not None:
            plt.axvline(x=self.pca_cutoff_dim - 1, color='b', linestyle='--', label=f'{self.pca_cutoff_dim} Components')
        plt.legend()
        plt.show()

    def plot_2d_projection(self, X_transformed, labels=None):
        if X_transformed.shape[1] < 2:
            raise ValueError("Need at least 2 components for 2D plot")
        plt.figure(figsize=(10, 8))
        scatter_labels = labels if labels is not None else self.labels
        if scatter_labels is not None:
            scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=scatter_labels, cmap='viridis', s=1, alpha=0.2)
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=1, alpha=0.2, color='green')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('PCA 2D Projection')
        plt.grid(True)
        plt.show()


import joblib
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid


class UMAPAnalyzer:
    def __init__(self, n_components=2, labels=None):
        self.n_components = n_components
        self.labels = labels
        self.scaler = StandardScaler()
        self.reducer = None
        self.model_fitted = False
        self.best_params = None

    def fit(self, X, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        # If n_components is provided in kwargs, use that and remove it from kwargs
        self.n_components = kwargs.pop('n_components', self.n_components)
        kwargs.pop('random_state', None)
        self.reducer = umap.UMAP(n_components=self.n_components, random_state=42, **kwargs)
        self.embedding_ = self.reducer.fit_transform(X_scaled)
        self.model_fitted = True
        return self.embedding_

    def transform(self, X):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before transform")
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)

    def save_model(self, filepath):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before saving")
        joblib.dump((self.scaler, self.reducer), filepath)

    def load_model(self, filepath):
        self.scaler, self.reducer = joblib.load(filepath)
        self.model_fitted = True

    
    def hyperparameter_grid_search(self, X, param_grid, **base_params):
        """
        param_grid: dict
            Dictionary of parameters names mapped to lists of values to try.
            Can include 'n_components', 'n_neighbors', 'min_dist', etc.
    
        Example:
          param_grid = {
            'n_components': range(2, 11),
            'n_neighbors': [5, 15, 30],
            'min_dist': [0.1, 0.5],
            'metric': ['euclidean', 'manhattan']
          }
        """
        X_scaled = self.scaler.fit_transform(X)
        best_score = -np.inf
        best_params = None
        best_embedding = None
        scores = {}
    
        for params in ParameterGrid(param_grid):
            current_params = base_params.copy()
            current_params.update(params)
            current_params["random_state"] = 42  # ensure reproducibility
    
            reducer = umap.UMAP(**current_params)
            embedding = reducer.fit_transform(X_scaled)
    
            if self.labels is not None:
                score = silhouette_score(embedding, self.labels)
            else:
                score = np.var(embedding)
    
            scores[tuple(params.items())] = score
    
            if score > best_score:
                best_score = score
                best_params = current_params
                best_embedding = embedding
                self.reducer = reducer
    
        self.model_fitted = True
        self.embedding_ = best_embedding
        self.best_params = best_params
    
        print(f"Best parameters: {best_params}, Best score: {best_score:.4f}")
    
        # Optional: add code to plot scores here based on `scores`
    
        return best_embedding, best_params, scores


    def plot_embedding(self, labels=None):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before plotting")
        plt.figure(figsize=(10,8))
        plot_labels = labels if labels is not None else self.labels
        if plot_labels is not None:
            scatter = plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1], c=plot_labels, cmap='Spectral', s=1, alpha=0.2)
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1], s=1, alpha=0.2, color='blue')
        plt.title("UMAP Projection")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)
        plt.show()


import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class KMeansAnalyzer:
    def __init__(self, n_clusters=2, labels=None):
        self.n_clusters = n_clusters
        self.labels = labels
        self.scaler = StandardScaler()
        self.kmeans = None
        self.model_fitted = False

    def fit(self, X, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, **kwargs)
        self.kmeans.fit(X_scaled)
        self.model_fitted = True
        return self.kmeans.labels_

    def predict(self, X):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before predicting")
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def save_model(self, filepath):
        if not self.model_fitted:
            raise RuntimeError("Fit the model before saving")
        joblib.dump((self.scaler, self.kmeans), filepath)

    def load_model(self, filepath):
        self.scaler, self.kmeans = joblib.load(filepath)
        self.model_fitted = True

    def find_optimal_clusters(self, X, max_k=15):
        X_scaled = self.scaler.fit_transform(X)
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        best_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_idx]
        print(f"Best number of clusters by Silhouette Score: {best_k}")

        # Plot inertia and silhouette score
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(k_range, inertias, 'bo-', label='Inertia')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Sum of Squared Distances)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

        return best_k, k_range, inertias, silhouette_scores

    def plot_clusters(self, X, labels=None):
        if labels is None:
            if not self.model_fitted:
                raise RuntimeError("Fit the model or provide labels for plotting.")
            labels = self.kmeans.labels_
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=1, alpha=0.2)
        plt.title('KMeans Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        plt.grid(True)
        plt.show()
