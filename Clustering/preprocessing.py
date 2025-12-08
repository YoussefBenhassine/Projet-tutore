"""
Preprocessing module for ESG dataset clustering pipeline.

This module handles:
- Data loading and cleaning
- Missing value imputation
- Feature scaling
- PCA for visualization
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handles all data preprocessing steps for the ESG clustering pipeline."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_path: Path to the CSV dataset file
        """
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.feature_names: Optional[list] = None
        self.original_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the ESG dataset and exclude Company_ID and Sector columns.
        
        Returns:
            DataFrame with only numerical ESG features
        """
        try:
            df = pd.read_csv(self.dataset_path)
            self.original_data = df.copy()
            
            # Exclude Company_ID and Sector
            if 'Company_ID' in df.columns:
                df = df.drop(columns=['Company_ID'])
            if 'Sector' in df.columns:
                df = df.drop(columns=['Sector'])
            
            self.feature_names = df.columns.tolist()
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using mean imputation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        if df.isnull().sum().sum() == 0:
            return df
        
        # Store column names
        columns = df.columns
        
        # Impute missing values
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=columns,
            index=df.index
        )
        
        return df_imputed
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled DataFrame
        """
        columns = df.columns
        
        if fit:
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(df),
                columns=columns,
                index=df.index
            )
        else:
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                columns=columns,
                index=df.index
            )
        
        return df_scaled
    
    def apply_pca_2d(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply PCA for 2D visualization.
        
        Args:
            df: Input DataFrame (should be scaled)
            fit: Whether to fit the PCA (True for training, False for inference)
            
        Returns:
            DataFrame with 2 principal components
        """
        if fit:
            pca_data = self.pca_2d.fit_transform(df)
        else:
            pca_data = self.pca_2d.transform(df)
        
        return pd.DataFrame(
            pca_data,
            columns=['PC1', 'PC2'],
            index=df.index
        )
    
    def apply_pca_3d(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply PCA for 3D visualization.
        
        Args:
            df: Input DataFrame (should be scaled)
            fit: Whether to fit the PCA (True for training, False for inference)
            
        Returns:
            DataFrame with 3 principal components
        """
        if fit:
            pca_data = self.pca_3d.fit_transform(df)
        else:
            pca_data = self.pca_3d.transform(df)
        
        return pd.DataFrame(
            pca_data,
            columns=['PC1', 'PC2', 'PC3'],
            index=df.index
        )
    
    def get_preprocessing_summary(self, df_original: pd.DataFrame, 
                                   df_processed: pd.DataFrame) -> dict:
        """
        Generate a summary of preprocessing steps.
        
        Args:
            df_original: Original DataFrame before preprocessing
            df_processed: Processed DataFrame after preprocessing
            
        Returns:
            Dictionary with preprocessing statistics
        """
        summary = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'missing_values_before': int(df_original.isnull().sum().sum()),
            'missing_values_after': int(df_processed.isnull().sum().sum()),
            'features': list(df_processed.columns),
            'n_features': len(df_processed.columns),
            'n_samples': len(df_processed)
        }
        
        if hasattr(self.pca_2d, 'explained_variance_ratio_'):
            summary['pca_2d_variance'] = {
                'PC1': float(self.pca_2d.explained_variance_ratio_[0]),
                'PC2': float(self.pca_2d.explained_variance_ratio_[1]),
                'total': float(self.pca_2d.explained_variance_ratio_.sum())
            }
        
        return summary
    
    def preprocess_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple of (scaled_data, pca_2d_data, pca_3d_data)
        """
        # Load data
        df = self.load_data()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Scale features
        df_scaled = self.scale_features(df, fit=True)
        
        # Apply PCA
        df_pca_2d = self.apply_pca_2d(df_scaled, fit=True)
        df_pca_3d = self.apply_pca_3d(df_scaled, fit=True)
        
        return df_scaled, df_pca_2d, df_pca_3d
