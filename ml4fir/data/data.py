
from ml4fir.data.load_data import process_sample_data, filter_sample_data, preprocess_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
import os


# TODO: rename modules: from ml4fir.data.process import process_sample_data

class DataHandler():
    """
    Class to handle data loading and preprocessing.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        """
        Load data from the specified path.
        """
        return pd.read_csv(self.data_path)
        
    def filter_sample_data(self, target: str, sample_type: str, ftir_columns: list, selected_group_fam: Optional[str] = None):
        """
        Preprocess the loaded data.
        """
        X, y = filter_sample_data(
            sample_data=self.load_data(),
            target=target,
            sample_type=sample_type,
            ftir_columns=ftir_columns,
            selected_group_fam=selected_group_fam,
        )
        self.X=X
        self.Y=y
        return X, y


    def encode_sample_data(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.Y

        wavenumbers = X.columns.values.astype(float)
        # Encode target labels
        y_encoded = pd.Categorical(y).codes
        labels = pd.Categorical(y).categories

        self.wavenumbers=wavenumbers
        self.y_encoded=y_encoded
        self.labels=labels

        return y_encoded, wavenumbers, labels

    def process_sample_data(self, target: str, sample_type: str, ftir_columns: list, selected_group_fam: Optional[str] = None):
        X,y = self.filter_sample_data(
            target=target,
            sample_type=sample_type,
            ftir_columns=ftir_columns,
            selected_group_fam=selected_group_fam,
        )
        y_encoded, wavenumbers, labels = self.encode_sample_data(X=X, y=y)
        return X, y_encoded, wavenumbers

    def preprocess_data(self, X=None, 
        y_encoded=None, train_percentage=0.8, random_seed=42,
        scale=True, apply_pls=True, apply_smote_resampling=True, n_components=10):
        """
        Preprocess the loaded data.
        """
        if X is None:
            X = self.X
        if y_encoded is None:
            y_encoded = self.y_encoded

        X_train, X_test, y_train, y_test, loadings = preprocess_data(
            X=X,
            y_encoded=y_encoded,
            train_percentage=train_percentage,
            random_seed=random_seed,
            scale=scale,  # Enable scaling
            apply_pls=apply_pls,  # Enable PLS-DA
            apply_smote_resampling=apply_smote_resampling,  # Enable SMOTE
            n_components=n_components,  # Number of PLS components
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loadings = loadings
        return X_train, X_test, y_train, y_test, loadings
