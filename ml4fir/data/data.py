import mlflow
import pandas as pd

from ml4fir.data.load_data import (
    filter_sample_data,
    preprocess_data,
)
from ml4fir.data.config import data_cols

# TODO: rename modules: from ml4fir.data.process import process_sample_data


class DataHandler:
    """
    Class to handle data loading and preprocessing.
    """

    def __init__(self, data_path: str, name: str = "FTIR", target: str = None, ftir_columns=None, data_cols_name=None):
        self.data_path = data_path
        self.name = name
        self.target = target
        self.ftir_columns = None
        self.data_cols_name = data_cols_name or data_cols
        self.set_ftrir_columns()

    def load_data(self):
        """
        Load data from the specified path.
        """
        return pd.read_csv(self.data_path)

    def set_ftrir_columns(self):
        """
        Set the FTIR columns.
        """
        df = self.load_data()
        ftir_columns = df.columns[~df.columns.isin(self.data_cols_name)]

        self.ftir_columns = ftir_columns

    def filter_sample_data(
        self,
        target: str,
        sample_type: str,
        selected_group_fam: str | None = None,
    ):
        """
        Preprocess the loaded data.
        """
        ftir_columns = self.ftir_columns
        X, y = filter_sample_data(
            sample_data=self.load_data(),
            target=target,
            sample_type=sample_type,
            ftir_columns=ftir_columns,
            selected_group_fam=selected_group_fam,
        )
        self.X = X
        self.Y = y
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

        self.wavenumbers = wavenumbers
        self.y_encoded = y_encoded
        self.labels = labels

        return y_encoded, wavenumbers, labels

    def process_sample_data(
        self,
        target: str,
        sample_type: str,
        selected_group_fam: str | None = None,
    ):
        target = target or self.target
        X, y = self.filter_sample_data(
            target=target,
            sample_type=sample_type,
            selected_group_fam=selected_group_fam,
        )
        y_encoded, wavenumbers, labels = self.encode_sample_data(X=X, y=y)
        if not self.target:
            self.target = target
        return X, y_encoded, wavenumbers

    def preprocess_data(
        self,
        X=None,
        y_encoded=None,
        train_percentage=0.8,
        random_seed=42,
        scale=True,
        apply_pls=True,
        apply_smote_resampling=True,
        n_components=10,
    ):
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
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loadings = loadings
        self.train_percentage = train_percentage
        self.n_components = n_components
        return X_train, X_test, y_train, y_test, loadings

    def get_mlflow_dataset_complete(self):
        """
        Set the dataset for MLflow tracking.
        """
        # Create an instance of a PandasDataset
        return mlflow.data.from_pandas(
            self.load_data(), source=self.data_path, name=self.name, targets=self.target
        )

    def get_mlflow_dataset(self):
        """
        Set the dataset for MLflow tracking.
        """
        name = self.name
        name = f"{name}_{self.target}_{self.train_percentage}_n_components_{self.n_components}"
        training_dataset = mlflow.data.from_numpy(
            self.x_train.astype(float),
            source=self.data_path,
            name=f"{name}_train",
            targets=self.y_train.astype(float),
        )
        testing_dataset = mlflow.data.from_numpy(
            self.x_test.astype(float),
            source=self.data_path,
            name=f"{name}_test",
            targets=self.y_test.astype(float),
        )
        return training_dataset, testing_dataset
