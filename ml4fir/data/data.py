import os

import mlflow
import numpy as np
import pandas as pd

from ml4fir.config import EXPERIMENTS_DIR, logger
from ml4fir.data.config import data_cols
from ml4fir.data.load_data import (
    filter_sample_data,
    preprocess_data,
)


class DataHandler:
    """
    Class to handle data loading and preprocessing.
    """

    def __init__(
        self,
        data_path: str,
        name: str = "FTIR",
        target: str = None,
        ftir_columns=None,
        data_cols_name=None,
        scale=None,
        apply_pls=None,
        apply_smote_resampling=None,
        n_components=None,
        num_classes=None,
        train=True,
    ):
        self.data_path = data_path
        self.name = name
        self.target = target
        self.ftir_columns = None
        self.data_cols_name = data_cols_name or data_cols
        self.scale = scale
        self.apply_pls = apply_pls
        self.apply_smote_resampling = apply_smote_resampling
        self.n_components = n_components
        self.train = train

        self.set_ftrir_columns()
        if num_classes is not None:
            self.set_num_classes(num_classes)
            if self.num_classes == 1:
                logger.info(
                    "Setting smote to False because is a "
                    "linear regression problem"
                )

                self.apply_smote_resampling = False

    def create_example(self):
        df = self.load_data()
        df = df.iloc[:1]
        example_path = os.path.join(EXPERIMENTS_DIR, self.target, "example.csv")
        os.makedirs(os.path.dirname(example_path), exist_ok=True)
        df.to_csv(example_path, index=False)

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

    def set_num_classes(self, num_classes=None):
        """
        Set the number of classes based on the target variable.
        """
        if num_classes is not None:
            self.num_classes = num_classes
            return
        df = self.load_data()
        self.num_classes = len(df[self.target].unique())

    def filter_sample_data(
        self,
        target: str,
        sample_type: str,
        selected_group_fam: str | None = None,
        timepoint: list[int] | int | None = None,
    ):
        """
        Preprocess the loaded data.
        """
        ftir_columns = self.ftir_columns
        df = self.load_data()
        if sample_type not in df["sample_type"].unique():
            raise ValueError(
                f"Sample type '{sample_type}' not found in the data. Available types: {df['sample_type'].unique()}"
            )
        X, y = filter_sample_data(
            sample_data=self.load_data(),
            target=target,
            sample_type=sample_type,
            ftir_columns=ftir_columns,
            selected_group_fam=selected_group_fam,
            timepoint=timepoint,
        )
        self.X = X
        self.Y = y
        return X, y

    def get_wavenumbers(self, X=None):
        """
        Get the wavenumbers from the loaded data.
        """
        if X is None:
            X = self.X

        if hasattr(self, "X"):
            return self.X.columns.values.astype(float)
        raise ValueError("Data not loaded. Please load data first.")

    def encode_sample_data(self, y=None):
        if y is None:
            y = self.Y

        # Make categorical if not already
        if y.unique().size != self.num_classes:
            y = pd.cut(
                y, bins=self.num_classes, labels=np.arange(self.num_classes)
            )
        # Encode target labels
        categorical_array = pd.Categorical(y)
        y_encoded = categorical_array.codes
        labels = categorical_array.categories

        self.y_encoded = y_encoded
        self.labels = labels

        return y_encoded, labels

    def process_sample_data(
        self,
        target: str,
        sample_type: str,
        selected_group_fam: str | None = None,
        num_classes: int | None = None,
        timepoint: list[int] | int | None = None,
    ):
        target = target or self.target
        self.set_num_classes(num_classes)
        X, y = self.filter_sample_data(
            target=target,
            sample_type=sample_type,
            selected_group_fam=selected_group_fam,
            timepoint=timepoint,
        )
        if self.num_classes == 1:
            y_encoded = y.to_numpy()
        else:
            y_encoded, labels = self.encode_sample_data(y=y)
        self.y_encoded = y_encoded
        wavenumbers = self.get_wavenumbers(X=X)
        self.wavenumbers = wavenumbers
        if not self.target:
            self.target = target
        return X, y_encoded, wavenumbers

    def preprocess_data(
        self,
        X=None,
        y_encoded=None,
        train_percentage=0.8,
        random_seed=42,
        scale=None,
        apply_pls=None,
        apply_smote_resampling=None,
        n_components=None,
    ):
        """
        Preprocess the loaded data.
        """
        if X is None:
            X = self.X
        if y_encoded is None:
            y_encoded = self.y_encoded

        if scale is None:
            scale = self.scale
        if apply_pls is None:
            apply_pls = self.apply_pls
        if apply_smote_resampling is None:
            apply_smote_resampling = self.apply_smote_resampling
            if self.num_classes == 1:
                apply_smote_resampling = False
        n_components = n_components or self.n_components

        X_train, X_test, y_train, y_test, loadings = preprocess_data(
            X=X,
            y_encoded=y_encoded,
            train_percentage=train_percentage,
            random_seed=random_seed,
            scale=scale,
            apply_pls=apply_pls,
            apply_smote_resampling=apply_smote_resampling,
            n_components=n_components,
            num_classes=self.num_classes,
        )
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loadings = loadings
        self.train_percentage = train_percentage
        self.n_components = n_components
        self.scale = scale
        self.apply_pls = apply_pls
        self.apply_smote_resampling = apply_smote_resampling
        if self.train:
            self.create_example()
        return X_train, X_test, y_train, y_test, loadings

    def get_mlflow_dataset_complete(self):
        """
        Set the dataset for MLflow tracking.
        """
        # Create an instance of a PandasDataset
        return mlflow.data.from_pandas(
            self.load_data(),
            source=self.data_path,
            name=self.name,
            targets=self.target,
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
