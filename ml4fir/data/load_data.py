from imblearn.over_sampling import SMOTE
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml4fir.config import random_seed


def filter_sample_data(
    sample_data: pd.DataFrame,
    target: str,
    sample_type: str,
    ftir_columns: list,
    selected_group_fam: str | None = None,
) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    """
    Filter and preprocess the sample data for a specific sample type and target.

    This function filters the input data based on the specified sample type and
    optionally by a group family. It ensures that the target column and spectral
    data are valid and free of missing or invalid values.

    Parameters
    ----------
        sample_data (pd.DataFrame): The input data containing samples and features.
        target (str): The name of the target column to predict.
        sample_type (str): The type of sample to filter (e.g., "solid", "liquid").
        ftir_columns (list): List of FTIR columns to use as features.
        selected_group_fam (str, optional): Specific group family to filter by.
            If None, no group family filtering is applied.

    Returns
    -------
        tuple: A tuple containing:
            - X (pd.DataFrame or None): The filtered feature matrix.
            - y (pd.Series or None): The filtered target values.
    """
    logger.info(f"\n--- Processing Sample Type: {sample_type} ---")
    sample_data = sample_data[sample_data["sample_type"] == sample_type].copy()

    # Filter by selected group family if specified
    if selected_group_fam:
        sample_data = sample_data[sample_data["group_fam"] == selected_group_fam]

    # Skip if no valid data for the target
    if sample_data[target].dropna().empty:
        logger.info(f"[!] Skipping: No data for {target} in {sample_type}")
        return None, None, None

    # Extract spectral data and target
    spectral_data = sample_data[ftir_columns]
    y = sample_data[target]

    # Create masks for valid data
    y_valid_mask = y.notna()
    x_valid_nan_mask = spectral_data.notna().all(axis=1)
    x_valid_zero_mask = (spectral_data != 0).any(axis=1)
    x_valid_mask = x_valid_nan_mask & x_valid_zero_mask
    valid_mask = y_valid_mask & x_valid_mask

    # Apply masks to filter valid data
    y = y[valid_mask]
    X = spectral_data.loc[valid_mask]
    return X, y


def process_sample_data(
    sample_data: pd.DataFrame,
    target: str,
    sample_type: str,
    ftir_columns: list,
    selected_group_fam: str | None = None,
) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    """
    Process a single combination of sample_data, target, and sample_type.

    Parameters
    ----------
        sample_data (pd.DataFrame): The data for the current sample type.
        target (str): The target column to predict.
        sample_type (str): The type of sample being processed.
        ftir_columns (list): List of FTIR columns to use as features.
        selected_group_fam (str, optional): Specific group family to filter by.

    Returns
    -------
        tuple: Processed X (features), y_encoded (encoded target), wavenumbers (feature names).
    """
    X, y = filter_sample_data(
        sample_data,
        target,
        sample_type,
        ftir_columns,
        selected_group_fam,
    )

    wavenumbers = X.columns.values.astype(float)

    # Encode target labels
    y_encoded = pd.Categorical(y).codes

    return X, y_encoded, wavenumbers


def split_data(
    X: pd.DataFrame, y_encoded: np.ndarray, train_percentage: float, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
        X (pd.DataFrame): The feature matrix.
        y_encoded (np.ndarray): The encoded target labels.
        train_percentage (float): The percentage of data to use for training (e.g., 0.8 for 80%).
        random_seed (int): The random seed for reproducibility.

    Returns
    -------
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info(f"Training with {train_percentage * 100:.0f}% of the data")
    test_size = 1 - train_percentage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


def scale_data(
    X_train: pd.DataFrame | np.ndarray, X_test: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale the training and testing data using StandardScaler.

    Parameters
    ----------
        X_train (pd.DataFrame or np.ndarray): The training feature matrix.
        X_test (pd.DataFrame or np.ndarray): The testing feature matrix.

    Returns
    -------
        tuple: Scaled X_train and X_test.
    """
    # NOTE: scaling is good and all, but I am always more adept to using real values.
    # In this case the scaling might be just a *1000 or something like that.
    # But whatever is best in academia. For production code, I would use the real values.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def apply_pls_da(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Partial Least Squares Discriminant Analysis (PLS-DA) to the data.

    Parameters
    ----------
        X_train_scaled (np.ndarray): Scaled training feature matrix.
        X_test_scaled (np.ndarray): Scaled testing feature matrix.
        y_train (np.ndarray): Training target labels.
        n_components (int): Number of components to use in PLS-DA.

    Returns
    -------
        tuple: Transformed X_train_pls, X_test_pls, and loadings.
    """
    # NOTE: this i myself need to learn more about. I dont really see how mathematicaly we win from this instead of raw data.
    pls_da = PLSRegression(n_components=n_components)
    X_train_pls = pls_da.fit_transform(X_train_scaled, y_train)[0]
    X_test_pls = pls_da.transform(X_test_scaled)
    loadings = pls_da.x_weights_
    return X_train_pls, X_test_pls, loadings


def apply_smote(
    X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to oversample the minority class in the training data.

    Parameters
    ----------
        X_train (np.ndarray or pd.DataFrame): The training feature matrix.
        y_train (np.ndarray): The training target labels.
        random_seed (int): The random seed for reproducibility.

    Returns
    -------
        tuple: Resampled X_train and y_train.
    """
    smote = SMOTE(random_state=random_seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def preprocess_data(
    X: pd.DataFrame | np.ndarray,
    y_encoded: np.ndarray,
    train_percentage: float,
    random_seed: int = random_seed,
    scale: bool = True,
    apply_pls: bool = True,
    apply_smote_resampling: bool = True,
    n_components: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Preprocess the data by splitting, scaling, applying PLS-DA, and optionally applying SMOTE.

    Parameters
    ----------
        X (pd.DataFrame or np.ndarray): The feature matrix.
        y_encoded (np.ndarray): The encoded target labels.
        train_percentage (float): The percentage of data to use for training (e.g., 0.8 for 80%).
        random_seed (int): The random seed for reproducibility.
        scale (bool): Whether to scale the data using StandardScaler. Default is True.
        apply_pls (bool): Whether to apply PLS-DA. Default is True.
        apply_smote_resampling (bool): Whether to apply SMOTE for oversampling. Default is True.
        n_components (int): Number of components to use in PLS-DA. Default is 10.

    Returns
    -------
        tuple: Preprocessed X_train, X_test, y_train, y_test, and optionally loadings.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y_encoded=y_encoded,
        train_percentage=train_percentage,
        random_seed=random_seed,
    )

    loadings = None

    # Scale the data if scaling is enabled
    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    # Apply PLS-DA if enabled
    if apply_pls:
        X_train, X_test, loadings = apply_pls_da(
            X_train, X_test, y_train, n_components=n_components
        )

    # Apply SMOTE for oversampling if enabled
    # NOTE: This I really dont like. I would rather class weights in the model itself. you balance model itself, and are not feeding artifical data.
    if apply_smote_resampling:
        X_train, y_train = apply_smote(X_train, y_train, random_seed)

    return X_train, X_test, y_train, y_test, loadings
