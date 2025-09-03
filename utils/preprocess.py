import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from typing import Dict, List, Tuple

DATASET_IDS = {
    "credit-g": [31, "classification"],
    "diabetes": [37, "classification"],
    "compas": [42192, "classification"],
    "heart-statlog": [53, "classification"],
    "liver": [1480, "classification"],
    "breast": [15, "classification"],
    "vehicle": [994, "classification"],
    "cholesterol": [204, "regression"],
    "wine": [287, "regression"],
    "wage": [534, "regression"],
    "abalone": [44956, "regression"],
    "cars": [44994, "regression"],
}

def get_raw_data(dataset_name: str, verbose: bool = False) -> pd.DataFrame:
    dataset_id, task_type = DATASET_IDS[dataset_name]
    X, y = fetch_openml(data_id=dataset_id, return_X_y=True)
    cat_feats = X.select_dtypes(include=["category"]).columns.tolist()
    num_feats = X.select_dtypes(include=["number"]).columns.tolist()
    n_cat = len(cat_feats)
    n_num = len(num_feats)
    if verbose:
        print(f"Fetched dataset '{dataset_name}' with ID {dataset_id} for {task_type} task.")
        print(f"Categorical features: {cat_feats}")
        print(f"Numerical features: {num_feats}")
        print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]} (Categorical: {n_cat}, Numerical: {n_num})")
    return X, y, cat_feats, num_feats, task_type

def impute(X: pd.DataFrame, cat_feats: List[str], num_feats: List[str]) -> pd.DataFrame:
    if len(cat_feats) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_feats] = cat_imputer.fit_transform(X[cat_feats])
    if len(num_feats) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        X[num_feats] = num_imputer.fit_transform(X[num_feats])
    return X

def sort_cat_feats(X_train: pd.DataFrame, y_train: np.ndarray, cat_feats: List[str]) -> List[str]:
    sorted_cat_feats = []
    df_combined = X_train.copy()
    df_combined["target"] = y_train
    for feat in cat_feats:
        avg = df_combined.groupby(feat)["target"].mean().reset_index()
        sorted_categories = avg.sort_values(by="target")[feat].tolist()
        sorted_categories = [str(cat) for cat in sorted_categories]
        sorted_cat_feats.append(sorted_categories)
    return sorted_cat_feats

def get_meta_data(X_train: pd.DataFrame, y_train: np.ndarray, cat_feats: List[str], num_feats: List[str], task_type: str) -> str:
    data_descriptions = f"TASK TYPE: {task_type}\n"
    if len(cat_feats) > 0:
        data_descriptions += "CATEGORICAL FEATURES:\n"
        for feat in cat_feats:
            data_descriptions += f"- {feat} has {X_train[feat].nunique()} unique categories: {X_train[feat].unique().tolist()}\n"

    if len(num_feats) > 0:
        data_descriptions += "NUMERICAL FEATURES:\n"
        for feat in num_feats:
            x = X_train[feat]
            data_descriptions += f"- {feat}: range=[{x.min()}, {x.max()}], mean={x.mean():.2f}, std={x.std():.2f}\n"

    if task_type == "classification":
        data_descriptions += "LABEL DISTRIBUTION:\n"
        label_counts = pd.Series(y_train).value_counts()
        total_count = len(y_train)
        for label, count in label_counts.items():
            data_descriptions += f"- {label}: {count / total_count:.2%}\n"

    if task_type == "regression":
        data_descriptions += f"TARGET VARIABLE: range=[{y_train.min()}, {y_train.max()}], mean={y_train.mean():.2f}, std={y_train.std():.2f}\n"

    return data_descriptions

def preprocess(
    dataset_name: str,
    ratio: List[float] = [0.6, 0.2, 0.2],
    seed: int = 42,
    verbose: bool = False
) -> Tuple[Dict, str]:
    """
    Preprocess the dataset by fetching, imputing, splitting, and encoding features.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to preprocess.
    ratio : list of float, optional
        The train/validation/test split ratios (default is [0.6, 0.2, 0.2]).
    seed : int, optional
        The random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print detailed information during preprocessing (default is False).
    """

    # Step 1: Fetch raw data
    if verbose:
        print("[INFO] Fetching raw data...")
    X, y, cat_feats, num_feats, task_type = get_raw_data(dataset_name, verbose)

    # Step 2: Impute missing values
    if verbose:
        print("[INFO] Imputing missing values...")
    X = impute(X, cat_feats, num_feats)

    # Step 3: Train/val/test split
    if verbose:
        print("[INFO] Splitting data into train/validation/test sets...")
    assert sum(ratio) == 1.0, "The sum of the split ratios must be 1.0"
    val_size = ratio[1]
    test_size = ratio[2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed
        )
    else:
        X_val, y_val = X_train, y_train

    # Step 4: Encode target variable
    if verbose:
        print("[INFO] Encoding/scaling target variable...")

    y_train_original = y_train.copy()
    if task_type == "classification":
        target_encoder = LabelEncoder()    
        y_train = target_encoder.fit_transform(y_train)
        y_val = target_encoder.transform(y_val)
        y_test = target_encoder.transform(y_test)
    elif task_type == "regression":
        y_train = y_train.astype(float).to_numpy().reshape(-1, 1)
        y_val = y_val.astype(float).to_numpy().reshape(-1, 1)
        y_test = y_test.astype(float).to_numpy().reshape(-1, 1)
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
        y_test = target_scaler.transform(y_test)

    # Step 5: Encode categorical features
    X_train_original = X_train.copy()
    if len(cat_feats) > 0:
        if verbose:
            print("[INFO] Encoding categorical features...")
        sorted_cat_feats = sort_cat_feats(X_train, y_train, cat_feats)
        ordinal_encoder = OrdinalEncoder(categories=sorted_cat_feats, handle_unknown="use_encoded_value", unknown_value=-1)

        X_train[cat_feats] = ordinal_encoder.fit_transform(X_train[cat_feats].astype(str))
        X_val[cat_feats] = ordinal_encoder.transform(X_val[cat_feats].astype(str))
        X_test[cat_feats] = ordinal_encoder.transform(X_test[cat_feats].astype(str))

    if verbose:
        print("[INFO] Shape of training set:", X_train.shape, y_train.shape)
        print("[INFO] Shape of validation set:", X_val.shape, y_val.shape)
        print("[INFO] Shape of test set:", X_test.shape, y_test.shape)

    data = {}
    data["train"] = (X_train.to_numpy(), y_train)
    data["val"] = (X_val.to_numpy(), y_val)
    data["test"] = (X_test.to_numpy(), y_test)
    data["task_type"] = task_type

    data_descriptions = get_meta_data(X_train_original, y_train_original, cat_feats, num_feats, task_type)

    if verbose:
        print("[INFO] Data preprocessing completed.")
        print(data_descriptions)

    return data, data_descriptions