import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

DATASET_IDS = {
    # Classification
    "adult": [1590, "classification"],
    "breast": [15, "classification"],
    "credit-g": [31, "classification"],
    "diabetes": [37, "classification"],
    "heart-statlog": [53, "classification"],
    "liver": [1480, "classification"],
    "phoneme": [1489, "classification"],
    "vehicle": [994, "classification"],

    # Regression
    "cpu_activity": [44978, "regression"],
    "wine": [287, "regression"],
    "wage": [534, "regression"],
    "abalone": [44956, "regression"],
    "cars": [44994, "regression"],
    "socmob": [44987, "regression"],
    "sulfur": [44145, "regression"],
    "house_sales": [44144, "regression"],
    "Bike_Sharing_Demand": [44142, "regression"],
    "Brazilian_houses": [44141, "regression"],
}


def impute(X: DataFrame, cat_feats: list[str], num_feats: list[str]) -> DataFrame:
    X = X.copy()
    if len(cat_feats) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_feats] = cat_imputer.fit_transform(X[cat_feats])
    if len(num_feats) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        X[num_feats] = num_imputer.fit_transform(X[num_feats])
    return X

def sort_cat_feats(X_train: DataFrame, y_train: ndarray, cat_feats: list[str]) -> list[str]:
    sorted_cat_feats = []
    df_combined = X_train.copy()
    df_combined["target"] = y_train
    for feat in cat_feats:
        avg = df_combined.groupby(feat)["target"].mean().reset_index()
        sorted_categories = avg.sort_values(by="target")[feat].tolist()
        sorted_categories = [str(cat) for cat in sorted_categories]
        sorted_cat_feats.append(sorted_categories)
    return sorted_cat_feats

def fetch_data(dataset_name: str, ratio: list[float], verbose: bool = False):
    """
    Fetch raw data from OpenML.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to fetch.
    verbose : bool
        Whether to print verbose output.
    """
    try:
        dataset_id, task_type = DATASET_IDS[dataset_name]
    except KeyError:
        print("Dataset not found!")
        print("Supported datasets are:")
        for key in DATASET_IDS.keys():
            print(f"  - {key}")
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = fetch_openml(data_id=dataset_id)
    X: DataFrame = dataset.data
    y: Series = dataset.target
    assert isinstance(X, DataFrame), "X should be a DataFrame"
    assert isinstance(y, Series), "y should be a Series"

    if task_type == "classification" and y.nunique() != 2:
        raise ValueError(f"Dataset '{dataset_name}' is a multi-class classification problem which is not supported.")
    label_col = y.name
    cat_feats = X.select_dtypes(include=["category"]).columns.tolist()
    num_feats = X.select_dtypes(include=["number"]).columns.tolist()
    all_feats = cat_feats + num_feats
    n_cat = len(cat_feats)
    n_num = len(num_feats)

    if verbose:
        print(f"Fetched dataset '{dataset_name}' with ID {dataset_id} for {task_type} task.")
        print(f"Label column: {label_col}")
        print(f"Categorical features: {cat_feats}")
        print(f"Numerical features: {num_feats}")
        print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]} (Categorical: {n_cat}, Numerical: {n_num})")

    ##### PREPROCESSING STEPS #####

    # STEP 1: Handle missing values
    if verbose: print("[1] Handling missing values...")
    X = impute(X, cat_feats, num_feats)
    
    # STEP 2: Split into train and test sets
    if verbose: print("[2] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio[2], random_state=0,
        stratify=y if task_type == "classification" else None
    )

    train = (X_train, y_train)
    test = (X_test, y_test)

    # STEP 3: Create meta-data descriptions
    if verbose: print("[3] Creating meta-data descriptions...")
    data_descriptions = "META-DATA:\n" + f"TASK TYPE: {task_type}\n" + f"LABEL/TARGET COLUMN: {label_col}\n"

    if len(cat_feats) > 0:
        data_descriptions += "CATEGORICAL FEATURES:\n"
        for feat in cat_feats:
            data_descriptions += f"- \"{feat}\" has {X_train[feat].nunique()} unique categories: {X_train[feat].unique().tolist()}\n"

    if len(num_feats) > 0:
        data_descriptions += "NUMERICAL FEATURES:\n"
        for feat in num_feats:
            x = X_train[feat]
            data_descriptions += f"- \"{feat}\": range=[{x.min()}, {x.max()}], mean={x.mean():.2f}, std={x.std():.2f}\n"

    if task_type == "classification":
        data_descriptions += "LABEL DISTRIBUTION:\n"
        label_counts = y_train.value_counts()
        total_count = len(y_train)
        for label, count in label_counts.items():
            data_descriptions += f"- {label}: {count / total_count:.2%}\n"

    if task_type == "regression":
        data_descriptions += f"TARGET VARIABLE: range=[{y_train.min()}, {y_train.max()}], mean={y_train.mean():.2f}, std={y_train.std():.2f}\n"

    data = {
        "task_type": task_type,
        "label_col": label_col,
        "train": train,
        "test": test,
        "all_feats": all_feats,
        "cat_feats": cat_feats,
        "num_feats": num_feats,
        "descriptions": data_descriptions
    }

    return data

def preprocess_data(
        X_train: DataFrame,
        y_train: Series,
        X_val: DataFrame,
        y_val: Series,
        X_test: DataFrame,
        y_test: Series,
        cat_feats: list[str],
        task_type: str
    ) -> tuple[DataFrame, ndarray, DataFrame, ndarray, DataFrame, ndarray]:

    assert (
        isinstance(X_train, DataFrame) 
        and isinstance(X_val, DataFrame) 
        and isinstance(X_test, DataFrame)
    ), "X inputs must be DataFrames"
    assert (
        isinstance(y_train, Series) 
        and isinstance(y_val, Series) 
        and isinstance(y_test, Series)
    ), "y inputs must be Series"

    # STEP 4: Encode/scale target variable
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

    # STEP 5: Encode categorical features
    if len(cat_feats) > 0:
        sorted_cat_feats = sort_cat_feats(X_train, y_train, cat_feats)
        ordinal_encoder = OrdinalEncoder(categories=sorted_cat_feats, handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_feats] = ordinal_encoder.fit_transform(X_train[cat_feats].astype(str))
        X_val[cat_feats] = ordinal_encoder.transform(X_val[cat_feats].astype(str))
        X_test[cat_feats] = ordinal_encoder.transform(X_test[cat_feats].astype(str))

    return X_train, y_train, X_val, y_val, X_test, y_test