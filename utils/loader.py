import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import fetch_data, preprocess_data

class Dataset():
    def __init__(self, name: str, ratio: list[float] = [0.2, 0.4, 0.4], verbose: bool = False):
        self.name = name
        self.ratio = ratio

        data = fetch_data(
            dataset_name=name,
            ratio=ratio,
            verbose=verbose
        )

        self.task_type = data["task_type"]
        self.label_col = data["label_col"]

        self.train = data["train"]
        self.test = data["test"]
        
        self.all_feats = data["all_feats"]
        self.cat_feats = data["cat_feats"]
        self.num_feats = data["num_feats"]
        self.descriptions = data["descriptions"]

    def get_cat_feats(self):
        return self.cat_feats

    def get_num_feats(self):
        return self.num_feats

    def get_data_descriptions(self):
        return self.descriptions

    def get_annotations(self):
        pass

    def split(self, seed: int = 0):
        X_train, y_train = self.train
        X_test, y_test = self.test
        X_test = X_test.copy(deep=True)
        y_test = y_test.copy(deep=True)
        val_size = self.ratio[1] / (self.ratio[0] + self.ratio[1])

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=seed,
            stratify=y_train if self.task_type == "classification" else None
        )

        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            self.cat_feats,
            self.task_type
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

