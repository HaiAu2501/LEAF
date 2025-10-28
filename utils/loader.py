import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import fetch_data, preprocess_data
from utils.client import LLMClient
from utils.logger import Logger

class Dataset:
    def __init__(
        self,
        name: str,
        ratio: list[float] = [0.2, 0.4, 0.4],
        model: str = None,
        verbose: bool = False,
        logger: Logger = None
    ):
        self.name = name
        self.ratio = ratio
        if model is not None:
            self.annotator = LLMClient(model=model)
        else:
            print("[WARNING] [Dataset] No model specified for Dataset annotator. Annotations will not be generated.")
            self.annotator = None

        data = fetch_data(
            dataset_name=name,
            ratio=ratio,
            verbose=verbose
        )

        self.task_type = data["task_type"]
        self.label_col = data["label_col"]

        self.train = data["train"]
        self.test = data["test"]

        self.all_feats: list[str] = data["all_feats"]
        self.cat_feats: list[str] = data["cat_feats"]
        self.num_feats: list[str] = data["num_feats"]
        self.descriptions: str = data["descriptions"]
        if model is not None and logger is not None:
            self.annotations: dict[str, str] = self.get_annotations()
            logger.log_to_json(self.annotations, f"{self.name}_annotations.json")

    def get_cat_feats(self):
        return self.cat_feats

    def get_num_feats(self):
        return self.num_feats

    def get_data_descriptions(self):
        return self.descriptions

    def get_annotations(self, n_trials: int = 3) -> dict[str, str]:
        system_msg = (
            "You are an expert data annotator. "
            "Provide concise and clear descriptions for the following data features."
        )

        human_msg = f"The dataset's name is '{self.name}'.\n"
        human_msg += f"The task type is '{self.task_type}', where the goal is to predict '{self.label_col}'.\n"
        human_msg += "Please provide descriptions for the following features:\n"
        for feat in self.all_feats:
            feat_type = "categorical" if feat in self.cat_feats else "numerical"
            if feat_type == "categorical":
                human_msg += f"- Feature Name: {feat} | Type: {feat_type} | Possible Values: {np.unique(self.train[0][feat]).tolist()}\n"
            else:
                human_msg += f"- Feature Name: {feat} | Type: {feat_type} | Range: [{self.train[0][feat].min()}, {self.train[0][feat].max()}]\n"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": human_msg}
        ]
        for trial in range(n_trials):
            try:
                response = self.annotator.get_annotations(messages=messages).descriptions
                annotations = {feat.name: feat.description for feat in response}
                # Check if all features are annotated (list equality)
                if set(annotations.keys()) != set(self.all_feats):
                    raise ValueError("Annotations do not cover all features.")
                return annotations
            except Exception as e:
                if trial == n_trials - 1:
                    print(f"All trials failed. Error: {e}. Proceeding with default annotations.")
                    annotations = {feat: "No description available." for feat in self.all_feats}
                    return annotations
                else:
                    print(f"Trial {trial + 1} failed with error: {e}. Retrying...")
                    continue

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

