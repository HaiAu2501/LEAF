import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error

from src.template import Algorithm
from utils.loader import Dataset

class Evaluator():
    def __init__(
        self,
        alg: Algorithm,
        dataset: Dataset,
        n_data_splits: int,
        n_alg_runs: int,
        random_state: int = 0
    ):
        self.alg = alg
        self.dataset = dataset
        self.n_data_splits = n_data_splits
        self.n_alg_runs = n_alg_runs
        self.random_state = random_state
        
        assert alg.task_type == dataset.task_type, "Algorithm and Dataset task types do not match."

        self.task_type = dataset.task_type
        self.val_size = dataset.ratio[1] / (dataset.ratio[1] + dataset.ratio[2])

    def run(self):
        scores = []
        for i in range(self.n_data_splits):        
            for j in range(self.n_alg_runs):
                score = self._evaluate(i, j)
                scores.append(score)
        return np.mean(scores), np.std(scores)

    def _evaluate(self, data_split: int, alg_run: int) -> float:
        train, val = self.dataset.split(seed=self.random_state + data_split)
        test = self.dataset.test
        self.alg.fit(train, val, seed=self.random_state + alg_run)
        preds = self.alg.predict(test[0])
        if self.task_type == "regression":
            return root_mean_squared_error(test[1], preds)
        elif self.task_type == "classification":
            return balanced_accuracy_score(test[1], preds)
            