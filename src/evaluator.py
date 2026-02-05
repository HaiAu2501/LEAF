import numpy as np

from sklearn.metrics import balanced_accuracy_score, r2_score

from src.template import Algorithm
from utils.loader import Dataset

class Evaluator():
    def __init__(
        self,
        alg: Algorithm,
        dataset: Dataset,
        model: str = None,
        n_data_splits: int = 5,
        n_alg_runs: int = 3,
        random_state: int = 0,
        verbose: bool = False
    ):
        self.alg = alg
        self.dataset = dataset
        self.model = model
        self.n_data_splits = n_data_splits
        self.n_alg_runs = n_alg_runs
        self.random_state = random_state
        self.verbose = verbose
        
        assert alg.task_type == dataset.task_type, "Algorithm and Dataset task types do not match."

        self.task_type = dataset.task_type
        self.val_size = dataset.ratio[1] / (dataset.ratio[1] + dataset.ratio[2])

    def run(self):
        self.alg.setup(self.dataset, self.model)
        scores = []
        for i in range(self.n_data_splits):        
            for j in range(self.n_alg_runs):
                score = self._evaluate(i, j)
                scores.append(score)
                if self.verbose: print(f"Split {i} | Run {j} | Score: {score:.4f}")
        return np.mean(scores), np.std(scores)

    def _evaluate(self, data_split: int, alg_run: int) -> float:
        train, val, test = self.dataset.split(seed=self.random_state + data_split)
        self.alg.fit(train, val, seed=self.random_state + alg_run)
        
        # Evaluate
        X_test, y_test = test
        y_pred = self.alg.predict(X_test)
        if self.task_type == "regression":
            return r2_score(y_test, y_pred)
        elif self.task_type == "classification":
            return balanced_accuracy_score(y_test, y_pred)
            