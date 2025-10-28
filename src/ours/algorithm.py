from src.template import Algorithm
from src.ours.prior_constructor import PriorConstructor
from utils.logger import Logger


class OursAlgorithm(Algorithm):
    def __init__(
        self,
        logger: Logger,
        param_grid: dict[str, list] = None,
    ):
        super().__init__(
            logger=logger,
            name="OursAlgorithm",
            task_type="classification",
            param_grid=param_grid,
        )
        self.prior_constructor = None
    
    def setup(self, dataset, model):
        self.prior_constructor = PriorConstructor(
            dataset=dataset,
            model=model,
            logger=self.logger
        )
        pass

    def fit(self, train, val, seed):
        ...

    def predict(self, X_test):
        ...