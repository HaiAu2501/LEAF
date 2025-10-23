from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.loader import Dataset
from utils.client import LLMClient

DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor

class PriorConstructor:
    def __init__(self, dataset: Dataset, model: str = "gpt-4o-mini") -> None:
        self.annotations = dataset.annotations
        self.prior_constructor = LLMClient(model=model)

        self.prior_feat: dict[str, float] = {}
        self.prior_int: dict[tuple[str, str], float] = {}

    def _prior_from_feature(self) -> None:
        self.prior_feat = {}
        pass

    def _prior_from_interaction(self) -> None:
        self.prior_int = {}
        pass

    def construct(
        self,
        tree: DecisionTree,
        ) -> float:
        pass