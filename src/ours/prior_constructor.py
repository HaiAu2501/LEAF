from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.loader import Dataset
from utils.client import LLMClient

DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor

class PriorConstructor:
    def __init__(self, dataset: Dataset, model: str = "gpt-4o-mini") -> None:
        self.dataset = dataset
        self.prior_constructor = LLMClient(model=model)
        self.human_msg = self._build_message()

        self._prior_from_feature()
        self._prior_from_interaction()
        self.n_trials = 3

    def _prior_from_feature(self) -> None:
        self.prior_feat: dict[str, float] = {}
        ...

    def _prior_from_interaction(self) -> None:
        self.prior_int: dict[tuple[str, str], float] = {}
        ...

    def construct(
            self,
            tree: DecisionTree,
        ) -> float:
        pass

    def _build_message(self) -> str:
        human_msg = f"The dataset's name is '{self.dataset.name}'.\n"
        human_msg += f"The task type is '{self.dataset.task_type}', where the goal is to predict '{self.dataset.label_col}'.\n"
        human_msg += "The following are the feature descriptions:\n"
        for feat, desc in self.dataset.annotations.items():
            human_msg += f"- Feature Name: {feat} | Description: {desc}\n"
        return human_msg
