from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.loader import Dataset
from utils.logger import Logger
from utils.client import LLMClient
from utils.format.weighting import FeatureWeights, InteractionWeights

DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor


class PriorConstructor:
    def __init__(
        self,
        dataset: Dataset,
        model: str = "gpt-4o-mini",
        logger: Logger = None,
        n_trials: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.dataset = dataset
        self.logger = logger
        self.n_trials = n_trials
        self.temperature = temperature

        # LLM client for prior construction
        self.constructor = LLMClient(model=model, temperature=temperature)

        # Prompt context
        self.human_msg = self._build_message()

        self.feature_weights: dict[str, float] = {}
        self.interaction_weights: dict[tuple[str, str], float] = {}

        self._prior_from_feature()
        self._prior_from_interaction()

    def construct(self, tree: DecisionTree) -> float:
        pass

    def _prior_from_feature(self) -> None:
        system_msg = (
            "You are an AutoML prior designer for decision-tree models. "
            "Given dataset context and feature descriptions, propose a weight in [0,1] for each feature. "
            "Higher weight means stronger prior to prefer splits using that feature. "
            "Only use feature names from the provided list; do not invent new names. "
            "Keep explanations concise and pragmatic."
        )

        user_msg = (
            self.human_msg
            + "\nYour task: Assign a weight in [0,1] for EACH feature listed above.\n"
            + "Guidelines:\n"
            + "- Consider type (categorical/numerical), ranges, cardinality, and plausible relevance to the target.\n"
            + "- If uncertain, assign a moderate weight (e.g., 0.4~0.6).\n"
            + "- Output must strictly follow the required schema.\n"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        valid_feats = set(self.dataset.all_feats)

        for trial in range(self.n_trials):
            try:
                fw: FeatureWeights = self.constructor.get_feature_weights(messages=messages)
                sanitized = self._sanitize_feature_weights(fw, valid_feats)
                # Persist best successful attempt
                self.feature_weights = {fp.name: fp.weight for fp in sanitized.weights}
                if self.logger is not None:
                    self.logger.log_to_json(
                        {
                            "explanation": sanitized.explanation,
                            "weights": self.feature_weights,
                        },
                        f"{self.dataset.name}_feature_priors.json",
                    )
                return
            except Exception as e:
                if trial == self.n_trials - 1:
                    print(f"All trials failed. Error: {e}. Proceeding without feature priors.")
                else:
                    print(f"Trial {trial + 1} failed with error: {e}. Retrying...")

    def _prior_from_interaction(self) -> None:
        system_msg = (
            "You are an AutoML prior designer for decision-tree models. "
            "Propose pairwise feature interaction weights in [0,1]. "
            "Higher weight means a stronger prior that splitting on feature A then feature B (or vice versa if symmetric) is useful. "
            "Only use feature names from the provided list; do not invent new names."
        )

        user_msg = (
            self.human_msg
            + "\nYour task: Suggest a small set of important pairwise interactions with weights in [0,1].\n"
            + "Guidelines:\n"
            + "- Focus on plausible dependencies (e.g., domain logic, confounding, complementary signals).\n"
            + "- Use is_symmetric=true when the interaction doesn't have a meaningful order.\n"
            + "- Prefer quality over quantity (e.g., 5-20 interactions depending on feature count).\n"
            + "- Output must strictly follow the required schema.\n"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        valid_feats = set(self.dataset.all_feats)

        for trial in range(self.n_trials):
            try:
                iw: InteractionWeights = self.constructor.get_interaction_weights(messages=messages)
                sanitized = self._sanitize_interaction_weights(iw, valid_feats)
                # Store as (a,b) -> weight; if symmetric, add both (a,b) and (b,a)
                self.interaction_weights = {}
                for ip in sanitized.weights:
                    a, b, w = ip.feature_a, ip.feature_b, ip.weight
                    if ip.is_symmetric:
                        self.interaction_weights[(a, b)] = w
                        self.interaction_weights[(b, a)] = w
                    else:
                        self.interaction_weights[(a, b)] = w
                if self.logger is not None:
                    weights_list = [
                        {"a": a, "b": b, "weight": w} for (a, b), w in self.interaction_weights.items()
                    ]
                    self.logger.log_to_json(
                        {
                            "explanation": sanitized.explanation,
                            "weights": weights_list,
                        },
                        f"{self.dataset.name}_interaction_priors.json",
                    )
                return
            except Exception as e:
                if trial == self.n_trials - 1:
                    print(f"All trials failed. Error: {e}. Proceeding without interaction priors.")
                else:
                    print(f"Trial {trial + 1} failed with error: {e}. Retrying...")

    def _sanitize_feature_weights(self, fw: FeatureWeights, valid_feats: set[str]) -> FeatureWeights:
        """
        - Keep only weights whose names exist in dataset.all_feats
        - If nothing valid remains, raise to trigger retry/fallback
        - Preserve original explanation
        """
        filtered = [fp for fp in fw.weights if fp.name in valid_feats]
        if len(filtered) == 0:
            raise ValueError("No valid feature names produced by LLM.")
        # If some valid features are missing, we do NOT force coverage; downstream can handle sparsity.
        # However, we can fill missing features with a conservative default to stabilize usage if desired.
        # Here: leave as-is (sparse prior) to respect LLM ranking.
        return FeatureWeights(weights=filtered, explanation=fw.explanation)

    def _sanitize_interaction_weights(self, iw: InteractionWeights, valid_feats: set[str]) -> InteractionWeights:
        """
        - Drop pairs where either feature is invalid or identical.
        - If no pairs remain, return empty list (caller may fallback to empty priors as needed).
        """
        filtered = []
        for ip in iw.weights:
            if ip.feature_a not in valid_feats or ip.feature_b not in valid_feats:
                continue
            if ip.feature_a == ip.feature_b:
                continue
            filtered.append(ip)
        return InteractionWeights(weights=filtered, explanation=iw.explanation)

    def _build_message(self) -> str:
        human_msg = f"DATASET: '{self.dataset.name}'\n"
        human_msg += f"TASK: '{self.dataset.task_type}' with target '{self.dataset.label_col}'.\n\n"

        annotations = self.dataset.annotations
        human_msg += "Feature descriptions (annotated):\n"
        for feat in self.dataset.all_feats:
            desc = annotations.get(feat, "No description available.")
            human_msg += f"- Feature Name: {feat} | Description: {desc}\n"

        return human_msg
