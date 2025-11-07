import numpy as np
from math import lgamma, log
from scipy.special import gammaln
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
        temperature: float = 1.0,
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

    def construct(self, tree: DecisionTree, feature_names: list[str] | None = None) -> float:
        """
        Compute log prior logP(T) for a fitted sklearn DecisionTree using ONLY LLM-derived priors.

        - Feature prior: Dirichlet-multinomial purely from LLM weights (no uniform mixing).
        - Interaction prior: multiplicative tilts per parent->child feature pair via log(1 + beta*s).

        Returns:
            float: logP(T)
        """
        # Hyperparameters
        tau = None     # Dirichlet concentration; default = d
        beta = 1.0     # interaction strength
        eps = 1e-12

        # --- feature names
        if feature_names is None:
            feature_names = getattr(tree, "feature_names_in_", None)
            if feature_names is None:
                # Fall back to using the DataFrame columns passed by caller; require explicit feature_names
                raise ValueError(
                    "Need feature_names_in_ on the fitted tree or pass feature_names explicitly to construct()."
                )
        feature_names = list(feature_names)
        # Optional safety: ensure provided names match the tree's dimensionality
        n_in = getattr(tree, "n_features_in_", None)
        if n_in is not None and len(feature_names) != int(n_in):
            raise ValueError(f"feature_names length {len(feature_names)} != tree.n_features_in_ {n_in}")
        d = len(feature_names)
        idx2name = {i: feature_names[i] for i in range(d)}

        # --- tree structure
        t = tree.tree_
        feat_idx = t.feature  # -2 is leaf
        INTERNAL = np.where(feat_idx >= 0)[0]
        L = int(INTERNAL.size)

        # --- LLM-only feature prior
        fw = dict(self.feature_weights or {})
        # Ensure all features used by this tree have weights
        missing = [f for f in feature_names if f not in fw]
        if len(missing) > 0:
            raise ValueError(
                f"feature_weights missing entries for tree features: {missing}"
            )
        # Slice to this tree's feature subset
        w = np.array([float(fw[f]) for f in feature_names], dtype=float)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            raise ValueError("Sum of feature weights must be positive and finite.")
        r = w / sw  # LLM-only proportions
        tau_eff = float(d) if tau is None else float(tau)
        alpha = tau_eff * r
        alpha = np.maximum(alpha, eps)
        alpha0 = float(alpha.sum())

        counts = np.zeros(d, dtype=int)
        for n in INTERNAL:
            counts[feat_idx[n]] += 1

        logP_feat = lgamma(alpha0) - lgamma(alpha0 + L) + float(np.sum(gammaln(alpha + counts) - gammaln(alpha)))

        # --- interactions prior
        s_map = {}
        for (a, b), s in (self.interaction_weights or {}).items():
            s = float(max(0.0, min(1.0, float(s))))
            s_map[(a, b)] = max(s_map.get((a, b), 0.0), s)

        n_nodes = t.node_count
        left, right = t.children_left, t.children_right
        parent = np.full(n_nodes, -1, dtype=int)
        for p in INTERNAL:
            for ch in (left[p], right[p]):
                if ch != -1:
                    parent[ch] = p

        logP_int = 0.0
        for n in INTERNAL:
            p = parent[n]
            if p != -1 and feat_idx[p] >= 0:
                g = idx2name[feat_idx[p]]
                f = idx2name[feat_idx[n]]
                s = s_map.get((g, f), 0.0)
                if s > 0.0 and beta != 0.0:
                    logP_int += log(1.0 + beta * s)

        logP = float(logP_feat + logP_int)
        return logP

    def _prior_from_feature(self) -> None:
        system_msg = (
            "You are an AutoML prior designer for decision-tree models. "
            "Given dataset context and feature descriptions, propose a weight in [0,1] for each feature. "
            "Higher weight means stronger prior to prefer splits using that feature. "
            "Use the full scale [0,1] when appropriate, and provide exactly one weight for every listed feature. "
            "Only use feature names from the provided list; do not invent, merge, or rename features. "
            "Keep explanations concise and pragmatic."
        )

        user_msg = (
            self.human_msg
            + "\nYour task: Assign a weight w in [0,1] to EACH feature (use the full range when justified).\n"
            + "Weighting rubric:\n"
            + "- 0.90-1.00: Direct proxy/causal driver of the target; very strong and reliable signal; clear monotonic trend.\n"
            + "- 0.70-0.80: Strong predictor with solid domain rationale or likely strong separation power.\n"
            + "- 0.50-0.60: Moderately informative; default when mildly relevant but uncertain.\n"
            + "- 0.30-0.40: Weak or ambiguous relation; noisy/redundant; high cardinality with arbitrary codes; many missing values.\n"
            + "- 0.10-0.20: Marginal relevance; unlikely to help much.\n"
            + "Guidelines:\n"
            + "- Provide exactly one weight for every listed feature; do not add/remove/rename features.\n"
            + "- Weights need not sum to 1; absolute scaling is acceptable (we will normalize downstream).\n"
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
        - Enforce FULL coverage: names must match dataset.all_feats exactly.
        - Preserve original explanation
        """
        filtered = [fp for fp in fw.weights if fp.name in valid_feats]
        names = {fp.name for fp in filtered}
        if names != valid_feats:
            missing = list(valid_feats - names)
            extra = list(names - valid_feats)
            raise ValueError(
                f"Feature weights do not cover all features. Missing: {missing}; Extra: {extra}"
            )
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
