import os
from openai import OpenAI
from typing import Any, Dict, List, Optional

from utils.format.annotating import Annotation
from utils.format.weighting import (
    FeaturePrior, FeatureWeights,
    InteractionPrior, InteractionWeights
)

class LLMClient:
    """
    LLM client for competitive MCTS.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0
    ) -> None:
        """
        Initialize the LLM client.
        
        Args:
            model: The model to use for generation
            temperature: The temperature to use for generation
        """
        self.model = model
        self.temperature = temperature

        # Setup client
        if model.startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("Unsupported model type. Only 'gpt' is supported.")

        self.client = OpenAI(api_key=api_key)
    
    def _get_structured_response(
        self,
        messages: List[Dict[str, str]],
        response_format: type,
        temperature: Optional[float] = None
    ) -> Any:
        temp = temperature if temperature is not None else self.temperature

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            response_format=response_format,
        ).choices[0].message.parsed

        return response

    def get_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            stream=False,
        ).choices[0].message.content
        
        return response

    def get_annotations(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Annotation:
        return self._get_structured_response(
            messages,
            response_format=Annotation,
            temperature=temperature
        )
    
    def get_feature_weights(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> FeatureWeights:
        return self._get_structured_response(
            messages,
            response_format=FeatureWeights,
            temperature=temperature
        )

    def get_interaction_weights(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> InteractionWeights:
        return self._get_structured_response(
            messages,
            response_format=InteractionWeights,
            temperature=temperature
        )