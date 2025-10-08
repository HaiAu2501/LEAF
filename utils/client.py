import os
from openai import OpenAI
from typing import Any, Dict, List, Optional

from utils.format.annotating import Annotating
from utils.format.coding import Coding
from utils.format.ranking import Ranking


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
    
    def get_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        Generate text response from the model.
        
        Args:
            messages: List of message dictionaries
            function_id: Optional function ID for logging
            temperature: Optional temperature override
            
        Returns:
            Text response from the model
        """
        temp = temperature if temperature is not None else self.temperature
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            stream=False,
        ).choices[0].message.content
        
        return response

    def get_annotations(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> dict[str, str]:
        """
        Generate annotations from the model.
        
        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override 

        Returns:
            Dictionary of annotations
        """
        temp = temperature if temperature is not None else self.temperature

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            format=Annotating,
            stream=False,
        ).choices[0].message.content

        return response