from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..core.schemas import ModelMetadata, SingleStepInspectResponse

class BaseInferenceEngine(ABC):
    """Abstract base class for all LLM inference engines (Transformer, RWKV, Custom)."""

    @abstractmethod
    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        """Load the model and tokenizer into memory."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return structural metadata about the loaded model."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize input text and return tokens and tensor representations."""
        pass

    @abstractmethod
    def decode_token(self, token_id: int) -> str:
        """Convert a single token ID back to text representation."""
        pass

    @abstractmethod
    def inspect_single_step(
        self,
        prompt: str,
        layer: int = 0,
        head: int = 0,
        top_k: int = 10,
        temperature: float = 0.7,
    ) -> SingleStepInspectResponse:
        """Run single-pass forward propagation, capturing attention/state and top-k candidate logits."""
        pass
