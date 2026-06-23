# core/base_handler.py
from abc import ABC, abstractmethod

class BaseHandler(ABC):
    def __init__(self, config=None):
        self.config = config or {}

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        """Load model weights and prepare for inference"""
        pass

    @abstractmethod
    def infer(self, input_data, **kwargs):
        """Run inference on the input data"""
        pass
