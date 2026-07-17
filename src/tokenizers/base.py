from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import torch

class BasePhonobyteTokenizer(ABC):
    """
    Abstract Base Class for all language-specific Phonobyte tokenizers.
    Guarantees a unified API interface for downstream deep learning loaders.
    """
    
    def __init__(self):
        # Every language tokenizer will track its own vocabulary limits
        self.vocab_size = 256  # Base output space for 8-bit registers
        
    @abstractmethod
    def encode(self, text: str, return_tensors: str = "list") -> Tuple[Union[List[int], torch.Tensor], Union[List[int], torch.Tensor]]:
        """
        Converts raw text into parallel phonotactic streams.
        
        Args:
            text: The raw input string.
            return_tensors: Format of returned variables ("list" or "pt" for PyTorch).
            
        Returns:
            A tuple of (acoustic_masks, lexical_ids).
        """
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Reconstructs original, human-readable text from a sequence of lexical IDs.
        
        Args:
            ids: A sequence of physical register IDs.
            
        Returns:
            The reconstructed, properly formatted string.
        """
        pass

    def _to_tensor(self, data: List[int], return_tensors: str) -> Union[List[int], torch.Tensor]:
        """
        Helper method to handle tensor conversion consistently across all tokenizers.
        """
        if return_tensors == "pt":
            return torch.tensor(data, dtype=torch.long)
        return data

    def pack_8bit_register(self, onset: int, nucleus: int, coda: int) -> int:
        """
        Packs physical phonetic features into a single, unified 8-bit unsigned integer.
        Provides a universally consistent packing utility for phonotactic registers.
        """
        # Ensure values fit their allocated bit ranges before shifting
        # Example assumes standard allocations, subclasses can override packing logic if bits shift
        return ((onset & 0x0F) << 4) | ((nucleus & 0x01) << 3) | (coda & 0x07)