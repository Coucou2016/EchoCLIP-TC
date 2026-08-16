"""EchoCLIP: vision-language foundation model for echocardiogram interpretation."""

from echoclip.checkpoint import load_checkpoint, save_checkpoint
from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP

__all__ = [
    "EchoCLIP",
    "EchoCLIPConfig",
    "load_checkpoint",
    "save_checkpoint",
]
__version__ = "0.2.0"
