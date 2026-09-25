"""Correctness-first CPU inference for nano-dsv4.1f.

This package follows the DeepSeek V4.1 inference architecture. It deliberately does not
import or copy from ``vllm.models.deepseek_v4``.
"""

from .model import NanoDeepseekV41CPU
from .config_io import model_config_from_export

__all__ = ["NanoDeepseekV41CPU", "model_config_from_export"]
