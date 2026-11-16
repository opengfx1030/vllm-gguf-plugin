# SPDX-License-Identifier: Apache-2.0

from .config_parser import GGUFConfigParser
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

# Largest expert count the MMQ MoE kernels (``moe_q*``) serve. Upstream
# releases skipped experts above id 255 (a hard-coded bound, no marker), so
# consumers treat a missing attribute as 256. Read by vLLM's RDNA GGUF loader.
MOE_MMQ_MAX_EXPERTS = 1 << 20

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "MOE_MMQ_MAX_EXPERTS",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
