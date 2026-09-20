"""Self-contained implementation of QG-PRQK Semantic IDs."""

from qg_prqk.config import QGPRQKConfig, QGPRQKConfigError, load_config
from qg_prqk.data.query_supervision_config import (
    CategoryBuildConfig,
    CategoryConfigError,
    load_category_config,
)

__all__ = [
    "CategoryBuildConfig",
    "CategoryConfigError",
    "QGPRQKConfig",
    "QGPRQKConfigError",
    "load_category_config",
    "load_config",
]
__version__ = "2.2.0"
