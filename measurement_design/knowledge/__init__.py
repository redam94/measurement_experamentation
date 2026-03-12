"""
Domain knowledge — structured data about elicitation topics, method assumptions,
data schemas, and red flag catalogs.
"""
from .schemas import (
    TOPIC_LABELS,
    ALL_TOPICS,
    SETUP_TOPIC_LABELS,
    ALL_SETUP_TOPICS,
    METHOD_NAMES,
    PANEL_METHODS,
    COMPUTATION_PHASES,
    METHOD_SCHEMAS,
)
from .topics import (
    ELICITATION_TOPIC_META,
    ALL_ELICITATION_TOPICS,
    SETUP_TOPIC_META,
    ALL_SETUP_TOPIC_KEYS,
)
from .assumptions import METHOD_ASSUMPTIONS
from .red_flags import RED_FLAG_CATALOG

__all__ = [
    "TOPIC_LABELS",
    "ALL_TOPICS",
    "SETUP_TOPIC_LABELS",
    "ALL_SETUP_TOPICS",
    "METHOD_NAMES",
    "PANEL_METHODS",
    "COMPUTATION_PHASES",
    "METHOD_SCHEMAS",
    "ELICITATION_TOPIC_META",
    "ALL_ELICITATION_TOPICS",
    "SETUP_TOPIC_META",
    "ALL_SETUP_TOPIC_KEYS",
    "METHOD_ASSUMPTIONS",
    "RED_FLAG_CATALOG",
]
