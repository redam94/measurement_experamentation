"""
Setup workflow prompts — re-exports from measurement_design.prompts.setup_prompts.

All prompt content now lives in the core domain package.
Domain knowledge (RED_FLAG_CATALOG, METHOD_ASSUMPTIONS) re-exported
from measurement_design.knowledge for backward compatibility.
"""
from measurement_design.prompts.setup_prompts import (  # noqa: F401
    SETUP_SYSTEM_PROMPT,
    SETUP_WELCOME_TEMPLATE,
    SETUP_TOPIC_QUESTIONS,
    SETUP_TOPIC_INDEX,
    SETUP_EXTRACTION_SYSTEM,
    SETUP_REPORT_PROMPT,
    BASELINE_FOLLOWUP_TEMPLATES,
    FEASIBILITY_PREAMBLE_TEMPLATE,
    REVIEW_RESULTS_PROMPT,
    REVIEW_DECISION_SYSTEM,
    REDESIGN_ELICIT_PROMPT,
    FAQ_SYSTEM_PROMPT,
)

# Re-export domain knowledge for backward compatibility
from measurement_design.knowledge.red_flags import RED_FLAG_CATALOG  # noqa: F401
from measurement_design.knowledge.assumptions import METHOD_ASSUMPTIONS  # noqa: F401
