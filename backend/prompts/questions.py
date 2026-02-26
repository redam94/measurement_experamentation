"""
Per-topic guiding question templates and extraction instructions for the agent.

Each topic entry defines:
  - topic: the canonical key used in covered_topics / ElicitedFacts
  - question: a plain-English question to ask the user
  - extraction_prompt: LLM instruction to parse the user's reply into structured facts
  - followup_triggers: conditions under which to probe deeper before marking the topic done
"""
from __future__ import annotations

TOPIC_QUESTIONS: list[dict] = [
    {
        "topic": "objective",
        "question": (
            "What is the main goal of this ad campaign, and what's the one number you most "
            "want to move? For example: 'increase purchases', 'grow app installs', "
            "'lift brand awareness scores'."
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- primary_objective: one of ['conversion', 'awareness', 'retention', 'engagement', 'other']\n"
            "- kpi: a short string naming the specific metric (e.g. 'purchases', 'install rate', 'brand recall score')\n"
            "Return JSON with those two keys. Use 'unknown' if genuinely unclear."
        ),
        "followup_triggers": ["kpi is 'unknown'", "primary_objective is 'other'"],
    },
    {
        "topic": "randomization",
        "question": (
            "Can you control *who* sees your ad? For instance, could you show the ad to half "
            "your customers and deliberately hide it from the other half? Or does the platform "
            "decide where the ad shows, making that kind of control tricky?"
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- randomization_unit: one of ['user', 'device', 'geo', 'market', 'unknown']\n"
            "- can_run_rct: true if the brand/team can control which users or geos are exposed, false otherwise\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": ["randomization_unit is 'unknown'"],
    },
    {
        "topic": "data_history",
        "question": (
            "How much historical data do you have *before* this campaign launches? "
            "Think in terms of weeks or months of sales, installs, or whatever metric you track."
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- pre_period_weeks: an integer number of weeks of pre-campaign history, or null if unknown\n"
            "- has_historical_data: true if any meaningful history exists\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": ["pre_period_weeks is null"],
    },
    {
        "topic": "geo_structure",
        "question": (
            "How many different cities, regions, or countries does your campaign touch? "
            "And would it be possible to run the campaign in *some* locations but not others, "
            "so you can compare them?"
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- num_markets: integer number of distinct geos/markets, or null if unknown\n"
            "- geo_holdout_feasible: true if some geos can be kept ad-free as a control\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": ["num_markets is null"],
    },
    {
        "topic": "treatment_control",
        "question": (
            "Does your team decide when and where the campaign runs, or does it happen "
            "automatically through a platform (like Google or Meta) with limited control "
            "over exact timing and placement?"
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- campaign_type: one of ['brand_controlled', 'platform_only', 'mixed', 'observational']\n"
            "- control_group_exists: true if a formal no-ad control group is planned or existed\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": [],
    },
    {
        "topic": "covariates",
        "question": (
            "Do you have detailed information about your customers or markets—things like "
            "demographics, past purchase history, location, or how they've behaved before? "
            "Rich background information can make the analysis much more precise."
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- has_rich_covariates: true if meaningful user/market features are available\n"
            "- covariate_description: a short string describing what's available "
            "(e.g. 'age, gender, past 90-day spend', 'city-level GDP and population')\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": [],
    },
    {
        "topic": "scale",
        "question": (
            "Roughly how many people will the campaign reach, and how long do you plan to run it? "
            "Even a ballpark is helpful -- for example 'a few thousand users over two weeks'."
        ),
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- sample_size_estimate: one of ['small (<10k)', 'medium (10k-1M)', 'large (>1M)', 'unknown']\n"
            "- test_duration_weeks: integer weeks the campaign will run, or null if unknown\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": ["sample_size_estimate is 'unknown'"],
    },
]

TOPIC_INDEX: dict[str, dict] = {t["topic"]: t for t in TOPIC_QUESTIONS}

EXTRACTION_SYSTEM = (
    "You are a data extraction assistant. Given a user's conversational reply, "
    "extract the requested fields and return ONLY valid JSON. "
    "Do not add explanation. If a field is genuinely unknown from the reply, use null or 'unknown'."
)

SCORING_EXPLANATION_PROMPT = """
You are an expert marketing measurement scientist. Below are six experimental methods and their
numeric scores (0–100) based on a stakeholder's answers to elicitation questions.

Elicited facts:
{facts_json}

Scores:
{scores_json}

For each method, write 2–3 sentences in plain English explaining WHY it received that score
given this specific situation. Focus on the most important factors. Be honest about limitations.
Return a JSON object mapping each method key to its explanation string.
Method keys: ab_test, did, ddml, geo_lift, synthetic_control, matched_market
"""
