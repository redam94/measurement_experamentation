"""
Prompts and topic definitions for the experiment setup workflow.
"""
from __future__ import annotations

# ── System prompt for the setup agent ────────────────────────────────────────

SETUP_SYSTEM_PROMPT = """\
You are a friendly, expert experimental design statistician helping a non-technical \
stakeholder set up their chosen measurement method for an ad campaign.

Your job now is to gather the practical details needed to:
1. Calculate the required sample size and statistical power.
2. Run a Monte Carlo simulation to find the minimum detectable effect (MDE).
3. Generate realistic synthetic data so the team can test their analysis before launch.

Guidelines:
- Ask ONE clear question at a time.
- Use plain English — no jargon without a simple explanation.
- Provide sensible defaults and let the user confirm or override.
- When the user does not know a value, suggest a reasonable range from industry benchmarks.
- **Probe for variability / standard deviation** — this is critical for power calculations. \
  If the user does not volunteer a standard deviation, ask about the typical range \
  (lowest vs highest week), or typical spread across units.  You can also offer to \
  estimate it from a coefficient of variation (e.g., "If your average is 5,000 and \
  it typically swings between 4,000 and 6,000, the standard deviation is roughly 500.").
- When you notice a potential problem with the design (e.g., very few markets, very \
  small expected effect relative to variability, very short pre-period), **raise the \
  concern explicitly** and suggest concrete alternatives the user could consider.
"""

SETUP_WELCOME_TEMPLATE = """\
Great — you've chosen **{method_name}** as your measurement approach! 🎯

Before we generate your analysis code, let me help you set up the experiment properly. \
I'll walk you through a few practical questions so I can:

1. **Calculate the sample size** you'll need for reliable results.
2. **Find the minimum detectable effect** — the smallest real improvement your test can spot.
3. **Generate synthetic test data** so you can validate the analysis pipeline before launch.

Let's start with the basics…
"""


# ── Setup elicitation topics ─────────────────────────────────────────────────

SETUP_TOPIC_QUESTIONS: list[dict] = [
    {
        "topic": "baseline_metrics",
        "question_template": {
            "ab_test": (
                "What is your current baseline conversion rate (or the average value of "
                "your KPI)? For example, if 3 out of every 100 visitors normally convert, "
                "that's a 3% baseline.\n\n"
                "Also — how much does this rate vary day-to-day or week-to-week? For "
                "instance, does it swing between 2% and 4%, or is it very stable? This "
                "variability is key for figuring out how many users you'll need."
            ),
            "did": (
                "What's the typical weekly average for your KPI across your treatment "
                "and control groups before the campaign starts?\n\n"
                "Equally important: how variable is it week to week? For example, "
                "'about 5,000 sales per market per week, but it ranges from 3,500 to "
                "6,500 in a typical month.' If you're not sure of the exact standard "
                "deviation, giving me the lowest and highest typical weeks is just as useful."
            ),
            "ddml": (
                "What's the average value of your outcome variable (KPI), and roughly "
                "how spread out is it across users? For example, 'average revenue per "
                "user is $50, with most users between $10 and $120.'\n\n"
                "I need both the average **and** the spread (standard deviation or range) "
                "to calculate how many observations you'll need."
            ),
            "geo_lift": (
                "What's the typical weekly KPI value for an average market/geo before "
                "the campaign starts? And how much does it vary **between** markets as "
                "well as **within** a market week-to-week?\n\n"
                "For example: 'avg 2,000 conversions/week/market, ranging from 500 to "
                "5,000 across markets, and a given market might swing ±300 week to week.'"
            ),
            "synthetic_control": (
                "What's the typical weekly KPI value for your treatment region and "
                "the average donor market? How stable is it over time?\n\n"
                "Specifically, I need the average level and how much it fluctuates "
                "week-to-week (e.g., '10,000 units/week for the treatment region, "
                "usually within ±1,000')."
            ),
            "matched_market": (
                "What's the typical weekly KPI value per market, and how much does it "
                "vary between your matched pairs?\n\n"
                "For example, 'about 1,200 sales/week/market, pairs are usually within "
                "200 of each other, and individual markets swing ±150 week-to-week.'"
            ),
            "_default": (
                "What is your current baseline value for the KPI you're measuring? "
                "A rough estimate is fine — for example, a conversion rate, "
                "average weekly sales, or another metric.\n\n"
                "Also, how much does this KPI typically vary? Knowing the range "
                "(lowest to highest typical value) helps me calculate the right sample size."
            ),
        },
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- baseline_rate: a float between 0 and 1 if they mention a rate/proportion, else null\n"
            "- baseline_metric_value: a float for the average metric value, or null\n"
            "- baseline_metric_std: a float for the standard deviation / variability, or null\n"
            "  If the user gives a range (low to high), estimate std ≈ (high − low) / 4.\n"
            "  If they describe variability qualitatively ('very stable' → low CV ~0.1; "
            "'quite variable' → high CV ~0.4), compute std = CV × mean.\n"
            "Return JSON with those three keys. Infer reasonable values if the user gives "
            "ranges or approximate descriptions."
        ),
    },
    {
        "topic": "expected_effect",
        "question_template": {
            "_default": (
                "How big an effect do you expect (or hope) the campaign will have? "
                "This can be a percentage lift (e.g., '10% improvement') or an absolute "
                "change (e.g., '+0.5 percentage points on conversion rate'). "
                "If you're unsure, I can suggest typical ranges for your type of campaign."
            ),
        },
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- expected_lift_pct: relative lift as a decimal (0.10 for 10%), or null\n"
            "- expected_lift_abs: absolute change in the metric, or null\n"
            "Return JSON with those two keys. If the user gives a percentage, "
            "set expected_lift_pct. If they give an absolute number, set expected_lift_abs. "
            "Both can be set if calculable."
        ),
    },
    {
        "topic": "statistical_design",
        "question_template": {
            "_default": (
                "I'll use some standard defaults, but let me check: are you happy with a "
                "**5% significance level** (95% confidence) and **80% power**? "
                "These are very standard choices. Also, do you want a one-sided test "
                "(only checking if the campaign *helped*) or two-sided "
                "(checking if it helped *or hurt*)?\n\n"
                "Just say 'defaults are fine' if you'd like to keep the standard settings."
            ),
        },
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- alpha: significance level as a decimal (default 0.05 if user accepts defaults)\n"
            "- power_target: desired power as a decimal (default 0.80 if user accepts defaults)\n"
            "- one_sided: true if user wants one-sided test, false for two-sided "
            "(default false if user accepts defaults)\n"
            "Return JSON with those three keys."
        ),
    },
    {
        "topic": "method_specific",
        "question_template": {
            "ab_test": (
                "Last detail: do you know roughly how many users will be in each group "
                "(treatment and control)? Or should I calculate the minimum needed?"
            ),
            "did": (
                "How many treatment units (markets/groups) and control units do you have? "
                "And how many time periods before and after the campaign? "
                "(e.g., '5 treatment markets, 10 control markets, 12 weeks pre, 8 weeks post')"
            ),
            "ddml": (
                "Roughly how many observations (users/rows) will you have? "
                "And how many covariates/features are available for the ML models? "
                "(e.g., '50,000 users with about 20 demographic and behavioral features')"
            ),
            "geo_lift": (
                "How many treatment geos and control geos will you have? "
                "And how many weeks before/after the campaign? "
                "(e.g., '8 test markets, 25 holdout markets, 12 weeks pre, 6 weeks post')"
            ),
            "synthetic_control": (
                "How many donor (control) markets do you have, and how many treatment "
                "regions? Also, how many pre-campaign and post-campaign time periods? "
                "(e.g., '1 treatment state, 20 donor states, 52 weeks pre, 12 weeks post')"
            ),
            "matched_market": (
                "How many matched pairs of markets do you have? "
                "And how long is the pre-period and test period? "
                "(e.g., '6 matched pairs, 8 weeks pre, 4 weeks test')"
            ),
            "_default": (
                "A few more details about your specific setup: how many treatment and "
                "control units do you have, and how many time periods before and after?"
            ),
        },
        "extraction_prompt": (
            "From the user's reply, extract any of these that apply:\n"
            "- num_treatment_units: int or null\n"
            "- num_control_units: int or null\n"
            "- num_pre_periods: int or null\n"
            "- num_post_periods: int or null\n"
            "- cluster_size: average units per cluster, int or null\n"
            "- icc: intra-cluster correlation, float or null\n"
            "Return JSON. Use null for anything not mentioned."
        ),
    },
]

SETUP_TOPIC_INDEX: dict[str, dict] = {t["topic"]: t for t in SETUP_TOPIC_QUESTIONS}


# ── Follow-up question templates ─────────────────────────────────────────────
# Used when extraction fails to get critical values on the first pass.

BASELINE_FOLLOWUP_TEMPLATES: dict[str, str] = {
    "missing_std": (
        "Thanks for the baseline value! One more thing I need to size your experiment "
        "properly: **how much does your KPI typically vary?**\n\n"
        "If you're not sure of the exact standard deviation, any of these would help:\n"
        "- The **lowest and highest** weekly values you've seen recently "
        "(I can estimate the spread from that)\n"
        "- A rough sense like 'it's pretty stable' vs 'it swings a lot'\n"
        "- The **coefficient of variation** if your data team can pull it\n\n"
        "This variability drives how many {units} you'll need, so it's worth getting "
        "a reasonable estimate."
    ),
    "missing_baseline": (
        "I didn't quite catch a concrete baseline number. Could you give me a rough "
        "estimate of your KPI's current average value?\n\n"
        "For example:\n"
        "- A conversion rate like '2.5%'\n"
        "- An average value like '$45 per user' or '3,200 sales per week'\n\n"
        "Even a ballpark is fine — I'll use it as a starting point for the power calculation."
    ),
    "missing_both": (
        "To calculate the right sample size, I need two pieces of information about "
        "your KPI:\n\n"
        "1. **The current average** (e.g., '3% conversion rate' or '$50/user')\n"
        "2. **How much it varies** (e.g., 'conversions range from 2% to 4% week-to-week')\n\n"
        "If you're not sure, I can suggest typical industry values — just let me know "
        "what type of KPI you're tracking (conversion rate, revenue, engagement, etc.)."
    ),
    "unreasonable_values": (
        "I want to double-check a couple of numbers: {detail}\n\n"
        "Could you confirm or correct these? Getting the baseline right is important "
        "because it directly affects how many {units} you'll need for a reliable test."
    ),
}


# ── Feasibility preamble template ────────────────────────────────────────────
# Prepended to the next question when interim power results are available.

FEASIBILITY_PREAMBLE_TEMPLATE = """\
Before we continue, here's a quick **feasibility check** based on what you've told me so far:

📊 **Estimated sample size needed**: ~{required_n:,} {unit_label}
⚡ **Achievable power** (with {achieved_power:.0%} at current settings)
🎯 **Effect you're targeting**: {effect_desc}

{red_flag_section}
{suggestion_section}\
Now let me ask about the next part of your setup…
"""


# ── Red flag description templates ───────────────────────────────────────────

RED_FLAG_CATALOG: dict[str, dict] = {
    "high_cv": {
        "title": "High variability relative to effect size",
        "detail": (
            "Your KPI's coefficient of variation is {cv:.0%}. With an expected lift "
            "of {lift}, the signal-to-noise ratio is low, meaning you'll need a very "
            "large sample to detect the effect reliably."
        ),
        "suggestion": (
            "Consider: (a) using a variance-reduction technique like CUPED/regression "
            "adjustment, (b) targeting a larger effect or a less noisy KPI, or "
            "(c) extending the test duration to accumulate more data."
        ),
    },
    "too_few_geos": {
        "title": "Too few geographic units",
        "detail": (
            "You have {n_total} total markets ({n_treat} treatment + {n_ctrl} control). "
            "Geo-based methods typically need at least 15–20 total markets for reliable "
            "inference, and ideally 30+."
        ),
        "suggestion": (
            "Consider: (a) adding more markets to both groups, (b) switching to a "
            "user-level A/B test if randomisation is feasible, or (c) using synthetic "
            "control with a single treatment region and many donors."
        ),
    },
    "short_pre_period": {
        "title": "Pre-period may be too short",
        "detail": (
            "You have {n_pre} pre-campaign periods. Methods like DiD and synthetic "
            "control rely on the pre-period to establish a baseline trend — fewer than "
            "8 periods makes parallel-trends assumptions harder to verify."
        ),
        "suggestion": (
            "If possible, extend the pre-period to at least 8–12 weeks. Alternatively, "
            "if you have daily data, consider using daily granularity for more observations."
        ),
    },
    "tiny_effect": {
        "title": "Very small expected effect",
        "detail": (
            "The expected lift of {lift} is quite small relative to the baseline "
            "variability. You would need approximately {required_n:,} {unit_label} "
            "to detect this reliably."
        ),
        "suggestion": (
            "Consider: (a) whether a larger effect is plausible (even 2× the lift "
            "cuts required sample size by ~75%), (b) running the test longer, or "
            "(c) accepting lower power (e.g., 70%) if this is an exploratory test."
        ),
    },
    "high_icc": {
        "title": "High intra-cluster correlation",
        "detail": (
            "The ICC of {icc:.2f} means observations within each cluster are quite "
            "similar, reducing the effective sample size substantially."
        ),
        "suggestion": (
            "Consider randomising at a finer level (e.g., user-level instead of "
            "market-level), increasing the number of clusters, or using covariates "
            "to absorb within-cluster correlation."
        ),
    },
    "few_donors_sc": {
        "title": "Too few donor units for synthetic control",
        "detail": (
            "You have only {n_donors} donor units. Synthetic control works best with "
            "a rich donor pool (at least 10, ideally 20+) so the algorithm can find "
            "a good weighted combination to match the treatment unit."
        ),
        "suggestion": (
            "Consider: (a) adding more potential donor regions, (b) using sub-regions "
            "or DMAs instead of states, or (c) switching to DiD if you can designate "
            "a proper control group."
        ),
    },
    "extreme_imbalance": {
        "title": "Large treatment/control imbalance",
        "detail": (
            "Your ratio of treatment to control units is {ratio:.1f}:1. Very "
            "unbalanced designs lose statistical efficiency."
        ),
        "suggestion": (
            "Aim for a ratio between 1:1 and 3:1. If you can't change group sizes, "
            "consider stratified randomisation or matching to improve balance."
        ),
    },
    "insufficient_sample": {
        "title": "Available sample may be too small",
        "detail": (
            "Based on your inputs, you'd need ~{required_n:,} {unit_label} but you "
            "indicated having roughly {available_n:,}. This means you're likely "
            "under-powered for the target effect size."
        ),
        "suggestion": (
            "Options: (a) run the test longer to accumulate more observations, "
            "(b) increase the treatment proportion, (c) accept a larger minimum "
            "detectable effect, or (d) reduce noise with covariates (CUPED)."
        ),
    },
}


# ── Extraction system prompt ─────────────────────────────────────────────────

SETUP_EXTRACTION_SYSTEM = (
    "You are a data extraction assistant for experiment setup parameters. "
    "Given a user's conversational reply, extract the requested numerical fields "
    "and return ONLY valid JSON. Do not add explanation. "
    "If a field is genuinely unknown from the reply, use null. "
    "Convert percentages to decimals (e.g. '10%' → 0.10). "
    "If the user describes variability as a range (low to high), estimate "
    "standard deviation as (high - low) / 4. "
    "If the user says 'defaults are fine' or similar, use the standard defaults "
    "mentioned in the question (alpha=0.05, power=0.80, one_sided=false)."
)


# ── Summary / report generation prompts ──────────────────────────────────────

SETUP_REPORT_PROMPT = """\
You are an expert experimental design statistician. Generate a clear, practical \
Markdown report summarising the experiment setup for a non-technical stakeholder.

Method: {method_name} ({method_key})
Elicited Facts: {facts_json}
Setup Parameters: {setup_params_json}
Power Analysis: {power_json}
MDE Simulation: {mde_json}
Synthetic Data Summary: {synthetic_summary}
Validation Results: {validation_json}
Red Flags & Warnings: {red_flags_json}

Structure the report with:
1. **Setup Summary** — key parameters at a glance
2. **Power Analysis** — required sample size and what it means in plain English
3. **Minimum Detectable Effect** — what the test can/can't detect, with practical context
4. **Synthetic Data Description** — what was generated and how it matches their scenario
5. **Code Validation** — whether the analysis pipeline ran successfully on synthetic data
6. **Warnings & Red Flags** — if any red flags were detected, explain each one in plain \
   language with the severity (⚠️ warning or 🚨 critical), what it means practically, \
   and what the team can do about it. If no red flags, say "No major concerns detected."
7. **Recommendations** — practical next steps before launch, incorporating any warnings
"""
