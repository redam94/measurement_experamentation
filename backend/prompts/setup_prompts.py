"""
Prompts and topic definitions for the experiment setup workflow.
"""
from __future__ import annotations

# ── System prompt for the setup agent ────────────────────────────────────────

SETUP_SYSTEM_PROMPT = """\
You are a friendly, expert experimental design statistician helping a non-technical \
stakeholder set up their chosen measurement method for an ad campaign.

The person you are speaking with may have NO statistics background. They might be a \
marketing manager, brand director, or business owner who wants to run a test but \
has never done one before.

Your job now is to gather the practical details needed to:
1. Calculate the required sample size and statistical power.
2. Run a Monte Carlo simulation to find the minimum detectable effect (MDE).
3. Generate realistic synthetic data so the team can test their analysis before launch.

## Language & Tone
- **Plain language first.** Never use jargon without immediately explaining it in \
  everyday terms. For example, say "how many people you'll need in your test" instead \
  of "required sample size."
- **Use analogies** to explain concepts. For instance, compare statistical power to \
  a metal detector's sensitivity — the more sensitive, the smaller the nugget it can find.
- Be warm, patient, and encouraging. If the user seems unsure, validate their uncertainty \
  and offer concrete options to choose from.

## Adaptive Strategy
- Ask **ONE clear question** at a time.
- **Provide sensible defaults** and let the user confirm or override. Frame defaults as: \
  "Most teams use __, which works well for most situations. Sound good?"
- When the user does not know a value, **suggest a reasonable range** from industry \
  benchmarks with context: "For an e-commerce conversion rate, 2-5% is typical."
- **Probe for variability / standard deviation** — this is critical for power calculations. \
  If the user does not volunteer a standard deviation, ask about the typical range \
  (lowest vs highest week), or typical spread across units. Offer to estimate it: \
  "If your average is 5,000 and it typically swings between 4,000 and 6,000, \
  the variability is roughly 500 — I can work with that."
- When you notice a potential problem with the design (e.g., very few markets, very \
  small expected effect relative to variability, very short pre-period), **raise the \
  concern explicitly in plain language** and suggest concrete alternatives.
- **Confirm understanding** after each answer before moving on: briefly restate what \
  you heard in one sentence.
- **Never make the user feel bad** for not knowing something. Use phrases like \
  "That's totally normal — most teams aren't sure about this" or "Great, I can work with that."
"""

SETUP_WELCOME_TEMPLATE = """\
Great — you've chosen **{method_name}** as your measurement approach! 🎯

Now I'll help you set everything up so you can run a solid test. Don't worry — I'll \
walk you through it step by step, and I'll suggest sensible starting points for \
anything you're not sure about.

Here's what we'll figure out together:

1. **How many people (or markets) you'll need** for reliable results
2. **The smallest real improvement your test can spot** — so you know what's realistic
3. **A practice dataset** so your team can test the analysis before launch

Let's start with the basics…
"""


# ── Setup elicitation topics ─────────────────────────────────────────────────

SETUP_TOPIC_QUESTIONS: list[dict] = [
    {
        "topic": "baseline_metrics",
        "question_template": {
            "ab_test": (
                "Let's talk about your starting point. What does your key metric look "
                "like *right now*, before the campaign?\n\n"
                "For example, if you're tracking conversions: 'About 3 out of every 100 "
                "visitors buy something — so a 3% conversion rate.'\n\n"
                "And here's one that's really important: **how much does that number "
                "bounce around?** Is it pretty steady week to week, or does it swing a "
                "lot? For instance, 'Some weeks it's 2%, other weeks it's 4%.' "
                "Knowing this helps me figure out how many people you'll need in your test."
            ),
            "did": (
                "What does your key metric typically look like in your markets *before* "
                "the campaign starts?\n\n"
                "For example: 'About 5,000 sales per market per week.'\n\n"
                "And just as important: **how much does it bounce around** from week "
                "to week? Like, 'It ranges from about 3,500 to 6,500 in a normal month.' "
                "If you're not sure of the exact spread, just telling me the highest and "
                "lowest typical weeks is great — I can work from there."
            ),
            "ddml": (
                "What's the typical value of the outcome you're measuring, per person? "
                "For example: 'Average revenue per user is about $50.'\n\n"
                "And how spread out is it? For instance: 'Most users spend between $10 "
                "and $120.' I need both the average **and** the spread to figure out "
                "how many observations you'll need."
            ),
            "geo_lift": (
                "For an average market, what does your key metric look like in a typical "
                "week *before* the campaign? For example: 'About 2,000 conversions per "
                "market per week.'\n\n"
                "I also need to know: **how different are markets from each other**, and "
                "**how much does a single market bounce around** week to week? Something "
                "like: 'Markets range from 500 to 5,000, and an individual market might "
                "swing ±300 in a normal week.'"
            ),
            "synthetic_control": (
                "For the region you want to test in (and the comparison regions), what "
                "does your metric typically look like each week? And **how stable** is "
                "it over time?\n\n"
                "For example: '10,000 units per week in our test region, usually within "
                "±1,000 of that.' The more stable the metric, the easier it is to detect "
                "a real change."
            ),
            "matched_market": (
                "For your matched markets, what's the typical weekly value of your metric? "
                "And **how closely do the matched pairs track each other**?\n\n"
                "For example: 'About 1,200 sales/week/market, and our matched pairs are "
                "usually within 200 of each other. Individual markets might swing ±150 "
                "week to week.'"
            ),
            "_default": (
                "What is your key metric currently at — before the campaign starts? "
                "A rough estimate is perfectly fine — for example, a conversion rate, "
                "average weekly sales, or another number.\n\n"
                "And here's one that really matters: **how much does it bounce around?** "
                "If you can tell me the lowest and highest values you see in a normal "
                "week or month, I can work from there."
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
                "Now for the big question: **how much of an improvement do you expect "
                "(or hope) the campaign will create?**\n\n"
                "This can be:\n"
                "- A percentage lift, like 'we're hoping for about a 10% increase'\n"
                "- An absolute change, like 'we want to add 0.5 percentage points to "
                "our conversion rate'\n\n"
                "If you're not sure, that's totally fine! I can suggest typical ranges "
                "for your type of campaign — just tell me and I'll give you some benchmarks."
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
                "For the technical settings, I'll suggest some standard defaults that "
                "work great for most teams:\n\n"
                "- **95% confidence level** — meaning we want to be 95% sure any result "
                "we see is real, not just random noise\n"
                "- **80% power** — meaning an 80% chance of catching a real effect if "
                "one exists (like a metal detector that finds the nugget 4 out of 5 times)\n"
                "- **Two-sided test** — checking whether the campaign helped *or* hurt, "
                "not just one direction\n\n"
                "Do these defaults sound good to you, or would you like to adjust "
                "anything? (Most teams stick with the defaults — they're a great starting point.)"
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
                "Last thing! Do you know roughly how many people will be in each group "
                "(the group that sees the ad and the group that doesn't)? Or would you "
                "like me to calculate the minimum number you'll need?"
            ),
            "did": (
                "A few more details about your setup: how many treatment groups (the "
                "ones getting the ad) and control groups (the ones without) do you have? "
                "And how many time periods before and after the campaign?\n\n"
                "For example: '5 markets with the ad, 10 without, 12 weeks of history "
                "before, and the campaign runs for 8 weeks.'"
            ),
            "ddml": (
                "Roughly how many people (or data rows) will you have to work with? "
                "And how many pieces of background information (like age, location, "
                "past purchases) are available per person?\n\n"
                "For example: '50,000 users with about 20 demographic and behavioral features.'"
            ),
            "geo_lift": (
                "How many markets will run the campaign (treatment group) and how many "
                "will be kept as a comparison (holdout group)? And how many weeks of data "
                "do you have before and after the campaign?\n\n"
                "For example: '8 test markets, 25 holdout markets, 12 weeks before, "
                "6 weeks during the campaign.'"
            ),
            "synthetic_control": (
                "How many comparison regions (we call them 'donors') do you have, and "
                "how many regions are getting the campaign? Also, how many time periods "
                "before and after?\n\n"
                "For example: '1 state with the campaign, 20 comparison states, a year "
                "of weekly data before, and 12 weeks of campaign data.'"
            ),
            "matched_market": (
                "How many matched pairs of markets do you have? And how long is the "
                "pre-period (before the campaign) and the test period?\n\n"
                "For example: '6 matched pairs of similar cities, 8 weeks of history "
                "before, and a 4-week test.'"
            ),
            "_default": (
                "A few more details: how many treatment and control units do you have, "
                "and how many time periods before and after the campaign?"
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
        "Thanks for the baseline number! One more thing that's really important for "
        "sizing your test: **how much does your metric bounce around?**\n\n"
        "Any of these would help me estimate it:\n"
        "- The **highest and lowest** values you've seen in a typical month "
        "(I can do the math from there)\n"
        "- A general sense like 'it's pretty steady' vs 'it swings a lot'\n"
        "- If your data team can pull it, the standard deviation or coefficient "
        "of variation\n\n"
        "This one really matters — a metric that bounces around a lot needs more "
        "{units} to produce a reliable test."
    ),
    "missing_baseline": (
        "I want to make sure I have a clear starting point. "
        "Could you give me a rough idea of what your metric typically looks like?\n\n"
        "For example:\n"
        "- A rate like '2.5% of visitors convert'\n"
        "- An average value like '$45 per customer' or '3,200 sales per week'\n\n"
        "Even a ballpark works great — I just need a starting point to figure out "
        "how big your test needs to be."
    ),
    "missing_both": (
        "To figure out how big your test needs to be, I need two things about "
        "your metric:\n\n"
        "1. **Where it is now** — the current average (like '3% conversion rate' "
        "or '$50 per customer')\n"
        "2. **How much it bounces around** — the typical range (like 'conversions "
        "range from 2% to 4% week-to-week')\n\n"
        "If you're not sure, no worries — just tell me what type of metric it is "
        "(conversion rate, revenue, engagement, etc.) and I can suggest typical "
        "industry values to start with."
    ),
    "unreasonable_values": (
        "I want to double-check a couple of numbers with you: {detail}\n\n"
        "Could you confirm or adjust these? Getting the starting point right is "
        "important because it directly affects how many {units} you'll need for "
        "a reliable test."
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
    "If the user says 'defaults are fine', 'sounds good', 'that works', or similar, "
    "use the standard defaults mentioned in the question (alpha=0.05, power=0.80, "
    "one_sided=false). "
    "The user is non-technical, so interpret colloquial language generously:\n"
    "- 'pretty stable' / 'doesn't change much' → low CV (~0.10–0.15)\n"
    "- 'swings a bit' / 'some variation' → moderate CV (~0.20–0.30)\n"
    "- 'all over the place' / 'really bounces around' → high CV (~0.40–0.60)\n"
    "- 'huge swings' / 'wildly unpredictable' → very high CV (~0.60+)\n"
    "When the user gives a CV description and a baseline value, compute "
    "std = CV × baseline_value."
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
