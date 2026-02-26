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


# ── Method Assumptions ────────────────────────────────────────────────────────

METHOD_ASSUMPTIONS: dict[str, dict] = {
    "ab_test": {
        "name": "A/B Test (Randomised Controlled Trial)",
        "assumptions": [
            {
                "name": "SUTVA (Stable Unit Treatment Value Assumption)",
                "plain_language": (
                    "Each person's outcome depends ONLY on whether THEY were in the "
                    "treatment group — not on what happened to anyone else."
                ),
                "why_it_matters": (
                    "If treatment 'leaks' between groups (e.g., users share a promo "
                    "code), you'll underestimate the real effect."
                ),
                "when_violated": (
                    "Social products where users interact, viral campaigns, or when "
                    "treatment affects shared resources like inventory."
                ),
                "how_to_check": (
                    "Look for evidence of spillover: do control users mention seeing "
                    "the campaign? Are network-connected users in different groups?"
                ),
            },
            {
                "name": "Random Assignment",
                "plain_language": (
                    "Users are randomly split into groups — like flipping a fair coin "
                    "for each person. No one picks which group they're in."
                ),
                "why_it_matters": (
                    "Without random assignment, the groups might differ in hidden ways "
                    "(e.g., more engaged users end up in treatment), making results "
                    "unreliable."
                ),
                "when_violated": (
                    "When assignment is based on user behaviour, geography, or opt-in "
                    "rather than true randomisation."
                ),
                "how_to_check": (
                    "Check that treatment and control groups look similar on key "
                    "characteristics (age, past purchases, etc.) before the test starts."
                ),
            },
            {
                "name": "No Interference / Spillover",
                "plain_language": (
                    "The campaign doesn't indirectly affect people in the control group."
                ),
                "why_it_matters": (
                    "If control group members hear about the campaign through word-of-mouth "
                    "or see it via shared devices, your measured effect is diluted."
                ),
                "when_violated": (
                    "Household-level treatments with user-level assignment, marketplace "
                    "effects where supply is shared, brand campaigns with broad awareness."
                ),
                "how_to_check": (
                    "Survey control users about campaign awareness; check for geographic "
                    "or network clustering between groups."
                ),
            },
        ],
        "key_terms": {
            "conversion_rate": "The percentage of users who take the desired action (e.g., purchase).",
            "lift": "The improvement caused by the campaign — the difference between treatment and control.",
            "statistical_power": "The probability that your test will detect a real effect if one exists. Like a metal detector's sensitivity.",
            "significance_level": "The threshold for declaring a result 'real' (typically 5%). Lower = stricter.",
            "MDE": "Minimum Detectable Effect — the smallest real improvement your test can reliably spot.",
        },
    },
    "did": {
        "name": "Difference-in-Differences (DiD)",
        "assumptions": [
            {
                "name": "Parallel Trends",
                "plain_language": (
                    "Before the campaign, the treatment and control groups were following "
                    "the same trend over time. If one was going up, so was the other."
                ),
                "why_it_matters": (
                    "DiD measures the effect by looking at how the treatment group's trend "
                    "DIVERGES from the control group's trend after the campaign. If they "
                    "were already diverging before, the estimate is biased."
                ),
                "when_violated": (
                    "When treatment markets have different seasonality, growth rates, "
                    "or are affected by other local events during the test."
                ),
                "how_to_check": (
                    "Plot both groups' trends for the pre-period. They should look parallel. "
                    "Use a placebo test: pretend the campaign started earlier and check for "
                    "a false 'effect'."
                ),
            },
            {
                "name": "No Spillover Between Groups",
                "plain_language": (
                    "The campaign in treatment markets doesn't affect control markets "
                    "(e.g., customers don't switch where they shop)."
                ),
                "why_it_matters": (
                    "If treatment draws customers away from control markets, the control "
                    "group looks worse than it should, inflating the measured effect."
                ),
                "when_violated": (
                    "Adjacent geographic markets, online campaigns where users can see "
                    "the treatment regardless of their assigned market."
                ),
                "how_to_check": (
                    "Use 'buffer zones' — exclude markets bordering treatment areas. "
                    "Check if control market outcomes dip during the test."
                ),
            },
            {
                "name": "Stable Composition",
                "plain_language": (
                    "The mix of people or businesses in each group stays roughly the same "
                    "throughout the test. Nobody important enters or leaves."
                ),
                "why_it_matters": (
                    "If a major retailer opens in a treatment market mid-test, the lift "
                    "might be from that, not your campaign."
                ),
                "when_violated": (
                    "Markets with high turnover, new store openings, competitor launches, "
                    "or seasonal population shifts (college towns, tourist areas)."
                ),
                "how_to_check": (
                    "Monitor for large changes in market composition during the test. "
                    "Exclude markets with known confounding events."
                ),
            },
        ],
        "key_terms": {
            "parallel_trends": "The assumption that treatment and control groups follow the same trajectory before the campaign.",
            "pre_period": "The time window BEFORE the campaign starts — used to establish the baseline trend.",
            "post_period": "The time window DURING/AFTER the campaign — where we measure the effect.",
            "treatment_effect": "The extra change in the treatment group beyond what the control group experienced.",
        },
    },
    "ddml": {
        "name": "Double/Debiased Machine Learning (DDML)",
        "assumptions": [
            {
                "name": "Unconfoundedness (Selection on Observables)",
                "plain_language": (
                    "All the important reasons why some people got the campaign and others "
                    "didn't are captured in the data you have (age, location, past behaviour, etc.)."
                ),
                "why_it_matters": (
                    "If there's a hidden factor that determines both who sees the campaign "
                    "AND the outcome, the estimate will be biased. DDML can only control "
                    "for things you measure."
                ),
                "when_violated": (
                    "When treatment depends on unobserved motivation, private information, "
                    "or decisions you can't track in data."
                ),
                "how_to_check": (
                    "Think hard about what drives treatment assignment. Include as many "
                    "relevant covariates as possible. Run sensitivity analyses to see "
                    "how much an unmeasured confounder would need to matter."
                ),
            },
            {
                "name": "Overlap (Common Support)",
                "plain_language": (
                    "For every combination of characteristics in your data, there are "
                    "some people who got the campaign AND some who didn't."
                ),
                "why_it_matters": (
                    "If a certain type of person ALWAYS gets the campaign (or never does), "
                    "DDML can't estimate the effect for that type — there's no comparison."
                ),
                "when_violated": (
                    "When treatment is deterministic for some subgroups (e.g., all users "
                    "in a certain city always see the ad)."
                ),
                "how_to_check": (
                    "Plot the predicted probability of treatment (propensity score). "
                    "Both groups should have overlapping distributions."
                ),
            },
            {
                "name": "SUTVA (No Interference)",
                "plain_language": (
                    "One person's treatment doesn't affect another person's outcome."
                ),
                "why_it_matters": (
                    "Same as A/B testing — spillover effects bias the results."
                ),
                "when_violated": (
                    "Network effects, marketplace dynamics, or shared resources."
                ),
                "how_to_check": (
                    "Check for clustering in treatment assignment. Look for network "
                    "connections between treated and untreated users."
                ),
            },
        ],
        "key_terms": {
            "covariates": "Background characteristics (age, location, past purchases) used to control for differences between groups.",
            "propensity_score": "The estimated probability that a person receives the treatment, based on their characteristics.",
            "cross_fitting": "A technique that prevents overfitting by splitting data into parts and training models on separate folds.",
            "debiasing": "Removing the bias that comes from using machine learning models for causal estimation.",
        },
    },
    "geo_lift": {
        "name": "Geo Lift Test",
        "assumptions": [
            {
                "name": "Geographic Targeting is Enforced",
                "plain_language": (
                    "The campaign actually runs ONLY in the treatment markets. "
                    "People in control markets do NOT see the campaign."
                ),
                "why_it_matters": (
                    "If the campaign leaks into control markets (e.g., national TV, "
                    "social media sharing), the control group is contaminated and "
                    "you'll underestimate the effect."
                ),
                "when_violated": (
                    "Campaigns with national media components, viral social content, "
                    "or e-commerce where shipping crosses market boundaries."
                ),
                "how_to_check": (
                    "Verify targeting settings in your ad platform. Monitor campaign "
                    "reach or impressions in control markets — they should be near zero."
                ),
            },
            {
                "name": "Market Homogeneity",
                "plain_language": (
                    "The markets in your test behave similarly enough that the control "
                    "markets are a good stand-in for what the treatment markets would "
                    "have done without the campaign."
                ),
                "why_it_matters": (
                    "If treatment markets are fundamentally different (e.g., big cities "
                    "vs rural) from control markets, the comparison breaks down."
                ),
                "when_violated": (
                    "When treatment markets are selected for convenience rather than "
                    "similarity — e.g., picking your biggest markets for treatment."
                ),
                "how_to_check": (
                    "Compare pre-period trends across treatment and control markets. "
                    "Match markets on size, demographics, and historical KPI patterns."
                ),
            },
            {
                "name": "No Cross-Market Contamination",
                "plain_language": (
                    "What happens in treatment markets doesn't spill over and affect "
                    "control markets, or vice versa."
                ),
                "why_it_matters": (
                    "If customers in control markets travel to treatment markets (or shop "
                    "online), they might be exposed to the campaign, diluting the measured "
                    "effect."
                ),
                "when_violated": (
                    "Adjacent markets with lots of commuting, online businesses where "
                    "geography is less meaningful, or when brand awareness spreads nationally."
                ),
                "how_to_check": (
                    "Use geographic buffer zones between treatment and control markets. "
                    "Check for unusual patterns in control market metrics during the test."
                ),
            },
        ],
        "key_terms": {
            "geo": "A geographic unit — could be a city, DMA (Designated Market Area), state, or region.",
            "holdout": "Markets deliberately kept free of the campaign, serving as the control group.",
            "lift": "The incremental effect of the campaign — how much more happened because of it.",
            "pre_period": "Weeks of data BEFORE the campaign, used to establish normal patterns.",
        },
    },
    "synthetic_control": {
        "name": "Synthetic Control Method",
        "assumptions": [
            {
                "name": "No Interference Between Units",
                "plain_language": (
                    "The campaign in the treated region doesn't affect the donor "
                    "(comparison) regions."
                ),
                "why_it_matters": (
                    "The 'synthetic' version of your region is built from donor regions. "
                    "If those donors are affected by the campaign, the synthetic "
                    "counterfactual is wrong."
                ),
                "when_violated": (
                    "When the campaign causes customers to shift between regions, "
                    "or when donor regions compete with the treated region."
                ),
                "how_to_check": (
                    "Check if donor region metrics change after the campaign starts. "
                    "Use donors that are geographically or economically distant."
                ),
            },
            {
                "name": "Interpolation, Not Extrapolation",
                "plain_language": (
                    "The treated region's behaviour (before the campaign) should fall "
                    "WITHIN the range of what the donor regions do — not outside it."
                ),
                "why_it_matters": (
                    "Synthetic control creates a weighted average of donors to match "
                    "the treated region. If the treated region is an outlier, no "
                    "combination of donors can replicate it."
                ),
                "when_violated": (
                    "When the treated region is much larger, richer, or more urban "
                    "than all available donors."
                ),
                "how_to_check": (
                    "Compare the treated region's pre-period metrics to the range "
                    "of donor metrics. The treated values should fall within that range."
                ),
            },
            {
                "name": "Sufficiently Large Donor Pool",
                "plain_language": (
                    "You have enough comparison regions to build a good 'synthetic twin' "
                    "of your treated region."
                ),
                "why_it_matters": (
                    "More donors give the algorithm more building blocks to construct "
                    "an accurate counterfactual. Too few donors = poor match."
                ),
                "when_violated": (
                    "When you only have 3-4 potential donor regions, or when most "
                    "donors are very different from the treated region."
                ),
                "how_to_check": (
                    "Aim for at least 10-15 donors. Check the pre-period fit: "
                    "does the synthetic control closely track the treated region "
                    "before the campaign?"
                ),
            },
        ],
        "key_terms": {
            "donor_pool": "The set of untreated regions used to construct the synthetic counterfactual.",
            "synthetic_counterfactual": "A weighted combination of donor regions that mimics what the treated region would have looked like without the campaign.",
            "pre_period_fit": "How well the synthetic control tracks the treated region before the campaign — a key quality check.",
            "gap": "The difference between the treated region's actual outcome and the synthetic counterfactual.",
        },
    },
    "matched_market": {
        "name": "Matched Market Test",
        "assumptions": [
            {
                "name": "Exchangeable Pairs",
                "plain_language": (
                    "Within each matched pair, the two markets are similar enough "
                    "that you could swap which one gets the campaign and expect "
                    "roughly the same result."
                ),
                "why_it_matters": (
                    "The whole method relies on comparing paired markets. If the pairs "
                    "aren't well-matched, the comparison is unfair."
                ),
                "when_violated": (
                    "When markets are matched on superficial characteristics (same state) "
                    "but differ in important ways (one is urban, one is rural)."
                ),
                "how_to_check": (
                    "Compare pre-period KPI trends within each pair. Good pairs should "
                    "track each other closely. Use correlation or RMSE to quantify match quality."
                ),
            },
            {
                "name": "No Cross-Market Spillover",
                "plain_language": (
                    "The campaign in treatment markets doesn't leak into the paired "
                    "control markets."
                ),
                "why_it_matters": (
                    "If treatment markets and their control partners are nearby, "
                    "customers might cross boundaries and dilute the measured effect."
                ),
                "when_violated": (
                    "When paired markets are geographically adjacent, or when the "
                    "campaign runs on channels that aren't geographically contained."
                ),
                "how_to_check": (
                    "Choose pairs that aren't adjacent. Monitor control market metrics "
                    "for unexpected changes during the test."
                ),
            },
            {
                "name": "Match Quality Maintained Throughout Test",
                "plain_language": (
                    "The paired markets continue to be comparable during the test — "
                    "no big external shocks hit one but not the other."
                ),
                "why_it_matters": (
                    "If an unrelated event (competitor opening, weather event) affects "
                    "only one market in a pair, it looks like a campaign effect."
                ),
                "when_violated": (
                    "During holidays in regions with different shopping patterns, "
                    "natural disasters, or major local competitor actions."
                ),
                "how_to_check": (
                    "Monitor for unusual events during the test. Run placebo tests "
                    "on pre-period data to verify that pairs are stable."
                ),
            },
        ],
        "key_terms": {
            "matched_pair": "Two markets deliberately chosen because they behave similarly — one gets the campaign, the other doesn't.",
            "within_pair_difference": "The difference in outcomes between the treatment and control market in each pair.",
            "match_quality": "How closely the two markets in a pair track each other before the campaign.",
        },
    },
}


# ── Review results prompts ───────────────────────────────────────────────────

REVIEW_RESULTS_PROMPT = """\
You are reviewing experiment design results with a non-technical stakeholder.
Present the results in plain, friendly language.

Method: {method_name} ({method_key})
Setup Parameters: {setup_params_summary}

Power Analysis Results:
- Required sample size: {required_n}
- Achieved power: {achieved_power}
- Effect size used: {effect_size}

MDE Results:
- Minimum Detectable Effect (absolute): {mde_abs}
- MDE as relative %: {mde_rel_pct}

Red Flags: {red_flags_text}

Key Assumptions for this method:
{assumptions_text}

Instructions:
1. Summarize results in 2-3 plain-language sentences. Explain what the numbers mean \
   practically (e.g., "You'd need about 10,000 customers in your test" not "sample \
   size is 10,000").
2. If achieved power is below 0.80 or there are critical red flags, clearly explain \
   the problem and what it means practically. Use analogies — e.g., "With 65% power, \
   your test is like a metal detector that misses the nugget 1 out of 3 times."
3. Briefly mention the top 2-3 assumptions the user should be aware of. Don't go into \
   exhaustive detail — just note them and mention the FAQ page for more info.
4. If there are problems, suggest SPECIFIC changes they could make, for example:
   - "Increase from 10 to 20 markets"
   - "Extend the test from 4 weeks to 8 weeks"
   - "Accept a larger minimum detectable effect (15% instead of 5%)"
5. End by asking: "Would you like to proceed with this design, or would you like \
   to adjust something? Just tell me what you'd like to change and I'll re-run \
   the analysis."
6. Be concise — no more than 15 sentences total. Be encouraging and supportive.
"""

REVIEW_DECISION_SYSTEM = """\
You are analyzing a user's response to an experiment design review.
Determine whether the user wants to:
1. "accept" — proceed with the current design (they're happy, say "looks good", etc.)
2. "modify" — make changes to the design parameters

If they want to modify, extract any specific parameter changes they mention.
Convert percentages to decimals. Interpret colloquial language generously:
- "more markets" / "bigger sample" → increase num_treatment_units and/or num_control_units
- "longer test" / "more time" → increase num_post_periods
- "more history" / "longer pre-period" → increase num_pre_periods
- "bigger effect" / "larger lift" → increase expected_lift_pct or expected_lift_abs
- "stricter" / "more confident" → decrease alpha
- "less strict" / "more lenient" → increase alpha
- "higher power" → increase power_target

Return ONLY valid JSON:
{
  "decision": "accept" or "modify",
  "changes": {
    "num_treatment_units": int_or_null,
    "num_control_units": int_or_null,
    "num_pre_periods": int_or_null,
    "num_post_periods": int_or_null,
    "expected_lift_pct": float_or_null,
    "expected_lift_abs": float_or_null,
    "alpha": float_or_null,
    "power_target": float_or_null,
    "baseline_metric_value": float_or_null,
    "baseline_metric_std": float_or_null,
    "baseline_rate": float_or_null,
    "cluster_size": int_or_null,
    "icc": float_or_null
  },
  "change_summary": "Brief description of what they want to change, or null"
}
"""

REDESIGN_ELICIT_PROMPT = """\
The user wants to modify their experiment design but hasn't specified enough details yet.
Based on what you know about their design and the issues identified, ask them \
specifically what they'd like to change.

Current design problems:
{problems}

Current parameters:
{current_params}

Offer concrete options with specific numbers. For example:
- "Would you like to increase from 10 to 15 or 20 markets?"
- "Should we extend the test from 4 weeks to 6 or 8 weeks?"
- "Would a 10% lift target work instead of the current 5%?"

Ask ONE focused question. Be warm and helpful.
"""


# ── FAQ system prompt ─────────────────────────────────────────────────────────

FAQ_SYSTEM_PROMPT = """\
You are a friendly, patient statistics educator helping a non-technical person \
understand the assumptions and concepts behind experimental design methods for \
measuring ad campaign effectiveness.

Your audience may be a marketing manager, brand director, or business owner with \
NO statistics background. Your job is to make complex concepts crystal clear.

## Guidelines:
- **Use analogies and real-world examples** liberally. Compare statistical power to \
  a metal detector's sensitivity. Compare parallel trends to two runners on a treadmill.
- **Never use jargon without explanation.** If you must use a technical term, \
  immediately follow it with a plain-language equivalent in parentheses.
- **Keep answers focused.** Answer the specific question asked, then offer to go deeper.
- **Use concrete numbers** when explaining concepts: "If your test has 80% power, \
  imagine running it 10 times — you'd catch the real effect 8 out of 10 times."
- **Be honest about limitations.** If an assumption is commonly violated in practice, \
  say so and explain what it means for results.
- **Relate everything back to their business.** Instead of abstract theory, frame \
  answers in terms of campaign launches, customer behavior, and marketing ROI.

## Knowledge Base:
You have deep knowledge of these measurement methods and their assumptions:

1. **A/B Test**: Random assignment, SUTVA, no spillover
2. **Difference-in-Differences (DiD)**: Parallel trends, no spillover, stable composition
3. **Double/Debiased ML (DDML)**: Unconfoundedness, overlap, SUTVA
4. **Geo Lift Test**: Geographic targeting enforced, market homogeneity, no contamination
5. **Synthetic Control**: No interference, interpolation not extrapolation, donor pool size
6. **Matched Market Test**: Exchangeable pairs, no spillover, match quality maintained

You also understand key statistical concepts:
- Statistical power, significance level, p-values
- Minimum Detectable Effect (MDE)
- Sample size determination
- Confidence intervals
- Type I and Type II errors
- Effect sizes and practical significance vs statistical significance

When the user asks about a specific method, provide its assumptions with practical \
examples of when they hold and when they break down.
"""
