"""
Per-topic guiding question templates and extraction instructions for the agent.

Each topic entry defines:
  - topic: the canonical key used in covered_topics / ElicitedFacts
  - question: a plain-English question to ask the user
  - examples: concrete examples to help the user understand what we're asking
  - clarification_hints: suggestions for rephrasing if the user seems confused
  - extraction_prompt: LLM instruction to parse the user's reply into structured facts
  - followup_triggers: conditions under which to probe deeper before marking the topic done
  - followup_question: a gentler follow-up to ask when triggers fire
  - why_it_matters: plain-language explanation so the LLM can justify the question if asked
"""
from __future__ import annotations

TOPIC_QUESTIONS: list[dict] = [
    {
        "topic": "objective",
        "question": (
            "What is the main goal of this ad campaign? If you could move just one "
            "number, what would it be? For example: 'get more people to buy our "
            "product,' 'increase app downloads,' or 'make more people aware of our brand.'"
        ),
        "examples": [
            "We want to increase online purchases by new customers.",
            "We're trying to get more people to sign up for our free trial.",
            "We want to lift our brand awareness score in key markets.",
            "We want to drive more foot traffic to our stores.",
        ],
        "clarification_hints": [
            "Think about what your boss would ask you to report on after the campaign ends.",
            "If you had a dashboard with only one number on it, what would you want to see go up (or down)?",
            "Imagine the campaign is over and a colleague asks 'Did it work?' — what number would you point to?",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- primary_objective: one of ['conversion', 'awareness', 'retention', 'engagement', 'other']\n"
            "- kpi: a short string naming the specific metric (e.g. 'purchases', 'install rate', 'brand recall score')\n"
            "Return JSON with those two keys. Use 'unknown' if genuinely unclear.\n"
            "Tip: 'sales', 'purchases', 'sign-ups', 'downloads', 'installs' \u2192 'conversion'.\n"
            "'brand recall', 'awareness', 'ad recall', 'consideration' \u2192 'awareness'.\n"
            "'repeat purchases', 'churn reduction', 'loyalty' \u2192 'retention'.\n"
            "'clicks', 'engagement rate', 'time on site', 'video views' \u2192 'engagement'."
        ),
        "followup_triggers": ["kpi is 'unknown'", "primary_objective is 'other'"],
        "followup_question": (
            "I want to make sure I understand \u2014 could you give me a concrete example "
            "of the number or result you'd look at to decide if the campaign worked? "
            "For instance, would it be a sales number, a sign-up count, a survey score, "
            "or something else?"
        ),
        "why_it_matters": (
            "Knowing the specific metric helps us choose the right statistical method "
            "and design an experiment that can actually detect a meaningful change."
        ),
    },
    {
        "topic": "randomization",
        "question": (
            "Here's an important one: can you control *who* sees your ad and who doesn't? "
            "For example, could you show the ad to half your customers and deliberately "
            "hide it from the other half? Or does the ad platform (like Google or Meta) "
            "decide who sees it?"
        ),
        "examples": [
            "Yes, we can split our email list and only show the ad to some people.",
            "No, we buy ads on Google/Meta and the platform decides who sees them.",
            "We could choose to run the campaign in some cities but not others.",
            "We have a mobile app and can control who sees in-app promotions.",
        ],
        "clarification_hints": [
            "Think of it like a taste test \u2014 could you give samples to some people and not others, "
            "or does everyone in the store get one automatically?",
            "If your ad runs on TV or billboards, everyone in that area sees it and you can't "
            "really control that. But if it's digital, you might be able to.",
            "This is about whether you can create a 'no-ad' group for comparison. "
            "It's fine if you can't \u2014 we just need to know so we pick the right approach.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- randomization_unit: one of ['user', 'device', 'geo', 'market', 'unknown']\n"
            "- can_run_rct: true if the brand/team can control which users or geos are exposed, false otherwise\n"
            "Return JSON with those two keys.\n"
            "Hints: if they mention splitting users/emails/app users \u2192 user-level, can_run_rct=true.\n"
            "If they mention cities/regions/states \u2192 geo-level or market-level.\n"
            "If a platform decides \u2192 usually can_run_rct=false unless they have a holdout feature."
        ),
        "followup_triggers": ["randomization_unit is 'unknown'"],
        "followup_question": (
            "No worries \u2014 this can be tricky! Let me put it differently: "
            "if you wanted to, could you run the campaign in some cities but keep "
            "it out of other similar cities, so you could compare the two? Or is the "
            "campaign going to run everywhere at once with no way to create a comparison group?"
        ),
        "why_it_matters": (
            "Whether you can create a fair comparison group (people who don't see the ad) "
            "determines which measurement methods are available. If you can, we have "
            "more powerful options. If not, we still have great approaches \u2014 they just "
            "work a bit differently."
        ),
    },
    {
        "topic": "data_history",
        "question": (
            "Do you have data from *before* this campaign starts? For example, weeks or "
            "months of past sales, website visits, or whatever metric you're tracking? "
            "Even a rough idea like 'we have about a year of weekly sales data' is helpful."
        ),
        "examples": [
            "We have about 2 years of monthly sales data by region.",
            "We track daily website conversions going back 6 months.",
            "We don't really have any historical data \u2014 this is a new product.",
            "We have some data but it's messy and inconsistent.",
        ],
        "clarification_hints": [
            "Think about any reports or dashboards you look at regularly. "
            "How far back does that data go?",
            "If someone asked you 'what were your sales last quarter?', could you answer? "
            "If so, that means you have historical data!",
            "Even imperfect data is useful. If you track anything over time \u2014 even in a spreadsheet \u2014 that counts.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- pre_period_weeks: an integer number of weeks of pre-campaign history, or null if unknown\n"
            "  (Convert months \u2192 weeks \u00d7 4, years \u2192 weeks \u00d7 52 approximately)\n"
            "- has_historical_data: true if any meaningful history exists\n"
            "Return JSON with those two keys."
        ),
        "followup_triggers": ["pre_period_weeks is null and has_historical_data is true"],
        "followup_question": (
            "Great that you have some past data! Could you give me a rough idea of "
            "how far back it goes? For example, is it a few weeks, a few months, "
            "or a year or more? This helps me figure out which analysis methods "
            "will work best for your situation."
        ),
        "why_it_matters": (
            "Historical data lets us establish what 'normal' looks like before the "
            "campaign starts. The more history we have, the more confident we can be "
            "that any change we see was actually caused by the campaign."
        ),
    },
    {
        "topic": "geo_structure",
        "question": (
            "How many different locations does your campaign cover \u2014 like cities, states, "
            "or countries? And would it be possible to run the campaign in *some* locations "
            "but deliberately skip others, so you can compare them?"
        ),
        "examples": [
            "We're running in 30 US cities \u2014 we could probably skip 10 of them.",
            "It's a national campaign, so it covers the whole country.",
            "We have 50 retail stores across 20 states.",
            "It's just in one city \u2014 we don't have other locations to compare.",
        ],
        "clarification_hints": [
            "Think of each city, state, or region as a separate 'market.' "
            "How many of those do you have?",
            "Imagine you're testing a new recipe at some of your restaurant locations "
            "but keeping the old menu at others. Could you do something like that with your ad?",
            "Even if you can't do a perfect holdout, knowing how many regions you have "
            "helps me understand your options.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- num_markets: integer number of distinct geos/markets, or null if unknown\n"
            "- geo_holdout_feasible: true if some geos can be kept ad-free as a control\n"
            "Return JSON with those two keys.\n"
            "If they say 'national' or 'everywhere', geo_holdout_feasible is likely false "
            "unless they mention the possibility of withholding from some areas."
        ),
        "followup_triggers": ["num_markets is null"],
        "followup_question": (
            "Even a rough count helps! For instance, do you sell in a handful of cities "
            "(say 5\u201310), dozens of markets (20\u201350), or hundreds of locations? "
            "And would your team be open to keeping the ad out of some of those locations "
            "as a comparison, or would that be too costly?"
        ),
        "why_it_matters": (
            "The number of separate locations determines whether we can use powerful "
            "geographic comparison methods. Even 10\u201315 locations can be enough if "
            "some can serve as a comparison group."
        ),
    },
    {
        "topic": "treatment_control",
        "question": (
            "Who's in the driver's seat for this campaign \u2014 your team or the ad platform? "
            "Specifically, does your team decide exactly when and where the campaign runs, "
            "or does a platform like Google or Meta handle the targeting and timing automatically?"
        ),
        "examples": [
            "We control everything \u2014 we decide which markets get the ads and when.",
            "It's all through Meta's ad manager, so they optimize delivery.",
            "Some mix \u2014 we pick the markets, but Google decides who within those markets sees the ad.",
            "We just set a budget and the platform does the rest.",
        ],
        "clarification_hints": [
            "Think of it like ordering a pizza vs. going to a buffet. Do you choose exactly "
            "what goes on the pizza (you control targeting), or do you just pick from "
            "what's available (the platform decides)?",
            "If you could pause the campaign in one city and keep it running in another, "
            "that means you have control. If the platform decides where to spend your "
            "budget, that's platform-controlled.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- campaign_type: one of ['brand_controlled', 'platform_only', 'mixed', 'observational']\n"
            "- control_group_exists: true if a formal no-ad control group is planned or existed\n"
            "Return JSON with those two keys.\n"
            "'brand_controlled': team has full control over targeting and timing.\n"
            "'platform_only': platform handles everything (e.g., Google/Meta auto-optimization).\n"
            "'mixed': team picks some aspects, platform handles others.\n"
            "'observational': no planned experiment, just measuring what happened naturally."
        ),
        "followup_triggers": [],
        "followup_question": None,
        "why_it_matters": (
            "How much control you have affects which testing methods are feasible. "
            "More control means more options, but even with less control we can still "
            "design a solid measurement approach."
        ),
    },
    {
        "topic": "covariates",
        "question": (
            "Do you have background information about your customers or markets \u2014 things "
            "like demographics, past purchase history, location details, or how they've "
            "behaved before? This kind of information can make the analysis much more precise, "
            "like adjusting for the fact that some stores are bigger than others."
        ),
        "examples": [
            "We have age, gender, and past 90-day spend for each customer.",
            "We know each market's population, GDP, and historical sales.",
            "We just have email addresses \u2014 no other info.",
            "We have a CRM with detailed customer profiles.",
        ],
        "clarification_hints": [
            "Think about what you know about your customers beyond their purchases. "
            "Do you know their age? Location? How long they've been a customer?",
            "Even simple info like 'big market vs. small market' or 'new customer vs. "
            "returning customer' counts as useful background data.",
            "If you have a CRM, a loyalty program, or a customer database, "
            "you probably have great background data.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- has_rich_covariates: true if meaningful user/market features are available\n"
            "- covariate_description: a short string describing what's available "
            "(e.g. 'age, gender, past 90-day spend', 'city-level GDP and population')\n"
            "Return JSON with those two keys.\n"
            "If only basic info (like email addresses) \u2192 has_rich_covariates=false.\n"
            "If demographic, behavioral, or market-level features \u2192 has_rich_covariates=true."
        ),
        "followup_triggers": [],
        "followup_question": None,
        "why_it_matters": (
            "Background data helps us reduce 'noise' in the results \u2014 like accounting "
            "for the fact that some customers spend more than others regardless of "
            "the campaign. The more we know, the smaller the experiment can be."
        ),
    },
    {
        "topic": "scale",
        "question": (
            "Last one! Roughly how many people will this campaign reach, and how long "
            "do you plan to run it? A ballpark is totally fine \u2014 for example, "
            "'a few thousand users over two weeks' or 'millions of people for three months.'"
        ),
        "examples": [
            "We expect about 100,000 visitors per month, running for 4 weeks.",
            "Small test \u2014 maybe 5,000 users over a week.",
            "National campaign reaching millions, 3-month flight.",
            "We're not sure how many people will see it \u2014 depends on our budget.",
        ],
        "clarification_hints": [
            "Think about how many people typically visit your website, use your app, "
            "or buy your product in a month. That's a good proxy for campaign reach.",
            "If you know your budget, that can help too \u2014 roughly what does your media "
            "buyer say the reach will be?",
            "Don't worry about being exact. 'Thousands' vs. 'tens of thousands' vs. "
            "'millions' is enough to help me recommend the right approach.",
        ],
        "extraction_prompt": (
            "From the user's reply, extract:\n"
            "- sample_size_estimate: one of ['small (<10k)', 'medium (10k-1M)', 'large (>1M)', 'unknown']\n"
            "- test_duration_weeks: integer weeks the campaign will run, or null if unknown\n"
            "Return JSON with those two keys.\n"
            "Convert months to weeks (\u00d7 4). If they say 'a few weeks' \u2192 ~3. "
            "'A month' \u2192 4. 'A quarter' \u2192 13."
        ),
        "followup_triggers": ["sample_size_estimate is 'unknown'"],
        "followup_question": (
            "No worries if you're not sure of the exact number! Even a rough range "
            "helps: do you think the campaign will reach closer to thousands of people, "
            "tens of thousands, hundreds of thousands, or millions? And is it more of "
            "a short burst (a week or two) or a longer effort (a month or more)?"
        ),
        "why_it_matters": (
            "The number of people and the duration together determine how much data "
            "we'll have to work with. More data means we can detect smaller effects "
            "and be more confident in the results."
        ),
    },
]

TOPIC_INDEX: dict[str, dict] = {t["topic"]: t for t in TOPIC_QUESTIONS}

EXTRACTION_SYSTEM = (
    "You are a data extraction assistant. Given a user's conversational reply, "
    "extract the requested fields and return ONLY valid JSON. "
    "Do not add explanation. If a field is genuinely unknown from the reply, use null or 'unknown'.\n\n"
    "The user is non-technical, so interpret their answers generously:\n"
    "- Informal descriptions like 'we sell in a bunch of cities' \u2192 try to infer a count or range\n"
    "- 'we have some data' / 'a few months' of history \u2192 has_historical_data=true, estimate weeks\n"
    "- 'the platform decides' / 'Google handles it' \u2192 platform_only, can_run_rct=false\n"
    "- 'we pick where it runs' / 'we control it' \u2192 brand_controlled, can_run_rct=true\n"
    "- Generic KPI descriptions like 'sales' \u2192 kpi='sales', primary_objective='conversion'\n"
    "- 'not many' / 'a handful' \u2192 small numbers (5-10); 'lots' / 'many' \u2192 larger (50+)\n"
    "If the user gives a vague but directional answer, extract the best reasonable value "
    "rather than defaulting to 'unknown'."
)

SCORING_EXPLANATION_PROMPT = """
You are an expert marketing measurement scientist. Below are six experimental methods and their
numeric scores (0\u2013100) based on a stakeholder's answers to elicitation questions.

Elicited facts:
{facts_json}

Scores:
{scores_json}

For each method, write 2\u20133 sentences in plain English explaining WHY it received that score
given this specific situation. Focus on the most important factors. Be honest about limitations.
Return a JSON object mapping each method key to its explanation string.
Method keys: ab_test, did, ddml, geo_lift, synthetic_control, matched_market
"""
