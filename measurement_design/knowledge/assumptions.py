"""
Domain knowledge: method assumptions, key terms, and statistical prerequisites.

Each entry provides structured information about an experimental method's assumptions,
when they hold, when they're violated, how to check, and key terminology.
This is pure domain knowledge — no LLM or framework dependencies.
"""
from __future__ import annotations

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
