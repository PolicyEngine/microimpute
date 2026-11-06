"""Render the results table and note from ``results/results.json``.

Every number in ``TABLE.md`` and ``NOTE.md`` comes from the results file
written by ``run.py``. The note's prose lives in ``NOTE.template.md``
with ``{placeholders}``; rendering fails on any placeholder the results
cannot fill.

Usage (from the repository root):

    python -m experiments.moop_spm.render
"""

from __future__ import annotations

import json
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = [
    "logit_cell_mean",
    "logit_reg_mean",
    "logit_draw_hotdeck",
    "pmm",
    "qrf",
    "zi_qrf",
]
SHORT = {
    "reported": "Reported",
    "logit_cell_mean": "(a) Logit + cell mean",
    "logit_reg_mean": "(a) Logit + OLS mean",
    "logit_draw_hotdeck": "(b) Logit draw + donor",
    "pmm": "PMM",
    "qrf": "(c) QRF",
    "zi_qrf": "(c) ZI-QRF",
}
DECOMPOSITION = [
    "logit_cell_mean",
    "logit_threshold_hotdeck",
    "logit_draw_cell_mean",
    "logit_draw_hotdeck",
    "logit_expected_value",
    "logit_group_mean",
]
# Bands of resources before MOOP shown in Table 2 (index into RATIO_BANDS).
BAND_ROWS = [
    (1, "1.0 to 1.25"),
    (2, "1.25 to 1.5"),
    (3, "1.5 to 2.0"),
    (4, "2.0 to 3.0"),
]
DIMENSION_TITLES = {
    "all": None,
    "age": "Age",
    "disability": "Disability",
    "coverage": "Current health insurance",
    "pre_moop_ratio": "Resources before MOOP / threshold",
}


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def signed(x: float, digits: int = 2) -> str:
    value = 100 * x
    if round(value, digits) == 0:
        value = 0.0
    return f"{value:+.{digits}f}"


def dollars(x: float) -> str:
    return f"${x:,.0f}"


def main_table(results: dict) -> list[str]:
    methods = results["methods"]
    groups = results["groups"]
    counts = results["group_person_counts"]
    header = ["Group", "People (n)", SHORT["reported"]] + [SHORT[m] for m in MAIN]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    last_dim = None
    for i, (dim, label) in enumerate(groups):
        if dim != last_dim and DIMENSION_TITLES.get(dim):
            lines.append(f"| **{DIMENSION_TITLES[dim]}** |" + " |" * (len(header) - 1))
        last_dim = dim
        row = [label, f"{counts[i]:,}", pct(methods["reported"]["rate"][i])]
        for method in MAIN:
            entry = methods[method]
            row.append(f"{signed(entry['diff'][i])} ({100 * entry['diff_se'][i]:.2f})")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def distribution_table(results: dict) -> list[str]:
    methods = results["methods"]
    columns = ["reported"] + MAIN
    header = ["Statistic"] + [SHORT[m] for m in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(columns)) + "|",
    ]
    rows = [
        ("Mean", lambda d: dollars(d["mean"])),
        ("Median", lambda d: dollars(d["median"])),
        ("Share with zero MOOP", lambda d: f"{100 * d['share_zero']:.1f}%"),
        ("75th percentile", lambda d: dollars(d["p75"])),
        ("90th percentile", lambda d: dollars(d["p90"])),
        ("99th percentile", lambda d: dollars(d["p99"])),
        ("Total ($ billions)", lambda d: f"{d['aggregate_billions']:,.0f}"),
    ]
    for b, label in BAND_ROWS:
        rows.append(
            (
                f"Mean, resources before MOOP {label} x threshold",
                lambda d, b=b: dollars(d["band_means"][b]),
            )
        )
    for name, fmt in rows:
        lines.append(
            "| "
            + " | ".join([name] + [fmt(methods[m]["distribution"]) for m in columns])
            + " |"
        )
    flows = [
        ("People newly poor (%)", "newly_poor_share"),
        ("People newly not poor (%)", "newly_nonpoor_share"),
    ]
    for name, key in flows:
        lines.append(
            "| "
            + " | ".join([name, "0.0"] + [f"{100 * methods[m][key]:.2f}" for m in MAIN])
            + " |"
        )
    subset = [
        (
            "Change in rate among people in units where Census imputes no "
            "MOOP or Medicare premium (pp)"
        ),
        "0.00",
    ] + [
        f"{signed(methods[m]['no_allocation_subset']['diff'])} "
        f"({100 * methods[m]['no_allocation_subset']['diff_se']:.2f})"
        for m in MAIN
    ]
    lines.append("| " + " | ".join(subset) + " |")
    return lines


def decomposition_table(results: dict) -> list[str]:
    methods = results["methods"]
    lines = [
        "| Participation rule | Amount rule | Rate (%) | Change (pp) | SE |",
        "|---|---|---:|---:|---:|",
    ]
    rules = {
        "logit_cell_mean": ("p >= 0.5", "Cell mean"),
        "logit_threshold_hotdeck": ("p >= 0.5", "Random donor in cell"),
        "logit_draw_cell_mean": ("Bernoulli(p)", "Cell mean"),
        "logit_draw_hotdeck": ("Bernoulli(p)", "Random donor in cell"),
        "logit_expected_value": ("None (all units)", "p x cell mean"),
        "logit_group_mean": ("p >= 0.5", "Coverage-group mean (5 cells)"),
    }
    for method in DECOMPOSITION:
        entry = methods[method]
        participation, amount = rules[method]
        lines.append(
            f"| {participation} | {amount} | {pct(entry['rate'][0], 2)} | "
            f"{signed(entry['diff'][0])} | {100 * entry['diff_se'][0]:.2f} |"
        )
    return lines


def summary_table(results: dict) -> list[str]:
    """Compact table for the note: headline rates, key groups, shape."""
    methods = results["methods"]
    keyed = {"::".join(g): i for i, g in enumerate(results["groups"])}
    columns = ["reported"] + MAIN
    header = [""] + [SHORT[m] for m in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(columns)) + "|",
    ]

    def change(group: str, with_se: bool = False):
        i = keyed[group]

        def fmt(m: str) -> str:
            if m == "reported":
                return ""
            entry = methods[m]
            text = signed(entry["diff"][i])
            if with_se:
                text += f" ({100 * entry['diff_se'][i]:.2f})"
            return text

        return fmt

    def dist(key: str, fmt):
        return lambda m: fmt(methods[m]["distribution"][key])

    def flows(m: str) -> str:
        if m == "reported":
            return ""
        entry = methods[m]
        return (
            f"{100 * entry['newly_poor_share']:.2f} / "
            f"{100 * entry['newly_nonpoor_share']:.2f}"
        )

    rows = [
        (
            "SPM rate, all people (%)",
            lambda m: pct(methods[m]["rate"][keyed["all::All people"]], 2),
        ),
        ("Change, all people (pp, SE)", change("all::All people", with_se=True)),
        ("Change, 65 and older (pp)", change("age::65 and older")),
        ("Change, with a disability (pp)", change("disability::With a disability")),
        ("Change, Medicare (pp)", change("coverage::Medicare")),
        (
            "Change, resources before MOOP 1.0 to 1.25 x threshold (pp)",
            change("pre_moop_ratio::1.0 to 1.25"),
        ),
        (
            "Mean MOOP, resources before MOOP 1.0 to 1.25 x threshold",
            lambda m: dollars(methods[m]["distribution"]["band_means"][1]),
        ),
        ("Units with zero MOOP (%)", dist("share_zero", lambda x: f"{100 * x:.1f}")),
        ("Median MOOP", dist("median", dollars)),
        ("99th percentile MOOP", dist("p99", dollars)),
        ("Total MOOP ($ billions)", dist("aggregate_billions", lambda x: f"{x:,.0f}")),
        ("People newly poor / newly not poor (%)", flows),
    ]
    for name, fmt in rows:
        lines.append("| " + " | ".join([name] + [fmt(m) for m in columns]) + " |")
    return lines


def placeholder_values(results: dict) -> dict[str, str]:
    """Flat name-to-string map the note template can reference."""
    methods = results["methods"]
    groups = results["groups"]
    data = results["data"]
    published = results["published"]
    values: dict[str, str] = {
        "n_persons": f"{data['n_persons']:,}",
        "n_units": f"{data['n_units']:,}",
        "n_households": f"{data['n_households']:,}",
        "population_millions": f"{data['population_millions']:.1f}",
        "folds": str(results["config"]["folds"]),
        "draws": str(results["config"]["draws"]),
        "share_units_any_allocated": f"{100 * data['share_units_any_allocated']:.0f}",
        "share_persons_moop_allocated": f"{100 * data['share_persons_moop_allocated']:.0f}",
        "share_persons_premium_allocated": f"{100 * data['share_persons_premium_allocated']:.0f}",
        "share_persons_any_allocated": f"{100 * data['share_persons_any_allocated']:.0f}",
        "share_units_reported_zero": f"{100 * data['share_units_reported_zero']:.1f}",
        "published_rate": f"{published['spm_rate_percent']['All people']:.2f}",
        "published_moe90": f"{published['spm_rate_moe90']['All people']:.2f}",
        "published_moop_effect": f"{published['medical_expenses_effect_pp']['All people']:.2f}",
        "published_poor_thousands": f"{published['number_in_poverty_thousands']:,}",
    }
    keyed = {"::".join(g): i for i, g in enumerate(groups)}
    slug = {
        "all::All people": "all",
        "age::Under 18": "child",
        "age::18 to 64": "adult",
        "age::65 and older": "senior",
        "disability::With a disability": "disabled",
        "disability::No disability": "nondisabled",
        "coverage::Medicare": "medicare",
        "coverage::Employer": "employer",
        "coverage::Direct purchase": "direct",
        "coverage::Medicaid or other means-tested": "medicaid",
        "coverage::Uninsured": "uninsured",
        "pre_moop_ratio::1.0 to 1.25": "band1",
        "pre_moop_ratio::1.25 to 1.5": "band2",
        "pre_moop_ratio::1.5 to 2.0": "band3",
        "pre_moop_ratio::2.0 to 3.0": "band4",
        "pre_moop_ratio::3.0 or more": "band5",
    }
    for key, name in slug.items():
        i = keyed[key]
        for method, entry in methods.items():
            values[f"rate_{name}_{method}"] = pct(entry["rate"][i])
            values[f"rate2_{name}_{method}"] = pct(entry["rate"][i], 2)
            values[f"diff_{name}_{method}"] = signed(entry["diff"][i])
            values[f"diff1_{name}_{method}"] = signed(entry["diff"][i], 1)
            # Unsigned magnitudes for prose such as "raises the rate by X".
            values[f"pp_{name}_{method}"] = f"{100 * abs(entry['diff'][i]):.2f}"
            values[f"pp1_{name}_{method}"] = f"{100 * abs(entry['diff'][i]):.1f}"
            values[f"se_{name}_{method}"] = f"{100 * entry['diff_se'][i]:.2f}"
    rep = methods["reported"]
    values["reported_rate_se"] = f"{100 * rep['rate_se'][0]:.2f}"
    values["reported_moe90"] = f"{1.645 * 100 * rep['rate_se'][0]:.2f}"
    values["moop_effect"] = f"{-100 * methods['no_moop']['diff'][0]:.2f}"
    values["moop_effect_se"] = f"{100 * methods['no_moop']['diff_se'][0]:.2f}"
    values["reported_poor_thousands"] = (
        f"{rep['rate'][0] * data['population_millions'] * 1000:,.0f}"
    )
    values["summary_table"] = "\n".join(summary_table(results))
    reported_total = rep["distribution"]["aggregate_billions"]
    moop_effect = -methods["no_moop"]["diff"][0]
    for method, entry in methods.items():
        dist = entry["distribution"]
        values[f"share_of_effect_{method}"] = (
            f"{100 * entry['diff'][0] / moop_effect:.0f}"
        )
        values[f"agg_{method}"] = f"{dist['aggregate_billions']:,.0f}"
        values[f"agg_gap_{method}"] = (
            f"{100 * (1 - dist['aggregate_billions'] / reported_total):.1f}"
        )
        values[f"mean_{method}"] = dollars(dist["mean"])
        values[f"median_{method}"] = dollars(dist["median"])
        values[f"p99_{method}"] = dollars(dist["p99"])
        values[f"p90_{method}"] = dollars(dist["p90"])
        values[f"zero_{method}"] = f"{100 * dist['share_zero']:.1f}"
        values[f"nearline_{method}"] = dollars(dist["near_line_mean"])
        for b, _ in BAND_ROWS:
            values[f"bandmean{b}_{method}"] = dollars(dist["band_means"][b])
        if method != "reported":
            values[f"newly_poor_{method}"] = f"{100 * entry['newly_poor_share']:.2f}"
            values[f"newly_nonpoor_{method}"] = (
                f"{100 * entry['newly_nonpoor_share']:.2f}"
            )
            values[f"drawsd_{method}"] = f"{100 * entry['overall_draw_sd']:.3f}"
            sub = entry["no_allocation_subset"]
            values[f"subset_diff_{method}"] = signed(sub["diff"])
            values[f"subset_se_{method}"] = f"{100 * sub['diff_se']:.2f}"
    values["share_units_no_allocation"] = (
        f"{100 * (1 - data['share_units_any_allocated']):.0f}"
    )
    # The note says methods (a), (b) and PMM "raise" the subset's rate.
    ab_pmm = ["logit_cell_mean", "logit_reg_mean", "logit_draw_hotdeck", "pmm"]
    subset_ab_pmm = [methods[m]["no_allocation_subset"]["diff"] for m in ab_pmm]
    if min(subset_ab_pmm) <= 0:
        raise ValueError("A method (a), (b) or PMM no longer raises the subset rate.")
    values["subset_pp_low_ab_pmm"] = f"{100 * min(subset_ab_pmm):.2f}"
    values["subset_pp_high_ab_pmm"] = f"{100 * max(subset_ab_pmm):.2f}"
    return values


def main() -> None:
    results = json.loads((HERE / "results" / "results.json").read_text())
    data = results["data"]
    folds = results["config"]["folds"]
    source = (
        f"Source: {data['survey']} (calendar-year {data['reference_year']} "
        f"income), {data['n_persons']:,} people in {data['n_units']:,} SPM "
        f"units and {data['n_households']:,} households. We impute each "
        f"unit's MOOP from models we fit on the other {folds - 1} of {folds} "
        "folds of whole households and average "
        f"{results['config']['draws']} draws for each stochastic method. We "
        "weight rates by the ASEC person weight and compute the standard "
        "errors in parentheses from the 160 replicate weights, treating the "
        "fitted imputation models as fixed."
    )
    table_1 = (
        "## Table 1. SPM poverty rate with reported MOOP (%), and the change "
        "under each imputation (percentage points, SE)"
    )
    table_1_note = (
        "Every method imputes MOOP of zero or more, so people whose unit's "
        "resources before MOOP fall below its threshold stay poor, and the "
        "'Under 1.0' row cannot change."
    )
    table_2 = (
        "## Table 2. MOOP distribution across SPM units (unit weights) and "
        "gross flows of people (person weights)"
    )
    table_3 = (
        "## Table 3. How the participation and amount rules of the logit "
        "approach move the rate (all people)"
    )
    table = [
        "# MOOP imputation and SPM poverty: tables",
        "",
        source,
        "",
        table_1,
        "",
        *main_table(results),
        "",
        table_1_note,
        "",
        table_2,
        "",
        *distribution_table(results),
        "",
        table_3,
        "",
        *decomposition_table(results),
        "",
    ]
    (HERE / "TABLE.md").write_text("\n".join(table))

    template = (HERE / "NOTE.template.md").read_text()
    values = placeholder_values(results)
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    missing = sorted(fields - set(values))
    if missing:
        raise KeyError(f"Template placeholders without values: {missing}")
    (HERE / "NOTE.md").write_text(template.format(**values))
    print("wrote TABLE.md and NOTE.md")


if __name__ == "__main__":
    main()
