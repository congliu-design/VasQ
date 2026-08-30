"""Simple gold-set evaluation for vasq.functions.analyze_query_intent.

Run this file from the directory that contains manage.py:

    export OPENAI_API_KEY="your-key"
    python evaluate_query_intent.py

Outputs:
    query_intent_predictions.csv
    query_intent_metrics.csv
    query_intent_evaluation.png

Edit TEST_CASES below to add your own manually annotated questions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from exc

from vasq import functions as vasq_functions


OUTPUT_DIR = Path("query_intent_evaluation")
BOOLEAN_FIELDS = ["is_scientific", "asks_expression", "asks_drugs", "use_vasq"]
ENTITY_FIELDS = ["genes", "diseases"]


# Gold labels: change or extend these cases for your final evaluation.
TEST_CASES = [
    {
        "query": "Hello",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of CLDN5 in capillary endothelial cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Compare ABCB1 expression between cortex and white matter.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCB1"], "diseases": [],
        },
    },
    {
        "query": "Which brain vascular cell types express TFRC?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Which genes are most highly expressed in pericytes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is APOE associated with Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["APOE"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Which genes are associated with Parkinson's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "Which drugs target LRRK2 in Parkinson's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["LRRK2"], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "Does lecanemab treat Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "What pathways involve PSEN1 and APP?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["PSEN1", "APP"], "diseases": [],
        },
    },
    {
        "query": "How much SLC2A1 is expressed in liver hepatocytes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression in brain endothelial cells and kidney endothelial cells.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "What causes blood-brain barrier breakdown during aging?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of APOE in Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["APOE"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Are there treatments that increase ABCB1 expression at the BBB?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["ABCB1"], "diseases": [],
        },
    },
    {
        "query": "What about its expression in white matter?",
        "history": [
            {"role": "user", "content": "Tell me about CLDN5."},
            {"role": "assistant", "content": "CLDN5 is an endothelial tight-junction gene."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Are there any drugs for it?",
        "history": [
            {"role": "user", "content": "What genes are associated with Alzheimer's disease?"},
            {"role": "assistant", "content": "The discussion concerns Alzheimer's disease."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Can VasQ export a PDF?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    # ------------------------------------------------------------------
    # VasQ manuscript / queue questions supplied by the user
    # ------------------------------------------------------------------
    {
        "query": "What is the expression of glial cells in the hippocampus?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which genes are the highest-ranked markers for capillary cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "How do PLVAP, AQP1, and SLCO2A1 rank as markers for fenestrated capillary cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["PLVAP", "AQP1", "SLCO2A1"], "diseases": [],
        },
    },
    {
        "query": "What diseases, biological pathways, and drugs are directly connected to EGFR?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    {
        "query": "Using only the VasQ matrix, compare CLDN5 expression between the exact region_layer values Cortex and White Matter, this is a single-gene Cell-type comparison.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "What are the genes that are associated with Alzheimers disease? Which brain regions are they expressed in?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Do fenestrated capillary cells have specific transporter genes expressed?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is TFRC expressed across the entire brain or certain cell types and brain regions only?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Which transporters are expressed more in midbrain than in cortical regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What's the expression of CAV1 across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CAV1"], "diseases": [],
        },
    },
    {
        "query": "Which drugs target BBB or crossing of the BBB using transcytosis?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression in the memory-related region of the brain.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Is the expression of SLC7A1 the same across the frontal and parietal lobes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["SLC7A1"], "diseases": [],
        },
    },
    {
        "query": "How did endothelial transporter expression differ between gray matter and white matter?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which vascular cell types and which brain regions are enriched for small-vessel-disease susceptibility genes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["small vessel disease"],
        },
    },
    {
        "query": "What are the transporter/receptor genes with high regional specificity? What are their functions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of INSR in capillary endothelial cells across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["INSR"], "diseases": [],
        },
    },
    {
        "query": "What drugs failed because they cannot cross the human BBB due to no route into receptor-mediated transcytosis?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which vascular cell types and which brain regions are enriched for ischemic stroke susceptibility genes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["ischemic stroke"],
        },
    },
    {
        "query": "Compare ABCA7 expression in the memory-related region of the brain.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCA7"], "diseases": [],
        },
    },
]


def normalize_entity(value: str) -> str:
    """Normalize harmless spelling/punctuation variation before scoring."""
    value = str(value).casefold().replace("’", "'")
    value = value.replace("alzheimer disease", "alzheimer's disease")
    value = value.replace("parkinson disease", "parkinson's disease")
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    aliases = {
        "alzheimer disease": "alzheimer disease",
        "alzheimer s disease": "alzheimer disease",
        "alzheimers disease": "alzheimer disease",
        "parkinson disease": "parkinson disease",
        "parkinson s disease": "parkinson disease",
        "parkinsons disease": "parkinson disease",
        "cerebral small vessel disease": "small vessel disease",
    }
    return aliases.get(value, value)


def normalized_set(values) -> set[str]:
    return {normalize_entity(value) for value in (values or []) if str(value).strip()}


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Count helper failures that analyze_query_intent catches internally and
    # routes to fallback_query_intent.
    original_helper = vasq_functions.call_helper_api
    helper_stats = {"calls": 0, "errors": 0}

    def tracked_helper(*args, **kwargs):
        helper_stats["calls"] += 1
        try:
            return original_helper(*args, **kwargs)
        except Exception:
            helper_stats["errors"] += 1
            raise

    vasq_functions.call_helper_api = tracked_helper

    rows = []
    for case_number, case in enumerate(TEST_CASES, start=1):
        calls_before = helper_stats["calls"]
        errors_before = helper_stats["errors"]
        prediction = vasq_functions.analyze_query_intent(
            case["query"],
            history=case.get("history"),
        )
        expected = case["expected"]

        row = {
            "case": case_number,
            "query": case["query"],
            "helper_called": helper_stats["calls"] > calls_before,
            "used_fallback": helper_stats["errors"] > errors_before,
            "resolved_question": prediction.get("resolved_question", ""),
        }

        boolean_matches = []
        for field in BOOLEAN_FIELDS:
            expected_value = bool(expected[field])
            predicted_value = bool(prediction.get(field, False))
            row[f"expected_{field}"] = expected_value
            row[f"predicted_{field}"] = predicted_value
            row[f"correct_{field}"] = expected_value == predicted_value
            boolean_matches.append(row[f"correct_{field}"])

        entity_matches = []
        for field in ENTITY_FIELDS:
            expected_values = normalized_set(expected.get(field, []))
            predicted_values = normalized_set(prediction.get(field, []))
            row[f"expected_{field}"] = json.dumps(expected.get(field, []), ensure_ascii=False)
            row[f"predicted_{field}"] = json.dumps(prediction.get(field, []), ensure_ascii=False)
            row[f"correct_{field}"] = expected_values == predicted_values
            entity_matches.append(row[f"correct_{field}"])

        row["routing_exact"] = all(boolean_matches)
        row["entities_exact"] = all(entity_matches)
        row["full_exact"] = row["routing_exact"] and row["entities_exact"]
        rows.append(row)

        print(
            f"[{case_number:02d}/{len(TEST_CASES)}] "
            f"{'PASS' if row['full_exact'] else 'FAIL'} - {case['query']}"
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUTPUT_DIR / "query_intent_predictions.csv", index=False)

    metrics = []
    for field in BOOLEAN_FIELDS:
        metrics.append({
            "metric": field,
            "value": predictions[f"correct_{field}"].mean(),
            "type": "accuracy",
        })

    for field in ENTITY_FIELDS:
        tp = fp = fn = 0
        for case, row in zip(TEST_CASES, rows):
            expected_values = normalized_set(case["expected"].get(field, []))
            predicted_values = normalized_set(json.loads(row[f"predicted_{field}"]))
            tp += len(expected_values & predicted_values)
            fp += len(predicted_values - expected_values)
            fn += len(expected_values - predicted_values)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        metrics.extend([
            {"metric": f"{field}_exact", "value": predictions[f"correct_{field}"].mean(), "type": "exact match"},
            {"metric": f"{field}_precision", "value": precision, "type": "micro"},
            {"metric": f"{field}_recall", "value": recall, "type": "micro"},
            {"metric": f"{field}_f1", "value": f1, "type": "micro"},
        ])

    metrics.extend([
        {"metric": "routing_exact", "value": predictions["routing_exact"].mean(), "type": "exact match"},
        {"metric": "entities_exact", "value": predictions["entities_exact"].mean(), "type": "exact match"},
        {"metric": "full_exact", "value": predictions["full_exact"].mean(), "type": "exact match"},
        {"metric": "fallback_rate", "value": predictions["used_fallback"].mean(), "type": "rate"},
    ])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "query_intent_metrics.csv", index=False)

    plot_metrics = metrics_df[
        metrics_df["metric"].isin(
            BOOLEAN_FIELDS
            + ["genes_exact", "diseases_exact", "routing_exact", "entities_exact", "full_exact"]
        )
    ].copy()

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = ["#8E3B70" if value < 0.9 else "#4C956C" for value in plot_metrics["value"]]
    axes[0].barh(plot_metrics["metric"], plot_metrics["value"], color=colors)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Accuracy / exact-match rate")
    axes[0].set_title("analyze_query_intent performance")
    axes[0].grid(axis="x", alpha=0.2)
    for position, value in enumerate(plot_metrics["value"]):
        axes[0].text(min(value + 0.02, 1.01), position, f"{value:.2f}", va="center", fontsize=9)

    outcome_counts = predictions["full_exact"].value_counts().reindex([True, False], fill_value=0)
    axes[1].bar(["Fully correct", "Mismatch"], outcome_counts.values, color=["#4C956C", "#C85C5C"])
    axes[1].set_ylabel("Number of test cases")
    axes[1].set_title(f"Overall results (n={len(predictions)})")
    axes[1].grid(axis="y", alpha=0.2)
    for position, value in enumerate(outcome_counts.values):
        axes[1].text(position, value + 0.15, str(value), ha="center", fontweight="bold")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "query_intent_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("\nSummary")
    print(metrics_df.to_string(index=False, formatters={"value": "{:.3f}".format}))
    print(f"\nFallbacks: {int(predictions['used_fallback'].sum())}/{len(predictions)}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")

    mismatches = predictions.loc[
        ~predictions["full_exact"],
        ["case", "query"]
        + [f"expected_{field}" for field in BOOLEAN_FIELDS + ENTITY_FIELDS]
        + [f"predicted_{field}" for field in BOOLEAN_FIELDS + ENTITY_FIELDS],
    ]
    if not mismatches.empty:
        print("\nMismatched cases:")
        print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
