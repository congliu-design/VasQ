"""Run a gold-set evaluation against VasQ's live hybrid entity resolver.

Usage from the ``vasq`` directory:

    python evaluate_entity_extraction.py \
        --cases entity_eval_cases.example.jsonl \
        --repeats 3

The live matrix files and OPENAI_API_KEY must be available because this calls
the same deterministic + LLM resolver used by the application.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vasq.entity_evaluation import evaluate_entity_cases


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    # Delay importing the live application until after argument parsing so
    # --help remains usable in lightweight development environments.
    from vasq import functions as vasq_functions

    cases = load_cases(args.cases)
    vasq_functions.ensure_matrix_expression_data_loaded()
    vocabularies = {
        "cell_types": vasq_functions.MATRIX_AVAILABLE_CELL_TYPES,
        "cell_classes": vasq_functions.MATRIX_AVAILABLE_CELL_CLASSES,
        "regions": vasq_functions.MATRIX_AVAILABLE_REGIONS,
        "region_layers": vasq_functions.MATRIX_AVAILABLE_REGION_LAYERS,
    }

    repeated_cases = cases * args.repeats
    outputs_by_query = {str(case["query"]): [] for case in cases}

    def live_predictor(query: str) -> dict[str, list[str]]:
        cell_types, cell_classes, regions, region_layers = (
            vasq_functions.resolve_matrix_entities(query)
        )
        prediction = {
            "cell_types": cell_types,
            "cell_classes": cell_classes,
            "regions": regions,
            "region_layers": region_layers,
        }
        outputs_by_query.setdefault(query, []).append(prediction)
        return prediction

    metrics = evaluate_entity_cases(
        repeated_cases,
        live_predictor,
        vocabularies=vocabularies,
    )
    metrics["unique_cases"] = len(cases)
    metrics["repeats"] = args.repeats
    stable_queries = sum(
        len({json.dumps(output, sort_keys=True) for output in outputs}) == 1
        for outputs in outputs_by_query.values()
        if outputs
    )
    metrics["repeat_stability"] = (
        stable_queries / len(outputs_by_query) if outputs_by_query else 0.0
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
