"""Reusable metrics for a gold-set evaluation of entity extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .entity_aliases import normalize_text


DEFAULT_DIMENSIONS = (
    "cell_types",
    "cell_classes",
    "regions",
    "region_layers",
)


def _normalized_set(values: Iterable[str] | None) -> set[str]:
    return {normalize_text(value) for value in (values or []) if normalize_text(value)}


def evaluate_entity_cases(
    cases: Iterable[Mapping],
    predictor: Callable[[str], Mapping[str, Iterable[str]]],
    *,
    vocabularies: Mapping[str, Iterable[str]] | None = None,
    dimensions: Iterable[str] = DEFAULT_DIMENSIONS,
) -> dict:
    """Score set-valued entity predictions against manually annotated cases.

    Returns per-dimension precision/recall/F1, whole-query exact-match accuracy,
    and the number of predictions outside the supplied controlled vocabularies.
    """
    dimensions = tuple(dimensions)
    counts = {
        dimension: {"tp": 0, "fp": 0, "fn": 0}
        for dimension in dimensions
    }
    exact_matches = 0
    invalid_predictions = 0
    total = 0

    vocabulary_sets = {
        dimension: _normalized_set(values)
        for dimension, values in (vocabularies or {}).items()
    }

    for case in cases:
        total += 1
        prediction = predictor(str(case["query"]))
        expected = case.get("expected", {})
        case_exact = True

        for dimension in dimensions:
            predicted_set = _normalized_set(prediction.get(dimension, []))
            expected_set = _normalized_set(expected.get(dimension, []))

            counts[dimension]["tp"] += len(predicted_set & expected_set)
            counts[dimension]["fp"] += len(predicted_set - expected_set)
            counts[dimension]["fn"] += len(expected_set - predicted_set)
            case_exact = case_exact and predicted_set == expected_set

            allowed = vocabulary_sets.get(dimension)
            if allowed is not None:
                invalid_predictions += len(predicted_set - allowed)

        exact_matches += int(case_exact)

    per_dimension = {}
    for dimension, dimension_counts in counts.items():
        tp = dimension_counts["tp"]
        fp = dimension_counts["fp"]
        fn = dimension_counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_dimension[dimension] = {
            **dimension_counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "n_cases": total,
        "query_exact_match_accuracy": exact_matches / total if total else 0.0,
        "invalid_prediction_count": invalid_predictions,
        "per_dimension": per_dimension,
    }
