from unittest import TestCase

from .entity_aliases import (
    add_valid_alias_groups,
    build_brain_region_alias_map,
    build_canonical_label_map,
    build_cell_type_alias_map,
    build_region_layer_alias_map,
    merge_hybrid_matches,
    resolve_entities_from_text,
    validate_controlled_vocabulary,
)
from .entity_evaluation import evaluate_entity_cases


class EntityAliasTests(TestCase):
    def test_canonical_map_handles_formatting_not_semantic_synonyms(self):
        aliases = build_canonical_label_map(["Fenestrated_Capillary"])
        self.assertEqual(aliases["fenestrated capillary"], "Fenestrated_Capillary")
        self.assertNotIn("fenestrated endothelial cells", aliases)

    def test_alias_target_must_exist_in_controlled_vocabulary(self):
        aliases = add_valid_alias_groups(
            {},
            ["Capillary"],
            {
                "Capillary": ["capillaries"],
                "Arterial": ["arterioles"],
            },
        )
        self.assertEqual(aliases, {"capillaries": "Capillary"})

    def test_canonical_label_wins_over_ambiguous_curated_alias(self):
        aliases = add_valid_alias_groups(
            build_canonical_label_map(["Venous", "Vein"]),
            ["Venous", "Vein"],
            {"Venous": ["vein"]},
        )
        self.assertEqual(aliases["vein"], "Vein")

    def test_cell_type_synonyms_resolve(self):
        aliases = build_cell_type_alias_map(["Capillary", "Arterial", "Pericyte"])
        result = resolve_entities_from_text(
            "Compare capillary endothelial cells with arterioles and pericytes.",
            aliases,
        )
        self.assertEqual(result, ["Capillary", "Arterial", "Pericyte"])

    def test_region_and_layer_abbreviations_resolve(self):
        region_aliases = build_brain_region_alias_map(
            ["Dorsolateral Prefrontal Cortex"]
        )
        layer_aliases = build_region_layer_alias_map(
            ["Cortex", "White Matter Tracts"]
        )
        self.assertEqual(
            resolve_entities_from_text("DLPFC", region_aliases),
            ["Dorsolateral Prefrontal Cortex"],
        )
        self.assertEqual(
            resolve_entities_from_text("cortex versus WM", layer_aliases),
            ["Cortex", "White Matter Tracts"],
        )

    def test_negated_alias_is_not_selected(self):
        aliases = build_cell_type_alias_map(["Capillary", "Arterial"])
        self.assertEqual(
            resolve_entities_from_text("Capillary, not arterial", aliases),
            ["Capillary"],
        )

    def test_generic_all_cell_types_does_not_create_specific_filter(self):
        aliases = build_cell_type_alias_map(["Capillary", "Arterial"])
        self.assertEqual(
            resolve_entities_from_text("Compare across all cell types", aliases),
            [],
        )

    def test_hybrid_merge_keeps_deterministic_match_when_model_omits_it(self):
        result = merge_hybrid_matches(
            ["Capillary"],
            [],
            ["Capillary", "Arterial"],
            dimension_name="cell_type",
        )
        self.assertEqual(result, ["Capillary"])

    def test_controlled_vocabulary_drops_hallucinated_label(self):
        result = validate_controlled_vocabulary(
            ["capillary", "Imaginary EC"],
            ["Capillary", "Arterial"],
            dimension_name="cell_type",
        )
        self.assertEqual(result, ["Capillary"])


class EntityEvaluationTests(TestCase):
    def test_evaluation_reports_exact_match_and_invalid_predictions(self):
        cases = [
            {
                "query": "capillaries in cortex",
                "expected": {
                    "cell_types": ["Capillary"],
                    "region_layers": ["Cortex"],
                },
            },
            {
                "query": "all cell types",
                "expected": {},
            },
        ]

        def predictor(query):
            if query == "capillaries in cortex":
                return {
                    "cell_types": ["Capillary"],
                    "cell_classes": [],
                    "regions": [],
                    "region_layers": ["Cortex"],
                }
            return {
                "cell_types": ["Imaginary EC"],
                "cell_classes": [],
                "regions": [],
                "region_layers": [],
            }

        metrics = evaluate_entity_cases(
            cases,
            predictor,
            vocabularies={
                "cell_types": ["Capillary", "Arterial"],
                "cell_classes": [],
                "regions": [],
                "region_layers": ["Cortex", "White Matter Tracts"],
            },
        )

        self.assertEqual(metrics["n_cases"], 2)
        self.assertEqual(metrics["query_exact_match_accuracy"], 0.5)
        self.assertEqual(metrics["invalid_prediction_count"], 1)
