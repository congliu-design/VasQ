"""Deterministic entity normalization for the VasQ controlled vocabulary.

This module intentionally contains no matrix loading and no OpenAI calls.  It
builds canonical/alias lookup tables, resolves explicit text mentions, and
validates both deterministic and model-produced labels against the dataset.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Iterable, Mapping

import pandas as pd


logger = logging.getLogger(__name__)


def normalize_text(value) -> str:
    """Normalize label formatting without attempting semantic matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def build_canonical_label_map(values: Iterable[str]) -> dict[str, str]:
    """Map normalized canonical labels back to their exact dataset labels."""
    alias_map: dict[str, str] = {}
    for value in pd.Series(list(values)).dropna().astype(str).unique():
        canonical = str(value).strip()
        normalized = normalize_text(canonical)
        if normalized:
            alias_map[normalized] = canonical
    return alias_map


# Backwards-compatible name for callers outside this module.  The new name is
# more accurate: this function covers canonical formatting variants, not true
# biological synonyms.
build_simple_alias_map = build_canonical_label_map


def add_valid_alias_groups(
    alias_map: dict[str, str],
    available_values: Iterable[str],
    alias_groups: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    """Add curated aliases only when their canonical target actually exists."""
    canonical_lookup = {
        normalize_text(value): str(value).strip()
        for value in available_values
        if normalize_text(value)
    }

    for requested_canonical, aliases in alias_groups.items():
        canonical = canonical_lookup.get(normalize_text(requested_canonical))
        if canonical is None:
            continue
        for alias in aliases:
            normalized = normalize_text(alias)
            if not normalized:
                continue
            existing = alias_map.get(normalized)
            if existing is not None and existing != canonical:
                logger.warning(
                    "Skipping ambiguous alias %r: already maps to %r, not %r",
                    alias,
                    existing,
                    canonical,
                )
                continue
            alias_map[normalized] = canonical
    return alias_map


def add_valid_multi_alias_groups(
    alias_map: dict[str, str | tuple[str, ...]],
    available_values: Iterable[str],
    alias_groups: Mapping[tuple[str, ...], Iterable[str]],
) -> dict[str, str | tuple[str, ...]]:
    """Add aliases that intentionally expand to multiple canonical labels."""
    canonical_lookup = {
        normalize_text(value): str(value).strip()
        for value in available_values
        if normalize_text(value)
    }

    for requested_targets, aliases in alias_groups.items():
        canonical_targets = tuple(
            canonical_lookup[normalize_text(target)]
            for target in requested_targets
            if normalize_text(target) in canonical_lookup
        )
        if not canonical_targets:
            continue

        for alias in aliases:
            normalized = normalize_text(alias)
            if not normalized:
                continue
            existing = alias_map.get(normalized)
            if existing is not None and existing != canonical_targets:
                logger.warning(
                    "Skipping ambiguous group alias %r: already maps to %r, not %r",
                    alias,
                    existing,
                    canonical_targets,
                )
                continue
            alias_map[normalized] = canonical_targets
    return alias_map


CELL_TYPE_ALIAS_GROUPS = {
    "Endothelial": ["endothelial", "endothelial cell", "endothelial cells"],
    "Capillary": [
        "capillary", "capillaries", "capillary cell", "capillary cells",
        "capillary endothelial cell", "capillary endothelial cells",
        "cap ec", "cap ecs", "capec", "capecs",
    ],
    "Arterial": [
        "arterial", "arteriole", "arterioles", "arteriolar",
        "arterial endothelial cell", "arterial endothelial cells",
        "arterial ec", "arterial ecs", "aec", "aecs",
    ],
    "Venous": [
        "venous", "vein", "veins", "venule", "venules", "venular",
        "venous endothelial cell", "venous endothelial cells",
        "venous ec", "venous ecs",
    ],
    "Fenestrated_Capillary": [
        "fenestrated capillary", "fenestrated capillaries",
        "fenestrated endothelial", "fenestrated endothelial cell",
        "fenestrated endothelial cells", "fenestrated endothelium",
        "fenestrated ec", "fenestrated ecs", "fenec", "fenecs",
    ],
    "Large_Artery": [
        "large artery", "large arteries", "large arterial",
        "large artery ec", "large artery ecs", "laec", "laecs",
    ],
    "EndoMT": [
        "endomt", "endomt cell", "endomt cells",
        "endothelial to mesenchymal transition cell",
        "endothelial to mesenchymal transition cells",
    ],
    "SMC_1": [
        "smc1", "smc 1", "smooth muscle cell 1", "smooth muscle subtype 1",
    ],
    "SMC_2": [
        "smc2", "smc 2", "smooth muscle cell 2", "smooth muscle subtype 2",
    ],
    "SMC_3": [
        "smc3", "smc 3", "smooth muscle cell 3", "smooth muscle subtype 3",
    ],
    "Pericyte": ["pericyte", "pericytes"],
    "Astrocyte": [
        "astrocyte", "astrocytes", "astroglia", "astro", "astros",
    ],
    "OPC": [
        "opc", "opcs", "oligodendrocyte precursor",
        "oligodendrocyte precursors", "oligodendrocyte precursor cell",
        "oligodendrocyte precursor cells",
    ],
    "Oligodendrocyte": [
        "oligodendrocyte", "oligodendrocytes", "oligo", "oligos",
    ],
    "Neuron": [
        "neuron", "neurons", "neuronal", "neuronal cell", "neuronal cells",
    ],
    "Fibroblast": ["fibroblast", "fibroblasts"],
    "Fib_1": ["fib1", "fib 1", "fibroblast 1", "fibroblast subtype 1"],
    "Fib_2": ["fib2", "fib 2", "fibroblast 2", "fibroblast subtype 2"],
    "Fib_3": ["fib3", "fib 3", "fibroblast 3", "fibroblast subtype 3"],
    "Fib_4": ["fib4", "fib 4", "fibroblast 4", "fibroblast subtype 4"],
    "Fib_5": ["fib5", "fib 5", "fibroblast 5", "fibroblast subtype 5"],
    "Fib_6": ["fib6", "fib 6", "fibroblast 6", "fibroblast subtype 6"],
    "Epithelial_Cell": ["epithelial", "epithelial cell", "epithelial cells"],
    "Ependymal_Cell": ["ependymal", "ependymal cell", "ependymal cells"],
    "Microglia_Macrophage_T": [
        "microglia", "microglial cell", "microglial cells", "macrophage",
        "macrophages", "t cell", "t cells",
    ],
}


# These aliases describe a biological group represented by several cell_type
# labels. They cannot be expressed by the ordinary one-alias/one-label map.
CELL_TYPE_MULTI_ALIAS_GROUPS = {
    ("SMC_1", "SMC_2", "SMC_3"): [
        "smc", "smcs", "vsmc", "vsmcs",
        "smooth muscle", "smooth muscle cell", "smooth muscle cells",
        "vascular smooth muscle", "vascular smooth muscle cell",
        "vascular smooth muscle cells",
    ],
    ("Arterial", "Large_Artery"): ["artery", "arteries"],
}


CELL_CLASS_ALIAS_GROUPS = {
    "Endothelial": [
        "endothelial", "endothelial cell", "endothelial cells",
        "ec", "ecs", "vascular ec", "vascular ecs",
        "vascular endothelial cell", "vascular endothelial cells",
        "brain ec", "brain ecs", "brain endothelial cell",
        "brain endothelial cells", "bec", "becs",
    ],
    "Mural_Cell": [
        "mural", "mural cell", "mural cells",
        "vascular mural cell", "vascular mural cells",
    ],
    "Astrocyte": [
        "astrocyte", "astrocytes", "astroglia", "astro", "astros",
    ],
    "Fibroblast": ["fibroblast", "fibroblasts", "fb", "fbs"],
    "OPC": [
        "opc", "opcs", "oligodendrocyte precursor",
        "oligodendrocyte precursor cell", "oligodendrocyte precursor cells",
    ],
    "Oligodendrocyte": [
        "oligodendrocyte", "oligodendrocytes", "oligo", "oligos",
    ],
    "Neuron": ["neuron", "neurons", "neuronal"],
    "Microglia_Macrophage_T": [
        "microglia macrophage t", "microglia", "microglial",
        "microglial cell", "microglial cells", "macrophage", "macrophages",
        "t cell", "t cells",
    ],
    "Epithelial_Cell": ["epithelial", "epithelial cell", "epithelial cells"],
    "Ependymal_Cell": ["ependymal", "ependymal cell", "ependymal cells"],
}


REGION_LAYER_ALIAS_GROUPS = {
    "Cortex": [
        "cortex", "cortical", "cortical cortex", "cortical gray matter",
        "cortical grey matter", "cortex gray matter", "cortex grey matter",
        "gray matter", "grey matter", "cerebral cortex", "neocortex",
    ],
    "White Matter Tracts": [
        "white matter", "white matter tract", "white matter tracts",
        "white matter tissue", "cerebral white matter", "deep white matter",
        "frontal white matter", "periventricular white matter", "wm",
    ],
    "Major Vessel": ["major vessel", "major vessels", "large vessel", "large vessels"],
    "Watershed": [
        "watershed", "watershed region", "watershed regions", "border zone",
        "borderzone",
    ],
    "Limbic": ["limbic", "limbic system", "limbic region"],
    "Brainstem": ["brainstem", "brain stem"],
    "Barrier": ["barrier", "barrier region"],
    "Olfactory": ["olfactory", "olfactory region", "olfactory system"],
    "Cerebellum": ["cerebellum", "cerebellar", "cerebellar region"],
}


BRAIN_REGION_ALIAS_GROUPS = {
    "Middle Cerebral Artery": ["middle cerebral arteries", "middle cerebral arterial", "mca"],
    "Anterior Cerebral Artery": ["anterior cerebral arteries", "anterior cerebral arterial", "aca"],
    "Basilar Artery/Circle Of Willis": [
        "basilar artery", "basilar arteries", "circle of willis",
        "basilar artery and circle of willis", "willis circle",
    ],
    "Lateral Temporal Gyrus": ["lateral temporal cortex", "lateral temporal gyri", "ltg"],
    "Insula": ["insular cortex", "insular region", "insular"],
    "Inferior Parietal Lobule": ["inferior parietal cortex", "inferior parietal lobules", "ipl"],
    "Midfrontal Anterior Watershed": [
        "mid frontal anterior watershed", "middle frontal anterior watershed",
        "midfrontal anterior border zone", "frontal anterior watershed",
    ],
    "Superior Parietal Lobule": ["superior parietal cortex", "superior parietal lobules", "spl"],
    "Cuneus": ["cuneal cortex", "cuneal region"],
    "Posterior Watershed": ["posterior watershed region", "posterior border zone", "posterior borderzone"],
    "Inferior Frontal Gyrus": ["inferior frontal cortex", "inferior frontal gyri", "ifg"],
    "White Matter Anterior Watershed": [
        "anterior watershed white matter", "white matter anterior border zone",
        "anterior white matter watershed",
    ],
    "Lateral Occipital Cortex": ["lateral occipital", "lateral occipital region", "loc"],
    "Dorsolateral Prefrontal Cortex": [
        "dorsolateral prefrontal", "dorsolateral pfc",
        "dorsal lateral prefrontal cortex", "dlpfc",
    ],
    "Inferior Temporal Gyrus": ["inferior temporal cortex", "inferior temporal gyri", "itg"],
    "Middle Temporal Gyrus": ["middle temporal cortex", "middle temporal gyri", "mtg"],
    "Midbrain": ["mid brain", "mesencephalon", "mesencephalic region"],
    "Orbitofrontal Cortex": [
        "orbitofrontal", "orbital frontal cortex", "orbital prefrontal cortex", "ofc",
    ],
    "Periventricular White Matter": ["periventricular wm", "periventricular white-matter", "pvwm"],
    "Cingulum": ["cingulum bundle", "cingulate bundle", "cingulate fasciculus"],
    "Lingual Gyrus": ["lingual cortex", "lingual gyri"],
    "Anterior Cingulate Cortex": ["anterior cingulate", "anterior cingulate region", "acc"],
    "Parahipocampal Gyrus": [
        "parahippocampal gyrus", "parahippocampal cortex", "parahippocampal gyri", "phg",
    ],
    "Posterior Cingulate Cortex": ["posterior cingulate", "posterior cingulate region", "pcc"],
    "Pons": ["pontine", "pontine region", "pons region"],
    "Hippocampus": ["hippocampal", "hippocampal formation", "hippocampal region"],
    "Superior Temporal Gyrus": ["superior temporal cortex", "superior temporal gyri", "stg"],
    "Choroid Plexus": ["choroidal plexus", "choroid plexuses", "choroid plexus tissue"],
    "Superior Frontal Gyrus And Rostromedial": [
        "superior frontal gyrus", "superior frontal gyri", "sfg",
        "rostromedial superior frontal gyrus", "superior frontal and rostromedial",
    ],
    "Precuneus": ["precuneal cortex", "precuneal region"],
    "Supramarginal Gyrus": ["supramarginal cortex", "supramarginal gyri", "smg"],
    "Entorhinal Cortex": ["entorhinal", "entorhinal region"],
    "Thalamus": ["thalamic", "thalamic region"],
    "Corpus Callosum": ["callosal", "callosal white matter", "corpus callosal"],
    "Amygdala": ["amygdalar", "amygdaloid", "amygdaloid complex"],
    "Fusiform Gyrus": ["fusiform cortex", "fusiform gyri"],
    "Leptomeninges": [
        "leptomeningeal", "leptomeningeal tissue", "pia arachnoid", "pia-arachnoid",
    ],
    "Olfactory Bulb": ["olfactory bulbs", "olfactory bulb region"],
    "Cerebellum": ["cerebellar", "cerebellar cortex", "cerebellar region"],
    "Spinal Cord": ["spinal cord tissue", "spinal region", "spinal-cord"],
    "Fornix": ["fornical", "fornical region", "fornix bundle"],
}


def build_cell_type_alias_map(
    available_values: Iterable[str],
) -> dict[str, str | tuple[str, ...]]:
    alias_map = add_valid_alias_groups(
        build_canonical_label_map(available_values),
        available_values,
        CELL_TYPE_ALIAS_GROUPS,
    )
    return add_valid_multi_alias_groups(
        alias_map,
        available_values,
        CELL_TYPE_MULTI_ALIAS_GROUPS,
    )


def build_cell_class_alias_map(available_values: Iterable[str]) -> dict[str, str]:
    return add_valid_alias_groups(
        build_canonical_label_map(available_values),
        available_values,
        CELL_CLASS_ALIAS_GROUPS,
    )


def build_region_layer_alias_map(available_values: Iterable[str]) -> dict[str, str]:
    return add_valid_alias_groups(
        build_canonical_label_map(available_values),
        available_values,
        REGION_LAYER_ALIAS_GROUPS,
    )


def build_brain_region_alias_map(available_values: Iterable[str]) -> dict[str, str]:
    return add_valid_alias_groups(
        build_canonical_label_map(available_values),
        available_values,
        BRAIN_REGION_ALIAS_GROUPS,
    )


def validate_controlled_vocabulary(
    candidates: Iterable[str] | None,
    available_values: Iterable[str],
    *,
    dimension_name: str,
) -> list[str]:
    """Return exact canonical labels and drop any out-of-vocabulary output."""
    lookup = {
        normalize_text(value): str(value).strip()
        for value in available_values
        if normalize_text(value)
    }
    matched: list[str] = []
    for candidate in candidates or []:
        canonical = lookup.get(normalize_text(candidate))
        if canonical is None:
            logger.warning(
                "Dropping unmatched %s candidate %r (not in controlled vocabulary)",
                dimension_name,
                candidate,
            )
            continue
        if canonical not in matched:
            matched.append(canonical)
    return matched


def merge_hybrid_matches(
    deterministic_matches: Iterable[str] | None,
    model_matches: Iterable[str] | None,
    available_values: Iterable[str],
    *,
    dimension_name: str,
) -> list[str]:
    """Merge deterministic and LLM matches, then enforce the vocabulary.

    Curated deterministic matches are ordered first.  A model failure (None)
    and a valid empty model result are both safe because deterministic matches
    remain available; generic all-values queries still produce an empty union.
    """
    combined = list(deterministic_matches or []) + list(model_matches or [])
    return validate_controlled_vocabulary(
        combined,
        available_values,
        dimension_name=dimension_name,
    )


# Ground-truth Cell_type -> Cell_class hierarchy from the VasQ matrix schema
# (see pdx[pdx['Cell_class'] == X]['Cell_type'].value_counts() for each class).
# This is schema metadata, not text-matching aliases: it lets us recognize
# that a resolved cell_type already pins down its parent cell_class, so an
# independently-matched class mention on the same query is redundant rather
# than a second, distinct filter dimension.
CELL_TYPE_TO_CELL_CLASS: dict[str, str] = {
    "Capillary": "Endothelial",
    "Venous": "Endothelial",
    "Arterial": "Endothelial",
    "Large_Artery": "Endothelial",
    "Fenestrated_Capillary": "Endothelial",
    "EndoMT": "Endothelial",
    "SMC_1": "Mural_Cell",
    "SMC_2": "Mural_Cell",
    "SMC_3": "Mural_Cell",
    "Pericyte": "Mural_Cell",
    "Fib_1": "Fibroblast",
    "Fib_2": "Fibroblast",
    "Fib_3": "Fibroblast",
    "Fib_4": "Fibroblast",
    "Fib_5": "Fibroblast",
    "Fib_6": "Fibroblast",
    "Astrocyte": "Astrocyte",
    "OPC": "OPC",
    "Oligodendrocyte": "Oligodendrocyte",
    "Neuron": "Neuron",
    "Microglia_Macrophage_T": "Microglia_Macrophage_T",
    "Epithelial_Cell": "Epithelial_Cell",
    "Ependymal_Cell": "Ependymal_Cell",
}


def exclude_classes_implied_by_cell_types(
    cell_classes: Iterable[str] | None,
    cell_types: Iterable[str] | None,
    *,
    type_to_class: Mapping[str, str] = CELL_TYPE_TO_CELL_CLASS,
) -> list[str]:
    """Drop a cell_class already implied by a more specific cell_type match.

    e.g. cell_types=["Capillary"] already pins the class to "Endothelial", so
    an independently-matched "Endothelial" alias (from wording like "capillary
    endothelial cells") is redundant, not a second filter the caller asked for.
    """
    implied = {
        normalize_text(type_to_class[value])
        for value in cell_types or []
        if value in type_to_class
    }
    return [
        value for value in cell_classes or []
        if normalize_text(value) not in implied
    ]


def exclude_normalized_duplicates(
    values: Iterable[str] | None,
    preferred_values: Iterable[str] | None,
) -> list[str]:
    """Drop identical labels duplicated in a more specific dimension."""
    preferred = {normalize_text(value) for value in preferred_values or []}
    return [
        value
        for value in values or []
        if normalize_text(value) not in preferred
    ]


def dimension_filter_is_disabled(user_input: str, dimension: str) -> bool:
    """Return True when the query explicitly disables a dimension filter."""
    text = normalize_text(user_input)
    dimension_names = {
        "cell_type": ["cell type"],
        "cell_class": ["cell class"],
        "brain_region": ["brain region", "region name", "region"],
        "region_layer": ["region layer"],
    }

    for name in dimension_names.get(dimension, []):
        escaped = re.escape(name)
        patterns = [
            rf"\bdo not apply (?:a |an )?{escaped} filter\b",
            rf"\bdon't apply (?:a |an )?{escaped} filter\b",
            rf"\bdo not filter (?:by|on) {escaped}\b",
            rf"\bdon't filter (?:by|on) {escaped}\b",
            rf"\bno {escaped} filter\b",
            rf"\bwithout (?:a |an )?{escaped} filter\b",
            rf"\bwithout filtering (?:by|on) {escaped}\b",
        ]
        if any(re.search(pattern, text) for pattern in patterns):
            return True
    return False


def apply_entity_selection_policy(
    user_input: str,
    *,
    cell_types: Iterable[str] | None = None,
    cell_classes: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    region_layers: Iterable[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Apply duplicate precedence and explicit no-filter instructions."""
    cell_types = list(dict.fromkeys(cell_types or []))
    cell_classes = exclude_normalized_duplicates(cell_classes, cell_types)
    cell_classes = exclude_classes_implied_by_cell_types(cell_classes, cell_types)
    regions = list(dict.fromkeys(regions or []))
    region_layers = list(dict.fromkeys(region_layers or []))

    normalized_query = normalize_text(user_input)
    if re.search(r"\bregion layers?\b|\bregion_layer\b", normalized_query):
        regions = exclude_normalized_duplicates(regions, region_layers)
    else:
        region_layers = exclude_normalized_duplicates(region_layers, regions)

    if dimension_filter_is_disabled(user_input, "cell_type"):
        cell_types = []
    if dimension_filter_is_disabled(user_input, "cell_class"):
        cell_classes = []
    if dimension_filter_is_disabled(user_input, "brain_region"):
        regions = []
    if dimension_filter_is_disabled(user_input, "region_layer"):
        region_layers = []

    return (
        list(dict.fromkeys(cell_types)),
        list(dict.fromkeys(cell_classes)),
        list(dict.fromkeys(regions)),
        list(dict.fromkeys(region_layers)),
    )


_NEGATION_PREFIX = re.compile(
    r"(?:\bnot|\bwithout|\bexcept|\bexclude|\bexcluding|\bdo not include|"
    r"\bdon't include)\s+(?:the\s+)?$"
)


def _is_negated_mention(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40):start]
    return bool(_NEGATION_PREFIX.search(prefix))


def resolve_entities_from_text(
    user_input: str,
    alias_map: Mapping[str, str | Iterable[str]],
    *,
    fuzzy_threshold: float = 0.86,
) -> list[str]:
    """Resolve explicit, non-negated aliases with a limited typo fallback."""
    text = normalize_text(user_input)
    matches: list[tuple[int, list[str]]] = []
    occupied_spans: list[tuple[int, int]] = []

    def target_values(target: str | Iterable[str]) -> list[str]:
        if isinstance(target, str):
            return [target]
        return list(target)

    def overlaps_existing(start: int, end: int) -> bool:
        return any(start < used_end and end > used_start for used_start, used_end in occupied_spans)

    # Longest aliases first so a specific phrase wins over nested aliases
    # (e.g. SMC2 must not also trigger the SMC group, and fenestrated
    # capillaries must not also return Capillary). This ordering only decides
    # *which* alias claims a span; matches are re-sorted by text position
    # below so the returned entities follow mention order in the query
    # regardless of which alias's phrase happened to be longer.
    for alias_norm, target in sorted(
        alias_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if not alias_norm:
            continue
        for match in re.finditer(rf"\b{re.escape(alias_norm)}\b", text):
            if _is_negated_mention(text, match.start()):
                continue
            if overlaps_existing(match.start(), match.end()):
                continue
            matches.append((match.start(), target_values(target)))
            occupied_spans.append((match.start(), match.end()))
            break

    found: list[str] = []
    for _, values in sorted(matches, key=lambda item: item[0]):
        found.extend(values)

    # Restrict fuzzy matching to single-token aliases and only use it when no
    # exact/curated alias was found.  This avoids broad fuzzy overmatching.
    if not found:
        words = re.findall(r"\w+", text)
        for word in words:
            for alias_norm, target in alias_map.items():
                if len(alias_norm.split()) != 1:
                    continue
                if difflib.SequenceMatcher(None, alias_norm, word).ratio() > fuzzy_threshold:
                    found.extend(target_values(target))

    return list(dict.fromkeys(found))
