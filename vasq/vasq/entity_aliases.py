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


CELL_TYPE_ALIAS_GROUPS = {
    "Endothelial": ["endothelial", "endothelial cell", "endothelial cells"],
    "Capillary": [
        "capillary", "capillaries", "capillary cell", "capillary cells",
        "capillary endothelial cell", "capillary endothelial cells",
    ],
    "Arterial": [
        "arterial", "arteriole", "arterioles", "arterial endothelial cell",
        "arterial endothelial cells",
    ],
    "Artery": ["artery", "arteries", "arterial cell", "arterial cells"],
    "Arteriole": ["arteriole", "arterioles", "arteriolar"],
    "Venous": [
        "venous", "vein", "veins", "venule", "venules",
        "venous endothelial cell", "venous endothelial cells",
    ],
    "Vein": ["vein", "veins", "venous", "venous endothelial cells"],
    "Venule": ["venule", "venules", "venular"],
    "Fenestrated_Capillary": [
        "fenestrated capillary", "fenestrated capillaries",
        "fenestrated endothelial cell", "fenestrated endothelial cells",
        "fenestrated endothelium",
    ],
    "Fenestrated Endothelial": [
        "fenestrated endothelial", "fenestrated endothelial cell",
        "fenestrated endothelial cells", "fenestrated endothelium",
    ],
    "Large_Artery": ["large artery", "large arteries", "large arterial"],
    "Large Artery": ["large artery", "large arteries", "large arterial"],
    "EndoMT": ["endomt", "endothelial to mesenchymal transition cell"],
    "Pericyte": ["pericyte", "pericytes"],
    "Astrocyte": ["astrocyte", "astrocytes", "astroglia"],
    "OPC": [
        "opc", "opcs", "oligodendrocyte precursor",
        "oligodendrocyte precursors", "oligodendrocyte precursor cell",
        "oligodendrocyte precursor cells",
    ],
    "Oligodendrocyte Precursor": [
        "opc", "opcs", "oligodendrocyte precursor",
        "oligodendrocyte precursors", "oligodendrocyte precursor cell",
        "oligodendrocyte precursor cells",
    ],
    "Oligodendrocyte": ["oligodendrocyte", "oligodendrocytes"],
    "Neuron": ["neuron", "neurons", "neuronal cell", "neuronal cells"],
    "Fibroblast": ["fibroblast", "fibroblasts"],
    "Epithelial": ["epithelial", "epithelial cell", "epithelial cells"],
    "Epithelial_Cell": ["epithelial", "epithelial cell", "epithelial cells"],
    "Ependymal_Cell": ["ependymal", "ependymal cell", "ependymal cells"],
    "Smooth Muscle": [
        "smooth muscle", "smooth muscle cell", "smooth muscle cells", "smc",
    ],
    "Microglia Macrophage or T Cell": [
        "microglia", "microglial cell", "microglial cells", "macrophage",
        "macrophages", "t cell", "t cells",
    ],
    "Microglia_Macrophage_T": [
        "microglia", "microglial cell", "microglial cells", "macrophage",
        "macrophages", "t cell", "t cells",
    ],
}


CELL_CLASS_ALIAS_GROUPS = {
    "Endothelial": ["endothelial", "endothelial cell", "endothelial cells"],
    "Astrocyte": ["astrocyte", "astrocytes", "astroglia"],
    "Fibroblast": ["fibroblast", "fibroblasts"],
    "OPC": [
        "opc", "opcs", "oligodendrocyte precursor",
        "oligodendrocyte precursor cell", "oligodendrocyte precursor cells",
    ],
    "Oligodendrocyte": ["oligodendrocyte", "oligodendrocytes"],
    "Pericyte": ["pericyte", "pericytes"],
    "Neuron": ["neuron", "neurons", "neuronal"],
    "Epithelial": ["epithelial", "epithelial cell", "epithelial cells"],
    "Smooth Muscle": [
        "smooth muscle", "smooth muscle cell", "smooth muscle cells", "smc",
    ],
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
    "Parahippocampal Gyrus": ["parahippocampal cortex", "parahippocampal gyri", "phg"],
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


def build_cell_type_alias_map(available_values: Iterable[str]) -> dict[str, str]:
    return add_valid_alias_groups(
        build_canonical_label_map(available_values),
        available_values,
        CELL_TYPE_ALIAS_GROUPS,
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


_NEGATION_PREFIX = re.compile(
    r"(?:\bnot|\bwithout|\bexcept|\bexclude|\bexcluding|\bdo not include|"
    r"\bdon't include)\s+(?:the\s+)?$"
)


def _is_negated_mention(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40):start]
    return bool(_NEGATION_PREFIX.search(prefix))


def resolve_entities_from_text(
    user_input: str,
    alias_map: Mapping[str, str],
    *,
    fuzzy_threshold: float = 0.86,
) -> list[str]:
    """Resolve explicit, non-negated aliases with a limited typo fallback."""
    text = normalize_text(user_input)
    found: list[str] = []

    # Longest aliases first makes a specific phrase win before a nested term.
    for alias_norm, canonical in sorted(
        alias_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if not alias_norm:
            continue
        for match in re.finditer(rf"\b{re.escape(alias_norm)}\b", text):
            if not _is_negated_mention(text, match.start()):
                found.append(canonical)
                break

    # Restrict fuzzy matching to single-token aliases and only use it when no
    # exact/curated alias was found.  This avoids broad fuzzy overmatching.
    if not found:
        words = re.findall(r"\w+", text)
        for word in words:
            for alias_norm, canonical in alias_map.items():
                if len(alias_norm.split()) != 1:
                    continue
                if difflib.SequenceMatcher(None, alias_norm, word).ratio() > fuzzy_threshold:
                    found.append(canonical)

    return list(dict.fromkeys(found))
