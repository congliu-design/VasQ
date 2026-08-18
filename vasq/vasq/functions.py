import difflib
import contextvars
import json
import openai
import os
import pandas as pd
import re
import requests
import time
import difflib
import ast
import logging
import numpy as np
from scipy import sparse

# Set gloabl variables
global func_flag
global init_flag
func_flag = False
init_flag = True
from openai import OpenAI


logger = logging.getLogger(__name__)
logger.info("OpenAI SDK version: %s", openai.__version__)


class TurnBudgetExceeded(TimeoutError):
    """Raised when there is not enough time left to start another stage."""


_TURN_DEADLINE = contextvars.ContextVar("vasq_turn_deadline", default=None)


def _env_float(name, default, minimum=1.0):
    """Read a positive float setting without allowing bad env values to crash."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using default=%s", name, default)
        value = float(default)
    return max(float(minimum), value)


def _env_int(name, default, minimum=0, maximum=5):
    """Read and bound small integer settings such as retry counts."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using default=%s", name, default)
        value = int(default)
    return max(minimum, min(maximum, value))


def _stage_timeout(stage_name, requested_seconds, reserve_seconds=5.0):
    """Cap a network call by both its stage limit and the current turn budget."""
    requested_seconds = max(1.0, float(requested_seconds))
    deadline = _TURN_DEADLINE.get()
    if deadline is None:
        return requested_seconds

    remaining = deadline - time.monotonic()
    usable = remaining - max(0.0, float(reserve_seconds))
    if usable < 1.0:
        raise TurnBudgetExceeded(
            f"Skipping {stage_name}: only {remaining:.1f}s remains in turn budget"
        )

    effective = min(requested_seconds, usable)
    logger.info(
        "Stage %s timeout=%.1fs turn_remaining=%.1fs reserve=%.1fs",
        stage_name,
        effective,
        remaining,
        reserve_seconds,
    )
    return effective

#def query_kg_rag(user_input):
#    url = os.getenv("KG_RAG_URL", "http://kg-rag.railway.internal:8080/query")
#    logger.info("Calling KG_RAG_URL=%s", url)
#    response = requests.post(url, json={"query": user_input}, timeout=120)
#    logger.info("kg-rag status=%s body=%s", response.status_code, response.text[:500])
#    response.raise_for_status()
#    return response.json()


# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Disable the SDK's long default retry chain at the shared-client level.
# Individual calls opt into a small, bounded retry count below.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=0,
)



### Helper Functions ###

# Call OpenAI API
def call_api(
    history,
    functions=None,
    *,
    stage_name="chat_completion",
    timeout_seconds=None,
    reserve_seconds=5.0,
):
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")

    request_args = {
        "model": model,
        "messages": history,
    }

    if functions:
        request_args["functions"] = functions
        
        if model.startswith("gpt-5.6"):
            request_args["reasoning_effort"] = "none"
            
    timeout_seconds = timeout_seconds or _env_float(
        "OPENAI_CHAT_TIMEOUT_SECONDS", 45
    )
    timeout_seconds = _stage_timeout(
        stage_name,
        timeout_seconds,
        reserve_seconds=reserve_seconds,
    )
    max_retries = _env_int("OPENAI_CHAT_MAX_RETRIES", 1)
    # SDK timeouts apply to an individual attempt. Divide the stage allowance
    # across attempts so one retry cannot double the wall-clock stage limit.
    attempt_timeout = max(1.0, timeout_seconds / (max_retries + 1))
    chat_co = client.with_options(
        timeout=attempt_timeout,
        max_retries=max_retries,
    ).chat.completions.create(**request_args)

    return chat_co.choices[0].message


def call_helper_api(
    system_prompt,
    user_prompt,
    *,
    stage_name="helper_completion",
):
    """Call the helper model with parameters compatible with GPT-4o and GPT-5.6."""
    model = os.getenv("OPENAI_HELPER_MODEL", "gpt-4o")

    request_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # GPT-5.6 only accepts its default sampling settings. Older models such as
    # GPT-4o can still use temperature=0 for deterministic helper tasks.
    if not model.startswith("gpt-5.6"):
        request_args["temperature"] = 0

    timeout_seconds = _stage_timeout(
        stage_name,
        _env_float("OPENAI_HELPER_TIMEOUT_SECONDS", 30),
        # Preserve enough time for the final answer even if an optional helper
        # is reached late in the request.
        reserve_seconds=_env_float("VASQ_SYNTHESIS_RESERVE_SECONDS", 50),
    )
    max_retries = _env_int("OPENAI_HELPER_MAX_RETRIES", 1)
    attempt_timeout = max(1.0, timeout_seconds / (max_retries + 1))
    return client.with_options(
        timeout=attempt_timeout,
        max_retries=max_retries,
    ).chat.completions.create(**request_args)


logger = logging.getLogger(__name__)


def run_openai_web_search(
    search_prompt,
    *,
    stage_name="web_search",
    search_context_size="high",
):
    try:
        timeout_seconds = _stage_timeout(
            stage_name,
            _env_float("OPENAI_WEB_TIMEOUT_SECONDS", 75),
            reserve_seconds=_env_float("VASQ_SYNTHESIS_RESERVE_SECONDS", 50),
        )
        logger.info(
            "Calling OpenAI Web Search stage=%s context=%s timeout=%.1fs",
            stage_name,
            search_context_size,
            timeout_seconds,
        )

        response = client.with_options(
            timeout=timeout_seconds,
            # A Web Search retry can repeat a large and expensive tool call.
            # Default to no retry; it can be enabled explicitly if desired.
            max_retries=_env_int("OPENAI_WEB_MAX_RETRIES", 0),
        ).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"),
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": search_context_size,
                    "external_web_access": True,
                }
            ],

            # å› ä¸ºåªæœ‰ web_search ä¸€ä¸ªå·¥å…·ï¼Œæ‰€ä»¥ required ä¿è¯ä¸€å®šæœç´¢
            tool_choice="required",

            include=["web_search_call.action.sources"],

            input=search_prompt,
        )

        result = response.output_text.strip()

        if not result:
            logger.warning("OpenAI Web Search returned no text")
            return None

        logger.info(
            "OpenAI Web Search succeeded stage=%s result_length=%s request_id=%s",
            stage_name,
            len(result),
            getattr(response, "_request_id", None),
        )
        return result

    except TurnBudgetExceeded as exc:
        logger.warning("OpenAI Web Search skipped: %s", exc)
        return None
    except openai.APITimeoutError:
        logger.warning(
            "OpenAI Web Search timed out stage=%s; continuing with partial evidence",
            stage_name,
            exc_info=True,
        )
        return None
    except (openai.APIConnectionError, openai.RateLimitError):
        logger.warning(
            "OpenAI Web Search unavailable stage=%s; continuing with partial evidence",
            stage_name,
            exc_info=True,
        )
        return None
    except Exception:
        logger.exception("OpenAI Web Search failed stage=%s", stage_name)
        return None


def search_openai_web(user_input):
    """Generic biomedical Web Search kept for backwards compatibility."""
    return run_openai_web_search(
        "Search the live web before answering the following biomedical "
        "question. Prioritize peer-reviewed literature, PubMed, FDA, "
        "ClinicalTrials.gov, and authoritative medical sources. "
        "Provide source citations.\n\n"
        f"Question: {user_input}",
        stage_name="generic_biomedical_web_search",
    )


def search_scientific_web(user_input, kg_context=None, kg_assessment=None):
    """First search: establish scientific knowledge and candidate genes."""
    kg_context = (kg_context or "").strip()
    kg_assessment = kg_assessment or {}

    if kg_context and kg_assessment.get("relevant"):
        context_block = (
            "A biomedical knowledge graph returned the following potentially "
            "relevant context. Use it only to guide the search; independently "
            "verify every scientific claim:\n"
            f"{kg_context[:4000]}\n\n"
        )
    else:
        context_block = (
            "The biomedical knowledge graph did not return sufficiently "
            "relevant context. Search from the original question directly.\n\n"
        )

    pathway_instruction = (
        "Pay particular attention to molecular function, biological pathways, "
        "mechanism, disease relevance, and human genes supported by the "
        "evidence. When the question concerns a disease or asks which genes "
        "are involved, provide a prioritized list of official human gene "
        "symbols and distinguish causal genes from risk-associated or "
        "mechanistic genes. If pathway, function, or gene-association evidence "
        "is not available, say so instead of inferring it. "
    )

    return run_openai_web_search(
        "Search the live web to answer this biomedical question. "
        + pathway_instruction
        + "Prioritize peer-reviewed literature, PubMed, FDA, "
        "ClinicalTrials.gov, and authoritative scientific sources. "
        "Provide source citations and distinguish established evidence from "
        "hypotheses.\n\n"
        + context_block
        + f"Question: {user_input}",
        stage_name="scientific_web_search",
        search_context_size=os.getenv(
            "OPENAI_SCIENTIFIC_SEARCH_CONTEXT_SIZE", "high"
        ),
    )
    
def search_gene_fallback(user_input):
    """Focused fallback when the main scientific search returns no genes."""

    return run_openai_web_search(
        (
            "Identify up to 10 well-established human genes associated with "
            "the disease or biological condition in the question below. "
            "Prioritize causal genes and strongly supported risk genes. "
            "Use official human gene symbols. Provide concise supporting "
            "evidence and citations from authoritative genetics resources "
            "or peer-reviewed literature. Do not invent associations.\n\n"
            f"Question: {user_input}"
        ),
        stage_name="gene_fallback_web_search",
        search_context_size="low",
    )

def search_drugs_and_small_molecules(user_input, genes=None, diseases=None):
    """Search current drug/small-molecule evidence for genes or diseases."""
    genes = [str(x).upper().strip() for x in (genes or []) if str(x).strip()]
    diseases = [str(x).strip() for x in (diseases or []) if str(x).strip()]

    if not genes and not diseases:
        return None

    entity_lines = []
    if genes:
        entity_lines.append("Genes/targets: " + ", ".join(genes))
    if diseases:
        entity_lines.append("Diseases/conditions: " + ", ".join(diseases))

    return run_openai_web_search(
        "Search the live web for drugs and small molecules related to the "
        "entities below. Search for direct gene/protein modulators, "
        "pathway-related compounds, and disease-directed treatments. Clearly "
        "separate these relationship types. For each credible candidate, "
        "report mechanism, indication, modality, and development stage "
        "(approved, clinical, preclinical, or research tool). Do not describe "
        "a disease treatment as directly targeting a gene unless the evidence "
        "supports that relationship. Distinguish small molecules from "
        "antibodies, nucleic-acid therapies, and other modalities. Prioritize "
        "FDA/EMA labels, ClinicalTrials.gov, PubMed, peer-reviewed literature, "
        "and authoritative company trial records. Use current information, "
        "provide source citations, and explicitly state when no reliable "
        "direct small-molecule match is found.\n\n"
        + "\n".join(entity_lines)
        + f"\n\nOriginal scientific question: {user_input}",
        stage_name="drug_web_search",
        # Drug searches can fan out across many entities. Medium context keeps
        # the optional branch bounded; override via env when high is required.
        search_context_size=os.getenv("OPENAI_DRUG_SEARCH_CONTEXT_SIZE", "medium"),
    )


# Update chat history
def update_history(history, role, content):
    message = {"role": role, "content": content}
    history.append(message)

# Initialize chat
def initialize(history):
    global init_flag
    system_prompt = (
        "You are a neuroscience and biomedical research assistant. Maintain "
        "context across follow-up questions and answer at a professional "
        "scientific level. For ordinary conversation such as greetings, "
        "respond naturally without claiming that external research was used. "
        "For scientific answers, distinguish four evidence types: biomedical "
        "knowledge-graph relationships, current web/literature evidence, "
        "VasQ single-nucleus brain-vasculature expression measurements, and "
        "drug/small-molecule evidence. Never present marker rank as absolute "
        "expression. Never describe a disease treatment as directly targeting "
        "a gene unless the supplied evidence supports that relationship. "
        "Distinguish approved drugs, clinical candidates, preclinical "
        "compounds, and research tools. Preserve citations from retrieved web "
        "evidence, state important limitations, and do not mention internal "
        "routing or tool implementation."
    )

    update_history(history, "system", system_prompt)
    init_flag = False

# Call function from chat

def func_call(user_input, chat_message, history):
    if wants_web_search(user_input):
        logger.info("func_call override: explicit web/literature intent -> Google")
        return search_openai_web(user_input)

    global func_flag
    func_flag = True
    content = None

    func_name = chat_message.function_call.name

    if (
        func_name == "marker_gene_expression"
        and wants_matrix_expression_query(user_input)
        and not wants_marker_query(user_input)
    ):
        logger.info(
            "Overriding model-selected marker_gene_expression -> matrix_expression"
        )
        func_name = "matrix_expression"

    print("Calling", func_name, "...")
    args = {"user_input": user_input}
    content = globals()[func_name](**args)

    func_flag = False
    return content


### Gene Expression Functions ###
DATA_DIR = "/data"
EXPR_PATH = os.path.join(DATA_DIR, "expression_markers.csv")
REGION_META_PATH = os.path.join(DATA_DIR, "region_metadata.csv")
MATRIX_NPZ_PATH = os.path.join(DATA_DIR, "VasQ_adata_X_sparse.npz")
CELL_META_PATH = os.path.join(DATA_DIR, "VasQ_cell_meta_table.csv")
GENE_NAMES_PATH = os.path.join(DATA_DIR, "VasQ_gene_names.csv")

MATRIX_EXPR = None
MATRIX_META = None
MATRIX_GENES = None
MATRIX_GENE_TO_IDX = None

MATRIX_AVAILABLE_CELL_TYPES = None
MATRIX_AVAILABLE_CELL_CLASSES = None
MATRIX_AVAILABLE_REGIONS = None
MATRIX_AVAILABLE_REGION_LAYERS = None

MATRIX_CELL_TYPE_ALIAS_MAP = None
MATRIX_CELL_CLASS_ALIAS_MAP = None
MATRIX_REGION_ALIAS_MAP = None
MATRIX_REGION_LAYER_ALIAS_MAP = None

# Do not expose expression summaries for groups with fewer than 10 cells.
# Keeping this threshold at the shared summarization layer ensures that the
# text table and every plot enforce the same rule.
MIN_CELLS_PER_GROUP = 10


def build_simple_alias_map(values):
    alias_map = {}
    for v in pd.Series(values).dropna().astype(str).unique():
        alias_map[normalize_text(v)] = str(v)
    return alias_map


def load_gene_names():
    genes_df = pd.read_csv(GENE_NAMES_PATH)

    genes = (
        genes_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .to_numpy()
    )
    return genes


def load_matrix_expression_data():
    npz = np.load(MATRIX_NPZ_PATH, allow_pickle=True)

    if {"data", "indices", "indptr", "shape"}.issubset(npz.files):
        X = sparse.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=tuple(npz["shape"])
        )
    elif "X" in npz.files:
        X = sparse.csr_matrix(npz["X"])
    else:
        raise ValueError(
            f"{MATRIX_NPZ_PATH} must contain either CSR arrays "
            f"(data, indices, indptr, shape) or a dense X array."
        )

    genes = np.array([str(g).upper().strip() for g in load_gene_names()], dtype=object)

    meta = pd.read_csv(CELL_META_PATH)
    meta = meta.drop(
        columns=[
            c for c in meta.columns
            if str(c).startswith("Unnamed:")
        ],
        errors="ignore",
    )
    
    meta.index = meta.index.astype(str)
    meta.index.name = "cell_id"
    meta["cell_id"] = meta.index
    meta = meta.reset_index(drop=True)

    rename_map = {
        "region_name": "brain_region",
        "region_layer": "region_layer",
        "Cell_class": "cell_class",
        "Cell_type": "cell_type",
        "ageatdeath": "age_at_death",
        "sex": "sex",
    }
    meta = meta.rename(columns=rename_map)

    required_meta = [
        "brain_region",
        "region_layer",
        "cell_class",
        "cell_type",
        "age_at_death",
        "sex",
    ]
    missing_meta = [c for c in required_meta if c not in meta.columns]
    if missing_meta:
        raise ValueError(f"{CELL_META_PATH} missing required columns: {missing_meta}")

    meta["brain_region"] = meta["brain_region"].astype(str).str.strip()
    meta["region_layer"] = meta["region_layer"].astype(str).str.strip()
    meta["cell_class"] = meta["cell_class"].astype(str).str.strip()
    meta["cell_type"] = meta["cell_type"].astype(str).str.strip()
    meta["age_at_death"] = pd.to_numeric(meta["age_at_death"], errors="coerce")
    meta["sex"] = meta["sex"].astype(str).str.strip()

    meta["brain_region_norm"] = meta["brain_region"].apply(normalize_text)
    meta["region_layer_norm"] = meta["region_layer"].apply(normalize_text)
    meta["cell_class_norm"] = meta["cell_class"].apply(normalize_text)
    meta["cell_type_norm"] = meta["cell_type"].apply(normalize_text)
    meta["sex_norm"] = meta["sex"].apply(normalize_text)

    if X.shape[0] != len(meta):
        raise ValueError(
            f"Matrix rows ({X.shape[0]}) do not match metadata rows ({len(meta)})"
        )

    if X.shape[1] != len(genes):
        raise ValueError(
            f"Matrix columns ({X.shape[1]}) do not match genes ({len(genes)})"
        )

    return X, meta, genes

def get_matrix_cell_indices(
    user_input,
    cell_types=None,
    cell_classes=None,
    regions=None,
    region_layers=None,
):
    ensure_matrix_expression_data_loaded()

    mask = pd.Series(True, index=MATRIX_META.index)

    if cell_types:
        mask &= MATRIX_META["cell_type"].isin(cell_types)

    if cell_classes:
        mask &= MATRIX_META["cell_class"].isin(cell_classes)

    if regions:
        mask &= MATRIX_META["brain_region"].isin(regions)

    if region_layers:
        mask &= MATRIX_META["region_layer"].isin(region_layers)

    sex_filters = extract_sex_filters(user_input)
    if sex_filters:
        mask &= MATRIX_META["sex_norm"].isin(sex_filters)

    return np.flatnonzero(mask.to_numpy())


def ensure_matrix_expression_data_loaded():
    global MATRIX_EXPR, MATRIX_META, MATRIX_GENES, MATRIX_GENE_TO_IDX
    global MATRIX_AVAILABLE_CELL_TYPES, MATRIX_AVAILABLE_CELL_CLASSES
    global MATRIX_AVAILABLE_REGIONS, MATRIX_AVAILABLE_REGION_LAYERS
    global MATRIX_CELL_TYPE_ALIAS_MAP, MATRIX_CELL_CLASS_ALIAS_MAP
    global MATRIX_REGION_ALIAS_MAP, MATRIX_REGION_LAYER_ALIAS_MAP

    if MATRIX_EXPR is None or MATRIX_META is None or MATRIX_GENES is None:
        MATRIX_EXPR, MATRIX_META, MATRIX_GENES = load_matrix_expression_data()
        MATRIX_GENE_TO_IDX = {g: i for i, g in enumerate(MATRIX_GENES)}

    if MATRIX_AVAILABLE_CELL_TYPES is None:
        MATRIX_AVAILABLE_CELL_TYPES = sorted(
            MATRIX_META["cell_type"].dropna().astype(str).unique().tolist()
        )

    if MATRIX_AVAILABLE_CELL_CLASSES is None:
        MATRIX_AVAILABLE_CELL_CLASSES = sorted(
            MATRIX_META["cell_class"].dropna().astype(str).unique().tolist()
        )

    if MATRIX_AVAILABLE_REGIONS is None:
        MATRIX_AVAILABLE_REGIONS = sorted(
            MATRIX_META["brain_region"].dropna().astype(str).unique().tolist()
        )

    if MATRIX_AVAILABLE_REGION_LAYERS is None:
        MATRIX_AVAILABLE_REGION_LAYERS = sorted(
            MATRIX_META["region_layer"].dropna().astype(str).unique().tolist()
        )

    if MATRIX_CELL_TYPE_ALIAS_MAP is None:
        MATRIX_CELL_TYPE_ALIAS_MAP = build_simple_alias_map(MATRIX_AVAILABLE_CELL_TYPES)
        MATRIX_CELL_TYPE_ALIAS_MAP.update({
            "capillary": "Capillary",
            "capillaries": "Capillary",
            "arterial": "Arterial",
            "arteriole": "Arterial",
            "arterioles": "Arterial",
            "venous": "Venous" if "Venous" in MATRIX_AVAILABLE_CELL_TYPES else "Vein",
            "opc": "OPC",
            "astrocyte": "Astrocyte",
            "astrocytes": "Astrocyte",
        })

    if MATRIX_CELL_CLASS_ALIAS_MAP is None:
        MATRIX_CELL_CLASS_ALIAS_MAP = build_simple_alias_map(MATRIX_AVAILABLE_CELL_CLASSES)
        MATRIX_CELL_CLASS_ALIAS_MAP.update({
            "endothelial": "Endothelial",
            "endothelial cells": "Endothelial",
            "astrocyte": "Astrocyte",
            "astrocytes": "Astrocyte",
            "fibroblast": "Fibroblast",
            "fibroblasts": "Fibroblast",
            "opc": "OPC",
        })

    if MATRIX_REGION_ALIAS_MAP is None:
        MATRIX_REGION_ALIAS_MAP = build_simple_alias_map(MATRIX_AVAILABLE_REGIONS)

    if MATRIX_REGION_LAYER_ALIAS_MAP is None:
        MATRIX_REGION_LAYER_ALIAS_MAP = build_simple_alias_map(MATRIX_AVAILABLE_REGION_LAYERS)


def dimension_filter_is_disabled(user_input, dimension):
    """Detect explicit instructions not to filter a metadata dimension."""
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


def resolve_matrix_entities(user_input):
    ensure_matrix_expression_data_loaded()

    local_cell_type_matches = resolve_entities_from_text(
        user_input, MATRIX_CELL_TYPE_ALIAS_MAP
    )
    local_cell_class_matches = resolve_entities_from_text(
        user_input, MATRIX_CELL_CLASS_ALIAS_MAP
    )
    local_region_matches = resolve_entities_from_text(
        user_input, MATRIX_REGION_ALIAS_MAP
    )
    local_region_layer_matches = resolve_entities_from_text(
        user_input, MATRIX_REGION_LAYER_ALIAS_MAP
    )

    (
        gpt_cell_type_matches,
        gpt_cell_class_matches,
        gpt_region_matches,
        gpt_region_layer_matches,
    ) = resolve_dataset_entities_with_gpt(
        user_input,
        MATRIX_AVAILABLE_CELL_TYPES,
        MATRIX_AVAILABLE_REGIONS,
        available_cell_classes=MATRIX_AVAILABLE_CELL_CLASSES,
        available_region_layers=MATRIX_AVAILABLE_REGION_LAYERS,
    )

    # A valid GPT result, including a valid empty list, takes priority over
    # substring matching. This prevents labels mentioned in negated clauses or
    # as examples (for example "report DLPFC when present, but do not filter by
    # region_name") from silently becoming filters. Local matching is used
    # only if the helper request failed and returned None for the dimension.
    cell_type_matches = (
        gpt_cell_type_matches
        if gpt_cell_type_matches is not None
        else local_cell_type_matches
    )
    cell_class_matches = (
        gpt_cell_class_matches
        if gpt_cell_class_matches is not None
        else local_cell_class_matches
    )
    region_matches = (
        gpt_region_matches
        if gpt_region_matches is not None
        else local_region_matches
    )
    region_layer_matches = (
        gpt_region_layer_matches
        if gpt_region_layer_matches is not None
        else local_region_layer_matches
    )

    if dimension_filter_is_disabled(user_input, "cell_type"):
        cell_type_matches = []
    if dimension_filter_is_disabled(user_input, "cell_class"):
        cell_class_matches = []
    if dimension_filter_is_disabled(user_input, "brain_region"):
        region_matches = []
    if dimension_filter_is_disabled(user_input, "region_layer"):
        region_layer_matches = []

    cell_type_matches = list(dict.fromkeys(cell_type_matches or []))
    cell_class_matches = list(dict.fromkeys(cell_class_matches or []))
    region_matches = list(dict.fromkeys(region_matches or []))
    region_layer_matches = list(dict.fromkeys(region_layer_matches or []))

    logger.info(
        "Matrix filters resolved cell_types=%s cell_classes=%s "
        "brain_regions=%s region_layers=%s",
        cell_type_matches,
        cell_class_matches,
        region_matches,
        region_layer_matches,
    )

    return cell_type_matches, cell_class_matches, region_matches, region_layer_matches

def extract_sex_filters(user_input):
    text = normalize_text(user_input)
    out = []

    if re.search(r"\bfemale\b|\bfemales\b|\bwoman\b|\bwomen\b", text):
        out.append("f")
    if re.search(r"\bmale\b|\bmales\b|\bman\b|\bmen\b", text):
        out.append("m")

    return out


def requested_matrix_group_columns(
    user_input,
    *,
    cell_types=None,
    cell_classes=None,
    regions=None,
    region_layers=None,
):
    """Choose dimensions that must remain separate in an expression result.

    Explicitly resolved values always preserve their dimension. Generic
    requests such as "compare cell types" or "across region layers" also
    preserve that dimension even when the user did not enumerate labels.
    """
    text = normalize_text(user_input)
    group_cols = []

    asks_regions = bool(regions) or bool(
        re.search(
            r"\bbrain regions?\b|\bregions?\b|\bregion names?\b|"
            r"\bregion name\b|\bregion_name\b",
            text,
        )
    )
    asks_region_layers = bool(region_layers) or bool(
        re.search(
            r"\bregion layers?\b|\bregion_layer\b|\bcortical layers?\b|"
            r"\blayers?\b|\bl[1-6](?:\s*/\s*l?[1-6])?\b",
            text,
        )
    )
    asks_cell_classes = bool(cell_classes) or bool(
        re.search(r"\bcell classes?\b|\bcell_class\b", text)
    )
    asks_cell_types = bool(cell_types) or bool(
        re.search(r"\bcell types?\b|\bcell_type\b", text)
    )

    if asks_regions:
        group_cols.append("brain_region")
    if asks_region_layers:
        group_cols.append("region_layer")
    if asks_cell_classes:
        group_cols.append("cell_class")
    if asks_cell_types:
        group_cols.append("cell_type")

    # Preserve the previous useful default for an expression question that
    # does not explicitly name a comparison dimension.
    if not group_cols:
        group_cols = ["brain_region", "cell_type"]

    return group_cols


def matrix_expression(user_input, genes_override=None):
    ensure_matrix_expression_data_loaded()

    if genes_override:
        genes = [
            str(g).upper().strip()
            for g in genes_override
            if str(g).strip()
        ]
        genes = list(dict.fromkeys(genes))
    else:
        genes = extract_genes(user_input)

    if not genes:
        return "Please specify a gene for matrix-based expression queries."

    cell_types, cell_classes, regions, region_layers = resolve_matrix_entities(user_input)
    group_cols = requested_matrix_group_columns(
        user_input,
        cell_types=cell_types,
        cell_classes=cell_classes,
        regions=regions,
        region_layers=region_layers,
    )

    present_genes = [g for g in genes if g in MATRIX_GENE_TO_IDX]
    missing_genes = [g for g in genes if g not in MATRIX_GENE_TO_IDX]

    notes = [
        "This answer uses log-normalized values from the HVG-filtered expression matrix.",
        (
            "Applied matrix filters — "
            f"Brain region: {', '.join(regions) if regions else 'ALL'}; "
            f"Region layer: {', '.join(region_layers) if region_layers else 'ALL'}; "
            f"Cell class: {', '.join(cell_classes) if cell_classes else 'ALL'}; "
            f"Cell type: {', '.join(cell_types) if cell_types else 'ALL'}."
        ),
    ]

    if missing_genes:
        notes.append(
            "These genes are not present in the supplied HVG-filtered matrix and may have been filtered out during HVG selection: "
            + ", ".join(missing_genes)
        )

    cell_indices = get_matrix_cell_indices(
        user_input,
        cell_types=cell_types,
        cell_classes=cell_classes,
        regions=regions,
        region_layers=region_layers,
    )

    if len(cell_indices) == 0 and present_genes:
        return "No matching cells found for the requested filters."

    if 0 < len(cell_indices) < MIN_CELLS_PER_GROUP and present_genes:
        return (
            "No expression data are displayed because the requested subset "
            f"contains fewer than {MIN_CELLS_PER_GROUP} cells."
        )

    all_sections = []
    regional_plot_frames = []

    if len(cell_indices) > 0:
        effective_cell_indices = cell_indices
    else:
        effective_cell_indices = MATRIX_META.index.to_numpy()

    for gene in present_genes:
        # Keep every requested comparison dimension separate. This prevents,
        # for example, Layer 2 and Layer 3 from being pooled into one mean.
        stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            group_cols,
            min_cells=MIN_CELLS_PER_GROUP,
        )

        all_sections.append(
            format_matrix_expression_summary(
                stats,
                gene,
                group_cols=group_cols,
                max_rows=40,
            )
        )

        # Use the same grouping and cell threshold in the visual so its points
        # correspond exactly to the rows that are eligible for the table.
        comparison_stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            group_cols,
            min_cells=MIN_CELLS_PER_GROUP,
        )
        if not comparison_stats.empty:
            regional_plot_frames.append(comparison_stats)

    plot_json = None
    if regional_plot_frames:
        # Cell-type comparisons need a purpose-built plot table. Keep the
        # requested filters, but always retain brain region on the x-axis and
        # cell type on the y-axis/panel dimension. Do not force the text table
        # to use this additional grouping.
        plot_group_cols = list(group_cols)
        if "cell_type" in group_cols and (
            len(present_genes) == 1 or len(present_genes) > 2
        ):
            plot_group_cols = ["brain_region"]
            if "region_layer" in group_cols:
                plot_group_cols.append("region_layer")
            plot_group_cols.append("cell_type")

            plot_frames = []
            for gene in present_genes:
                gene_plot_stats = summarize_group_expression(
                    gene,
                    effective_cell_indices,
                    plot_group_cols,
                    min_cells=MIN_CELLS_PER_GROUP,
                )
                if not gene_plot_stats.empty:
                    plot_frames.append(gene_plot_stats)
            plot_stats = (
                pd.concat(plot_frames, ignore_index=True)
                if plot_frames
                else pd.DataFrame()
            )
        else:
            plot_stats = pd.concat(regional_plot_frames, ignore_index=True)

        plot_json = build_matrix_expression_plot(
            plot_stats,
            gene_order=present_genes,
            comparison_cols=plot_group_cols,
        )

    return {
        "text": "\n\n".join(notes + [""] + all_sections),
        "graph_json": plot_json
    }



def get_gene_vector(cell_indices, gene):
    gene_idx = MATRIX_GENE_TO_IDX[gene]
    values = MATRIX_EXPR[cell_indices, gene_idx]

    if sparse.issparse(values):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()

    return values


def summarize_group_expression(
    gene,
    cell_indices,
    group_cols,
    min_cells=MIN_CELLS_PER_GROUP,
):
    if len(cell_indices) == 0:
        return pd.DataFrame()

    obs = MATRIX_META.iloc[cell_indices][group_cols].copy()
    rows = []

    groupby_arg = group_cols[0] if len(group_cols) == 1 else group_cols

    for key, g in obs.groupby(groupby_arg):
        idx = g.index.to_numpy()

        # Suppress small groups before reading or calculating their expression
        # values. No row or plot point is created for an n < 10 group.
        if len(idx) < min_cells:
            continue

        vals = get_gene_vector(idx, gene)

        row = {
            "gene": gene,
            "n_cells": int(len(idx)),
            "mean_expr": float(vals.mean()) if len(vals) else 0.0,
            "pct_expr": float((vals > 0).mean()) if len(vals) else 0.0,
        }

        if not isinstance(key, tuple):
            key = (key,)

        row.update(dict(zip(group_cols, key)))
        rows.append(row)

    return pd.DataFrame(rows)


def select_balanced_expression_rows(
    stats_df,
    group_cols,
    max_rows=40,
):
    """Keep requested comparison values represented in a capped table.

    A global top-N by mean expression can accidentally retain only Cortex and
    hide White Matter Tracts. Select at least one strong row per observed value
    of each comparison dimension before filling remaining slots by rank.
    """
    ranked = stats_df.sort_values(
        ["mean_expr", "pct_expr", "n_cells"],
        ascending=[False, False, False],
    )
    if max_rows is None or len(ranked) <= max_rows:
        return ranked

    priority_cols = [
        col
        for col in [
            "region_layer",
            "brain_region",
            "cell_type",
            "cell_class",
        ]
        if col in group_cols and col in ranked.columns
    ]
    chosen = []
    chosen_set = set()

    for col in priority_cols:
        for value in ranked[col].drop_duplicates().tolist():
            candidates = ranked[ranked[col] == value]
            for idx in candidates.index:
                if idx not in chosen_set:
                    chosen.append(idx)
                    chosen_set.add(idx)
                    break
            if len(chosen) >= max_rows:
                return ranked.loc[chosen]

    for idx in ranked.index:
        if idx not in chosen_set:
            chosen.append(idx)
            chosen_set.add(idx)
        if len(chosen) >= max_rows:
            break

    return ranked.loc[chosen]


def format_comparison_coverage(stats_df, group_cols, max_values=30):
    """Describe all observed metadata values before detailed rows are capped."""
    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_class": "Cell class",
        "cell_type": "Cell type",
    }
    lines = [
        "Observed comparison values after filtering and the minimum-cell rule:"
    ]
    for col in group_cols:
        if col not in stats_df.columns:
            continue
        values = sorted(
            stats_df[col].dropna().astype(str).drop_duplicates().tolist()
        )
        shown = values[:max_values]
        suffix = (
            f"; plus {len(values) - max_values} additional values"
            if len(values) > max_values
            else ""
        )
        lines.append(
            f"- {display_names.get(col, col)}: "
            + ", ".join(shown)
            + suffix
        )
    return lines


def format_matrix_expression_summary(
    stats_df,
    gene,
    group_cols=None,
    max_rows=40,
):
    if stats_df.empty:
        return (
            f"No data are displayed for {gene}: no matching comparison "
            f"group contains at least {MIN_CELLS_PER_GROUP} cells."
        )

    group_cols = [
        col
        for col in (group_cols or ["brain_region", "cell_type"])
        if col in stats_df.columns
    ]

    work = select_balanced_expression_rows(
        stats_df,
        group_cols,
        max_rows=max_rows,
    )

    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_class": "Cell class",
        "cell_type": "Cell type",
    }
    dimension_headers = [display_names.get(col, col) for col in group_cols]
    header = (
        "| "
        + " | ".join(
            dimension_headers
            + [
                "Mean expression (log-normalized)",
                "Expressing cells",
                "Cells analyzed (n)",
            ]
        )
        + " |"
    )
    alignment = (
        "| "
        + " | ".join(
            ["---"] * len(dimension_headers)
            + ["---:", "---:", "---:"]
        )
        + " |"
    )

    lines = [
        (
            f"Measured comparison groups for {gene}, ranked by average "
            "log-normalized expression (groups with fewer than "
            f"{MIN_CELLS_PER_GROUP} cells are not displayed):"
        ),
        "",
    ]
    lines.extend(format_comparison_coverage(stats_df, group_cols))
    lines.extend([
        "",
        header,
        alignment,
    ])

    for _, row in work.iterrows():
        dimension_values = [
            str(row.get(col, "All matched values")).replace("|", "/")
            for col in group_cols
        ]
        lines.append(
            "| "
            + " | ".join(dimension_values)
            + " | "
            f"{row['mean_expr']:.3f} | "
            f"{100.0 * row['pct_expr']:.1f}% | "
            f"{int(row['n_cells'])} |"
        )

    lines.extend([
        "",
        (
            "Mean expression is averaged across all cells in the group, "
            "including zero values. Expressing cells is the percentage with "
            "nonzero expression; Cells analyzed (n) is the total group size."
        ),
    ])

    return "\n".join(lines)


def wants_marker_query(user_input):
    text = user_input.lower()
    triggers = [
        "marker", "markers", "top marker", "top markers",
        "rank", "ranked", "top genes", "enriched"
    ]
    return any(t in text for t in triggers)


def wants_matrix_expression_query(user_input):
    text = user_input.lower()
    triggers = [
        "expression","express", "expressed", "mean expression", "average expression",
        "most highly expressed", "highest expressed", "how much expression",
        "percent expressing", "pct expr", "lowly expressed", "absent", "not expressed"
    ]
    return any(t in text for t in triggers)




### ranked expression

def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def load_expression_data():
    with open(EXPR_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    print("EXPR_PATH:", EXPR_PATH)
    print("FIRST LINE:", first_line)

    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{EXPR_PATH} is a Git LFS pointer, not the real CSV file."
        )

    df = pd.read_csv(EXPR_PATH)

    print("Expression columns:", df.columns.tolist())
    print(df.head(3).to_string())

    rename_map = {
        "tissue": "region",
        "Region": "region",
        "CellType": "cell_type",
        "Gene": "gene"
    }
    df = df.rename(columns=rename_map)

    required = {"gene", "cell_type", "region", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expression table missing required columns: {missing}")

    # Marker rows without a gene, context, or numeric rank cannot be used.
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["rank"] = df["rank"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["gene", "cell_type", "region", "rank"]).copy()

    # Optional statistics must be numeric before sorting or JSON encoding.
    for column in ["score", "logFC", "pct_expr"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)

    df["gene"] = df["gene"].astype(str).str.upper().str.strip()
    df["cell_type"] = df["cell_type"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["region"] = df["region"].str.replace(r"^\d+_", "", regex=True)

    df["cell_type_norm"] = df["cell_type"].apply(normalize_text)
    df["region_norm"] = df["region"].apply(normalize_text)

    return df


def load_region_metadata():
    meta = pd.read_csv(REGION_META_PATH)

    # standardize likely column names
    rename_map = {
        "Final_abb": "final_abb",
        "region_abb": "region_abb",
        "Region_layer_1": "region_layer_1",
        "Region_layer_2": "region_layer_2",
        "Region_layer_3": "region_layer_3",
        "Region_layer_4": "region_layer_4",
        "Region_layer_5": "region_layer_5",
        "Origin": "origin",
        "Other notes": "other_notes"
    }
    meta = meta.rename(columns=rename_map)

    return meta


def build_region_alias_map(meta):
    alias_map = {}

    for _, row in meta.iterrows():
        aliases = set()

        for col in [
            "final_abb",
            "region_abb",
            "region_layer_1",
            "region_layer_2",
            "region_layer_3",
            "region_layer_4",
            "region_layer_5",
            "origin",
            "other_notes"
        ]:
            if col in meta.columns and pd.notna(row.get(col, None)):
                value = str(row[col]).strip()
                if value and value.lower() != "none" and value.upper() != "NA":
                    aliases.add(value)

        canonical = str(row.get("region_abb", "")).strip()
        if canonical:
            for alias in aliases:
                alias_map[normalize_text(alias)] = canonical

        # also allow final abbreviation itself to resolve to region_abb
        final_abb = str(row.get("final_abb", "")).strip()
        if final_abb and canonical:
            alias_map[normalize_text(final_abb)] = canonical

    # manual helpful aliases
    manual = {
        "hippocampus": "Hip-EC",
        "hippocampal": "Hip-EC",
        "hip": "Hip-EC",
        "pons": "Pons",
        "amygdala": "Amygdala",
        "thalamus": "Thalamus",
        "midbrain": "Midbrain",
        "cerebellum": "CB",
        "entorhinal cortex": "EC",
        "entorhinal ctx": "EC",
        "choroid plexus": "CP",
        "leptomeninges": "Leptomeninges",
        "olfactory bulb": "OB",
        "spinal cord": "Spinal-cord",
        "anterior cerebral artery": "ACA",
        "middle cerebral artery": "MCA",
        "basilar artery": "BA.CoW",
        "circle of willis": "BA.CoW",
        "basilar artery circle of willis": "BA.CoW",
        "corpus callosum": "CC",
        "fornix": "Fornix",
        "cingulum": "Cingulum",
        "periventricular white matter": "PVWM",
    }

    alias_map.update({normalize_text(k): v for k, v in manual.items()})
    return alias_map


def build_cell_type_alias_map(df):
    canonical_cell_types = sorted(df["cell_type"].dropna().unique().tolist())
    alias_map = {}

    for ct in canonical_cell_types:
        alias_map[normalize_text(ct)] = ct

    manual = {
        "arterial": "Arterial",
        "arteriole": "Arteriole",
        "artery": "Artery",
        "capillary": "Capillary",
        "endothelial": "Endothelial",
        "fenestrated endothelial": "Fenestrated Endothelial",
        "fenestrated endothelium": "Fenestrated Endothelial",
        "pericyte": "Pericyte",
        "smooth muscle": "Smooth Muscle",
        "venous": "Vein",
        "vein": "Vein",
        "venule": "Venule",
        "astrocyte": "Astrocyte",
        "neuron": "Neuron",
        "fibroblast": "Fibroblast",
        "epithelial": "Epithelial",
        "oligodendrocyte": "Oligodendrocyte",
        "oligodendrocyte precursor": "Oligodendrocyte Precursor",
        "opc": "Oligodendrocyte Precursor",
        "microglia": "Microglia Macrophage or T Cell",
        "macrophage": "Microglia Macrophage or T Cell",
        "t cell": "Microglia Macrophage or T Cell",
        "microglia macrophage or t cell": "Microglia Macrophage or T Cell",
        "large artery": "Large Artery",
    }

    alias_map.update({normalize_text(k): v for k, v in manual.items()})
    return alias_map


def select_marker_rows(
    df,
    *,
    gene_order=None,
    max_rows=18,
    max_per_gene=4,
    unique_genes=False,
):
    """Select a balanced, deterministic subset of precomputed marker rows."""
    if df.empty:
        return df.copy()

    work = df.copy()
    work["rank"] = pd.to_numeric(work["rank"], errors="coerce")
    work["rank"] = work["rank"].replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["gene", "cell_type", "region", "rank"])
    work = work.drop_duplicates(
        subset=["gene", "cell_type", "region"],
        keep="first",
    )

    if work.empty:
        return work

    gene_order = [
        str(g).upper().strip()
        for g in (gene_order or [])
        if str(g).strip()
    ]

    # For explicit genes, retain several contexts per gene so one strong gene
    # cannot crowd every other requested gene out of the chart.
    if gene_order:
        frames = []
        per_gene = max(
            1,
            min(max_per_gene, max_rows // max(1, len(gene_order))),
        )
        for gene in gene_order:
            group = work[work["gene"] == gene].copy()
            if group.empty:
                continue
            if "score" in group.columns and group["score"].notna().any():
                group = group.sort_values(
                    ["score", "rank"],
                    ascending=[False, True],
                    na_position="last",
                )
            else:
                group = group.sort_values("rank", ascending=True)
            frames.append(group.head(per_gene))

        if not frames:
            return work.head(0)
        return pd.concat(frames, ignore_index=True).head(max_rows)

    # For an open-ended top-marker request, the table's precomputed rank is
    # the primary ordering. Score is only a tie-breaker when it is available.
    sort_columns = ["rank"]
    ascending = [True]
    if "score" in work.columns:
        sort_columns.append("score")
        ascending.append(False)
    ranked = work.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    )
    if unique_genes:
        ranked = ranked.drop_duplicates(subset=["gene"], keep="first")
    return ranked.head(max_rows)


def build_marker_bar_plot(marker_rows, title=None):
    """Build a readable horizontal chart for precomputed marker statistics."""
    plot_df = marker_rows.copy()
    if plot_df.empty:
        return None

    # Prefer a statistic whose values are valid for every selected bar. When
    # optional statistics are incomplete, reciprocal rank is deterministic,
    # finite, and keeps rank 1 visually strongest.
    metric = None
    for candidate in ["score", "logFC", "pct_expr"]:
        if candidate not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[candidate], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.notna().all():
            plot_df[candidate] = values
            metric = candidate
            break

    plot_df["rank"] = pd.to_numeric(plot_df["rank"], errors="coerce")
    plot_df["rank"] = plot_df["rank"].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["rank"])
    if plot_df.empty:
        return None

    if metric == "score":
        plot_df["_plot_value"] = plot_df["score"].astype(float)
        x_title = "Marker score"
        plot_df["_value_label"] = [
            f"score {value:.2f}" for value in plot_df["score"]
        ]
    elif metric == "logFC":
        plot_df["_plot_value"] = plot_df["logFC"].astype(float)
        x_title = "Marker log fold-change"
        plot_df["_value_label"] = [
            f"logFC {value:.2f}" for value in plot_df["logFC"]
        ]
    elif metric == "pct_expr":
        pct_values = plot_df["pct_expr"].astype(float)
        plot_df["_plot_value"] = np.where(
            pct_values.abs() <= 1.0,
            pct_values * 100.0,
            pct_values,
        )
        x_title = "Expressing cells (%)"
        plot_df["_value_label"] = [
            f"{(100.0 * value if abs(value) <= 1.0 else value):.1f}%"
            for value in pct_values
        ]
    else:
        safe_rank = plot_df["rank"].clip(lower=1.0).astype(float)
        plot_df["_plot_value"] = 1.0 / safe_rank
        x_title = "Reciprocal marker rank (higher = stronger)"
        plot_df["_value_label"] = [
            f"rank {int(value)}" for value in safe_rank
        ]

    plot_df = plot_df.sort_values("_plot_value", ascending=False)
    value_labels = plot_df["_value_label"].tolist()
    labels = [
        (
            f"{row['gene']} · {pretty_region_name(row['region'])}"
            f" | {row['cell_type']}"
        )
        for _, row in plot_df.iterrows()
    ]

    hover_text = []
    for _, row in plot_df.iterrows():
        details = [
            f"<b>{row['gene']}</b>",
            f"Region: {pretty_region_name(row['region'])}",
            f"Cell type: {row['cell_type']}",
            f"Marker rank: {int(row['rank'])}",
        ]
        if "score" in plot_df.columns and pd.notna(row.get("score")):
            details.append(f"Marker score: {float(row['score']):.3f}")
        if "logFC" in plot_df.columns and pd.notna(row.get("logFC")):
            details.append(f"logFC: {float(row['logFC']):.3f}")
        if "pct_expr" in plot_df.columns and pd.notna(row.get("pct_expr")):
            pct_value = float(row["pct_expr"])
            if abs(pct_value) <= 1.0:
                pct_value *= 100.0
            details.append(f"Expressing cells: {pct_value:.1f}%")
        hover_text.append("<br>".join(details))

    values = plot_df["_plot_value"].astype(float).tolist()
    fig = {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": values,
                "y": labels,
                "text": value_labels,
                "textposition": "outside",
                "cliponaxis": False,
                # Keep bars visually balanced with the category labels.
                "width": 0.58,
                "hovertext": hover_text,
                "hoverinfo": "text",
                "marker": {
                    "color": values,
                    "colorscale": [
                        [0.00, "#8ab4c4"],
                        [0.55, "#5b3d8b"],
                        [1.00, "#32175a"],
                    ],
                    "showscale": False,
                    "line": {"color": "#ffffff", "width": 1},
                },
            }
        ],
        "layout": {
            "title": {
                "text": (
                    f"<b>{title or 'VasQ marker-gene evidence'}</b>"
                    "<br><span style='font-size:12px;color:#64748b'>"
                    "Precomputed marker statistics; not absolute expression"
                    "</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            "height": max(420, 165 + 38 * len(plot_df)),
            "autosize": True,
            "bargap": 0.34,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#ffffff",
            "font": {
                "family": "Satoshi, Arial, sans-serif",
                "color": "#32175a",
                "size": 14,
            },
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": "#8ab4c4",
                "font": {"color": "#32175a"},
            },
            "xaxis": {
                "title": {
                    "text": x_title,
                    "standoff": 16,
                    "font": {"size": 15},
                },
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1.25,
                "ticks": "outside",
                "ticklen": 6,
                "tickwidth": 1.5,
                "tickcolor": "#32175a",
                "tickfont": {"size": 13},
                "showticklabels": True,
                "nticks": 7,
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "gridwidth": 1,
                "zeroline": True,
                "zerolinecolor": "#cbd5e1",
                "rangemode": "tozero",
                "automargin": True,
            },
            "yaxis": {
                "categoryorder": "array",
                "categoryarray": labels,
                "autorange": "reversed",
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1,
                "ticks": "outside",
                "ticklen": 5,
                "tickwidth": 1.25,
                "tickcolor": "#32175a",
                "tickfont": {"size": 14},
                "automargin": True,
            },
            "margin": {"l": 285, "r": 110, "t": 95, "b": 85},
        },
    }

    return json.dumps(fig, allow_nan=False)



# global cached objects
#EXPR_DF = load_expression_data()
#REGION_META_DF = load_region_metadata()
#REGION_ALIAS_MAP = build_region_alias_map(REGION_META_DF)
#CELL_TYPE_ALIAS_MAP = build_cell_type_alias_map(EXPR_DF)

# global cached objects
EXPR_DF = None
REGION_META_DF = None
REGION_ALIAS_MAP = None
CELL_TYPE_ALIAS_MAP = None
AVAILABLE_CELL_TYPES = None
AVAILABLE_REGIONS = None


def ensure_expression_data_loaded():
    global EXPR_DF, REGION_META_DF, REGION_ALIAS_MAP, CELL_TYPE_ALIAS_MAP
    global AVAILABLE_CELL_TYPES, AVAILABLE_REGIONS

    if EXPR_DF is None:
        EXPR_DF = load_expression_data()

    if REGION_META_DF is None:
        REGION_META_DF = load_region_metadata()

    if REGION_ALIAS_MAP is None:
        REGION_ALIAS_MAP = build_region_alias_map(REGION_META_DF)

    if CELL_TYPE_ALIAS_MAP is None:
        CELL_TYPE_ALIAS_MAP = build_cell_type_alias_map(EXPR_DF)

    if AVAILABLE_CELL_TYPES is None:
        AVAILABLE_CELL_TYPES = sorted(EXPR_DF["cell_type"].dropna().unique().tolist())

    if AVAILABLE_REGIONS is None:
        AVAILABLE_REGIONS = sorted(EXPR_DF["region"].dropna().unique().tolist())

def resolve_dataset_entities_with_gpt(
    user_input,
    available_cell_types,
    available_regions,
    available_cell_classes=None,
    available_region_layers=None,
):
    available_cell_types = available_cell_types or []
    available_regions = available_regions or []
    available_cell_classes = available_cell_classes or []
    available_region_layers = available_region_layers or []
    # system_prompt = (
     #   "You are helping map a biology question onto a fixed dataset schema. "
      #  "Choose the closest matching dataset labels from the provided lists. "
       # "Return JSON only with keys: "
        #'{"cell_types": [], "regions": []}. '
        #"Only use labels that appear in the provided lists. "
        #"Do not invent labels."
   # )
    system_prompt = (
        "Map explicitly requested biological entities to labels from a fixed "
        "dataset schema. Resolve cell types, cell classes, brain regions, and "
        "region layers independently. Preserve qualifiers and prefer the most "
        "specific reliable label. Return a label only when the user asks to "
        "include or filter on that value. A value mentioned only as an example "
        "to report when present, in a negated clause, in an exclusion, or in a "
        "question about dataset availability is not a filter. In particular, "
        "obey phrases such as 'do not filter', 'no filter', 'without filtering', "
        "'report when present', and 'whether the data are limited to'. Return "
        "multiple labels in a dimension when the user asks to compare multiple "
        "named values. Do not add every "
        "available value merely because the user asks for an all-values "
        "comparison; leave that dimension empty so downstream code can group "
        "the complete dataset. Never invent labels. Return JSON only with keys: "
        '{"cell_types": [], "cell_classes": [], "regions": [], '
        '"region_layers": []}. Only use exact labels from the supplied lists.'
    )

    user_prompt = (
        f"User query: {user_input}\n\n"
        f"Available cell types: {available_cell_types}\n\n"
        f"Available cell classes: {available_cell_classes}\n\n"
        f"Available regions: {available_regions}\n\n"
        f"Available region layers: {available_region_layers}"
    )

    try:
        response = call_helper_api(system_prompt, user_prompt)

        raw = response.choices[0].message.content.strip()
        logger.info("GPT dataset entity raw response: %s", raw)

        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)

        cell_types = parsed.get("cell_types", [])
        cell_classes = parsed.get("cell_classes", [])
        regions = parsed.get("regions", [])
        region_layers = parsed.get("region_layers", [])

        cell_types = [x for x in cell_types if x in available_cell_types]
        cell_classes = [
            x for x in cell_classes if x in available_cell_classes
        ]
        regions = [x for x in regions if x in available_regions]
        region_layers = [
            x for x in region_layers if x in available_region_layers
        ]

        return cell_types, cell_classes, regions, region_layers

    except Exception as e:
        logger.exception("GPT dataset entity resolution failed: %s", e)
        # None means the helper failed and allows the caller to use local
        # matching. A successful helper response can intentionally return [].
        return None, None, None, None



def resolve_entities_from_text(user_input, alias_map):
    text = normalize_text(user_input)
    found = []

    # exact substring pass
    for alias_norm, canonical in alias_map.items():
        if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", text):
            found.append(canonical)

    # fuzzy single-token fallback
    if not found:
        words = re.findall(r"\w+", text)
        for word in words:
            for alias_norm, canonical in alias_map.items():
                if len(alias_norm.split()) == 1:
                    if difflib.SequenceMatcher(None, alias_norm, word).ratio() > 0.86:
                        found.append(canonical)

    return sorted(set(found))


def extract_entities(user_input):
    ensure_expression_data_loaded()

    # Local alias matching is retained as a fallback.
    alias_cell_matches = resolve_entities_from_text(
        user_input,
        CELL_TYPE_ALIAS_MAP,
    )
    alias_region_matches = resolve_entities_from_text(
        user_input,
        REGION_ALIAS_MAP,
    )

    # GPT selects the best matching labels from the actual dataset schema.
    (
        gpt_cell_matches,
        _,
        gpt_region_matches,
        _,
    ) = (
        resolve_dataset_entities_with_gpt(
            user_input,
            AVAILABLE_CELL_TYPES,
            AVAILABLE_REGIONS,
        )
    )

    # GPT's more specific result takes priority.
    # Local alias matching is used only if the GPT helper failed. A valid
    # empty GPT list means that the dimension was not requested as a filter.
    selected_cell_matches = (
        gpt_cell_matches
        if gpt_cell_matches is not None
        else alias_cell_matches
    )

    selected_region_matches = (
        gpt_region_matches
        if gpt_region_matches is not None
        else alias_region_matches
    )

    selected_cell_matches = list(
        dict.fromkeys(selected_cell_matches)
    )
    selected_region_matches = list(
        dict.fromkeys(selected_region_matches)
    )

    logger.info(
        "extract_entities user_input=%s "
        "alias_cell_matches=%s "
        "gpt_cell_matches=%s "
        "selected_cell_matches=%s "
        "alias_region_matches=%s "
        "gpt_region_matches=%s "
        "selected_region_matches=%s",
        user_input,
        alias_cell_matches,
        gpt_cell_matches,
        selected_cell_matches,
        alias_region_matches,
        gpt_region_matches,
        selected_region_matches,
    )

    return selected_cell_matches, selected_region_matches


def extract_genes(user_input):
    system_prompt = (
        "You are an expert molecular biologist. Extract all human gene symbols "
        "or gene names explicitly mentioned in the user's message. "
        "Return only a Python list, e.g. ['APOE', 'SLC2A1', 'CLDN5']. "
        "If no genes are explicitly mentioned, return []."
    )

    response = call_helper_api(system_prompt, user_input)

    raw_text = response.choices[0].message.content.strip()

    try:
        genes = ast.literal_eval(raw_text)
        if isinstance(genes, list):
            return [str(g).upper().strip() for g in genes if str(g).strip()]
    except Exception:
        pass

    return []


def derive_genes_from_first_search(
    user_input,
    scientific_web_result,
    kg_result=None,
    existing_genes=None,
):
    """Build the gene list that drives VasQ and the second drug search.

    The first Web Search identifies disease biology and candidate genes. This
    helper converts that evidence into a short, ordered list of human gene
    symbols. It does not perform another Web Search.
    """
    existing_genes = [
        str(g).upper().strip()
        for g in (existing_genes or [])
        if str(g).strip()
    ]

    try:
        max_genes = int(os.getenv("MAX_DERIVED_GENES", "10"))
    except ValueError:
        max_genes = 10
    max_genes = max(1, min(max_genes, 20))

    evidence_parts = []
    if scientific_web_result:
        evidence_parts.append(
            "Web/literature evidence:\n"
            + cap_source_text(scientific_web_result, 9000)
        )
    if kg_result:
        evidence_parts.append(
            "Knowledge-graph evidence:\n"
            + cap_source_text(kg_result, 3000)
        )

    if not evidence_parts:
        return existing_genes[:max_genes]

    system_prompt = (
        "Extract and prioritize human gene symbols for a downstream gene-"
        "expression analysis. Return JSON only in the form "
        "{\"genes\":[\"APOE\",\"APP\"]}. Include genes explicitly supported "
        "by the supplied evidence as causal genes, risk genes, associated "
        "genes, or central mechanistic genes for the user's question. Do not "
        "include genes that are merely mentioned incidentally. Use official "
        "HGNC-style uppercase symbols, remove duplicates, rank the strongest "
        "evidence first, and return no more than the requested maximum. Keep "
        "any explicitly supplied user genes when relevant."
    )
    user_prompt = (
        f"User question:\n{user_input}\n\n"
        f"Explicitly supplied genes:\n{existing_genes}\n\n"
        f"Maximum genes:\n{max_genes}\n\n"
        + "\n\n".join(evidence_parts)
    )

    try:
        response = call_helper_api(system_prompt, user_prompt)
        parsed = parse_json_object(response.choices[0].message.content)
        if not parsed:
            raise ValueError("Gene derivation helper returned invalid JSON")

        derived_genes = [
            str(g).upper().strip()
            for g in parsed.get("genes", [])
            if re.fullmatch(r"[A-Z][A-Z0-9.-]{1,19}", str(g).upper().strip())
        ]
        combined = list(dict.fromkeys(existing_genes + derived_genes))
        return combined[:max_genes]
    except Exception:
        logger.exception("Could not derive genes from first-search evidence")
        return existing_genes[:max_genes]

def all_regions(user_input):
    text = user_input.lower()
    triggers = [
        "other regions",
        "all regions",
        "across regions",
        "across all regions",
        "rest of brain",
        "rest of the brain",
        "compared to other regions",
        "versus other regions",
        "than other regions",
        "highest in the brain",
        "unique to"
    ]
    return any(t in text for t in triggers)

def is_cross_region_comparison(user_input):
    system_prompt = (
        "Determine whether the user is asking for comparison against other brain regions "
        "or across the whole dataset. Return only True or False."
    )

    response = call_helper_api(system_prompt, user_input)

    return "true" in response.choices[0].message.content.strip().lower()


def is_region_filtered_query(user_input):
    system_prompt = (
        "Determine whether the user mainly wants results filtered to one or more explicitly "
        "named brain regions, rather than compared to all other regions. Return only True or False."
    )

    response = call_helper_api(system_prompt, user_input)

    return "true" in response.choices[0].message.content.strip().lower()

def format_marker_rows(df, max_rows=20):
    """Format precomputed marker-table rows without implying absolute expression."""
    lines = []

    for _, row in df.head(max_rows).iterrows():
        cell_type = row.get("cell_type", "Unknown cell type")
        region = row.get("region", "Unknown region")
        gene = row.get("gene", "Unknown gene")

        line = (
            f"- {gene} — {pretty_region_name(region)} | {cell_type}: "
            f"rank {int(row['rank'])}"
        )

        if "score" in df.columns and pd.notna(row.get("score")):
            line += f"; score {float(row['score']):.3f}"
        if "logFC" in df.columns and pd.notna(row.get("logFC")):
            line += f"; logFC {float(row['logFC']):.3f}"
        if "pct_expr" in df.columns and pd.notna(row.get("pct_expr")):
            pct_value = float(row["pct_expr"])
            if abs(pct_value) <= 1.0:
                pct_value *= 100.0
            line += f"; expressing cells {pct_value:.1f}%"
        lines.append(line)

    return "\n".join(lines)


def wants_top_genes(user_input):
    text = user_input.lower()
    triggers = [
        "top 5 genes",
        "top 10 genes",
        "top 20 genes",
        "top markers",
        "top marker genes",
        "top genes",
        "marker genes",
        "markers",
    ]
    return any(t in text for t in triggers)


def requested_top_marker_count(user_input, default=12, maximum=25):
    """Read requests such as 'top 5 markers' or '前 10 个 marker genes'."""
    text = str(user_input or "")
    patterns = [
        r"\btop\s+(\d{1,3})\b",
        r"\bfirst\s+(\d{1,3})\b",
        r"前\s*(\d{1,3})\s*(?:个|名)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return max(1, min(maximum, int(match.group(1))))
    return max(1, min(maximum, int(default)))


def wants_specific_gene(user_input, genes):
    return len(genes) > 0


def pretty_region_name(region):
    region_map = {
        "CP": "choroid plexus",
        "Hip-EC": "hippocampal-entorhinal vasculature",
        "ACA": "anterior cerebral artery",
        "BA.CoW": "basilar artery / circle of Willis",
    }
    return region_map.get(region, region)

def marker_gene_expression(user_input, genes_override=None):
    """Query the precomputed marker table for top markers or explicit genes."""

    ensure_expression_data_loaded()

    all_regions_flag = all_regions(user_input)
    cell_types, regions = extract_entities(user_input)

    # Use explicitly supplied genes when available.
    if genes_override is not None:
        gene_names = [
            str(g).upper().strip()
            for g in genes_override
            if str(g).strip()
        ]
        gene_names = list(dict.fromkeys(gene_names))
    else:
        gene_names = extract_genes(user_input)

    requested_cell_types = (
        list(dict.fromkeys(cell_types))
        if cell_types
        else []
    )

    requested_regions = (
        list(dict.fromkeys(regions))
        if regions and not all_regions_flag
        else []
    )

    # Start with the complete expression_markers.csv table.
    df = EXPR_DF.copy()

    # Strict cell-type filtering.
    # Do not automatically add broader cell types.
    if requested_cell_types:
        df = df[
            df["cell_type"].isin(requested_cell_types)
        ]

    # Optional strict region filtering.
    if requested_regions:
        df = df[
            df["region"].isin(requested_regions)
        ]

    # Optional explicit-gene filtering.
    if gene_names:
        df = df[
            df["gene"].isin(gene_names)
        ]

    logger.info(
        "marker_gene_expression "
        "requested_cell_types=%s regions=%s genes=%s rows=%s",
        requested_cell_types,
        requested_regions,
        gene_names,
        len(df),
    )

    # No automatic broadening or fallback to related cell types.
    if df.empty:
        filters = []

        if requested_cell_types:
            filters.append(
                "cell type: " + ", ".join(requested_cell_types)
            )

        if requested_regions:
            filters.append(
                "region: " + ", ".join(requested_regions)
            )

        if gene_names:
            filters.append(
                "genes: " + ", ".join(gene_names)
            )

        if filters:
            return {
                "text": (
                    "No exact precomputed marker-table rows were found for "
                    + "; ".join(filters)
                    + ". The query was not automatically broadened to other "
                    "cell types or regions."
                ),
                "graph_json": None,
                "genes": [],
            }

        return {
            "text": (
                "No matching marker-gene data were found for the "
                "specified query."
            ),
            "graph_json": None,
            "genes": [],
        }

    disclaimer = (
        "These results come from the precomputed VasQ marker table. "
        "Rank and optional score/logFC values are relative marker "
        "statistics, not absolute or matrix mean expression."
    )

    # Case 1:
    # The user supplied one or more genes and wants their marker evidence.
    if gene_names:
        selected = select_marker_rows(
            df,
            gene_order=gene_names,
            max_rows=min(
                24,
                max(8, 4 * len(gene_names)),
            ),
            max_per_gene=4,
        )

        present_genes = (
            selected["gene"]
            .drop_duplicates()
            .tolist()
        )

        missing_genes = [
            gene
            for gene in gene_names
            if gene not in present_genes
        ]

        text_parts = [
            disclaimer,
            "Marker-table evidence for the requested genes:",
            format_marker_rows(
                selected,
                max_rows=24,
            ),
        ]

        if missing_genes:
            text_parts.append(
                "No matching marker rows were found for: "
                + ", ".join(missing_genes)
            )

        if requested_cell_types:
            chart_context = (
                " in "
                + ", ".join(requested_cell_types)
            )
        else:
            chart_context = ""

        graph_json = build_marker_bar_plot(
            selected,
            title=(
                "Marker evidence for "
                + ", ".join(present_genes)
                + chart_context
            ),
        )

        return {
            "text": "\n\n".join(
                part
                for part in text_parts
                if part
            ),
            "graph_json": graph_json,
            "genes": present_genes,
        }

    # Case 2:
    # No genes were supplied. Discover top marker genes directly
    # from expression_markers.csv.
    top_n = requested_top_marker_count(user_input)

    selected = select_marker_rows(
        df,
        max_rows=top_n,
        unique_genes=True,
    )

    selected_genes = (
        selected["gene"]
        .drop_duplicates()
        .tolist()
    )

    context_parts = []

    if requested_cell_types:
        context_parts.append(
            "cell type: "
            + ", ".join(requested_cell_types)
        )

    if requested_regions:
        context_parts.append(
            "region: "
            + ", ".join(requested_regions)
        )

    context_text = (
        "; ".join(context_parts)
        if context_parts
        else "all matched contexts"
    )

    title = (
        f"Top {len(selected_genes)} marker genes — "
        + context_text
    )

    result_text = (
        disclaimer
        + "\n\n"
        + title
        + "\n"
        + format_marker_rows(
            selected,
            max_rows=top_n,
        )
    )

    graph_json = build_marker_bar_plot(
        selected,
        title=title,
    )

    return {
        "text": result_text,
        "graph_json": graph_json,
        "genes": selected_genes,
    }


### KG-RAG Functions ###

# Invoke KG_RAG

def query_kg_rag(user_input):
    url = os.getenv(
        "KG_RAG_URL",
        "http://kg-rag.railway.internal:8080/query"
    )

    try:
        timeout_seconds = _stage_timeout(
            "kg_rag",
            _env_float("KG_RAG_TIMEOUT_SECONDS", 35),
            reserve_seconds=_env_float("VASQ_RETRIEVAL_RESERVE_SECONDS", 110),
        )
    
        logger.info("Calling KG-RAG timeout=%.1fs", timeout_seconds)

        kg_started_at = time.monotonic()
        
        response = requests.post(
            url,
            json={"query": user_input},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        
        logger.info(
            "KG-RAG HTTP succeeded elapsed=%.1fs status=%s response_bytes=%s",
            time.monotonic() - kg_started_at,
            response.status_code,
            len(response.content),
        )

        payload = response.json()
        result = str(payload.get("result", "")).strip()

        failure_markers = [
            "no vectorstore hit",
            "no entity-level hits found",
            "no specific information",
            "i don't have specific information",
        ]

        if not result or any(
            marker in result.lower()
            for marker in failure_markers
        ):
            logger.warning(
                "KG-RAG returned no useful information: %s",
                result[:500]
            )
            return None

        return result

    except requests.exceptions.Timeout:
        logger.warning(
            "KG-RAG timed out; continuing with scientific Web Search"
        )
        return None
    
    except requests.exceptions.RequestException:
        logger.exception("KG-RAG request failed")
        return None


def wrap_plot_label(value, max_chars=22, max_lines=3):
    """Wrap long categorical labels for Plotly without losing hover detail."""
    words = str(value).replace("_", " ").split()
    if not words:
        return ""
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max_chars - 1].rstrip() + "…"
    return "<br>".join(lines)


def matrix_plot_marker(plot_df, *, showscale=True, color_max=None):
    """Shared dot-matrix encoding: color=mean, size=expressing fraction."""
    if color_max is None:
        positive = plot_df.loc[plot_df["mean_expr"] > 0, "mean_expr"]
        color_max = float(positive.quantile(0.95)) if not positive.empty else 1.0
    color_max = max(float(color_max), 0.001)
    return {
        "size": (
            7.0 + 20.0 * np.sqrt(plot_df["pct_expr"].to_numpy())
        ).round(1).tolist(),
        "sizemode": "diameter",
        "color": plot_df["mean_expr"].astype(float).tolist(),
        "cmin": 0,
        "cmax": color_max,
        "colorscale": [
            [0.00, "#eef6f8"],
            [0.25, "#8ab4c4"],
            [0.60, "#5b3d8b"],
            [1.00, "#32175a"],
        ],
        "opacity": [
            0.5 if int(n) < 20 else 0.92
            for n in plot_df["n_cells"]
        ],
        "line": {"color": "#ffffff", "width": 1},
        "showscale": showscale,
        "colorbar": {
            "title": {"text": "Mean<br>expression"},
            "thickness": 14,
            "len": 0.72,
            "outlinewidth": 0,
        },
    }


def matrix_plot_hover(plot_df, comparison_cols):
    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_class": "Cell class",
        "cell_type": "Cell type",
    }
    hover = []
    for _, row in plot_df.iterrows():
        details = [f"<b>{row['gene']}</b>"]
        for col in comparison_cols:
            value = row[col]
            if col == "brain_region":
                value = pretty_region_name(value)
            details.append(f"{display_names.get(col, col)}: {value}")
        details.extend([
            f"Mean expression: {float(row['mean_expr']):.3f}",
            f"Expressing cells: {100.0 * float(row['pct_expr']):.1f}%",
            f"Cells analyzed: {int(row['n_cells']):,}",
        ])
        hover.append("<br>".join(details))
    return hover


def build_single_gene_cell_type_matrix(plot_df, comparison_cols):
    """One gene: region columns by cell-type rows."""
    region_cols = [
        col for col in ["brain_region", "region_layer"]
        if col in comparison_cols
    ]
    if not region_cols:
        return None

    def region_label(row):
        values = []
        for col in region_cols:
            value = row[col]
            if col == "brain_region":
                value = pretty_region_name(value)
            values.append(str(value))
        return " · ".join(values)

    plot_df = plot_df.copy()
    plot_df["_region_full"] = plot_df.apply(region_label, axis=1)
    region_order = sorted(plot_df["_region_full"].drop_duplicates())
    cell_order = sorted(plot_df["cell_type"].drop_duplicates())
    tick_text = [wrap_plot_label(value) for value in region_order]
    gene = str(plot_df["gene"].iloc[0])

    fig = {
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": plot_df["_region_full"].tolist(),
            "y": plot_df["cell_type"].tolist(),
            "hovertext": matrix_plot_hover(plot_df, comparison_cols),
            "hoverinfo": "text",
            "marker": matrix_plot_marker(plot_df),
            "showlegend": False,
        }],
        "layout": {
            "title": {
                "text": (
                    f"<b>{gene} expression: region × cell type</b>"
                    "<br><span style='font-size:12px;color:#64748b'>"
                    "Color = mean expression · Size = expressing-cell fraction"
                    f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            "height": max(650, 260 + 34 * len(cell_order)),
            "autosize": True,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#ffffff",
            "font": {"family": "Satoshi, Arial, sans-serif", "color": "#32175a", "size": 14},
            "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#8ab4c4", "font": {"color": "#32175a"}},
            "xaxis": {
                "title": {"text": "Region", "standoff": 20},
                "categoryorder": "array",
                "categoryarray": region_order,
                "tickmode": "array",
                "tickvals": region_order,
                "ticktext": tick_text,
                "tickangle": -40,
                "tickfont": {"size": 11},
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "automargin": True,
            },
            "yaxis": {
                "title": {"text": "Cell type", "standoff": 14},
                "categoryorder": "array",
                "categoryarray": list(reversed(cell_order)),
                "tickfont": {"size": 12},
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "automargin": True,
            },
            "margin": {"l": 175, "r": 115, "t": 105, "b": 190},
        },
    }
    return json.dumps(fig, allow_nan=False)


def build_cell_type_gene_panels(plot_df, gene_order, comparison_cols):
    """>2 genes: one vertically stacked region-by-gene panel per cell type."""
    region_cols = [
        col for col in ["brain_region", "region_layer"]
        if col in comparison_cols
    ]
    if not region_cols:
        return None

    def region_label(row):
        values = []
        for col in region_cols:
            value = row[col]
            if col == "brain_region":
                value = pretty_region_name(value)
            values.append(str(value))
        return " · ".join(values)

    plot_df = plot_df.copy()
    plot_df["_region_full"] = plot_df.apply(region_label, axis=1)
    region_order = sorted(plot_df["_region_full"].drop_duplicates())
    tick_text = [wrap_plot_label(value) for value in region_order]
    cell_types = sorted(plot_df["cell_type"].drop_duplicates())
    available_genes = plot_df["gene"].drop_duplicates().tolist()
    gene_order = [g for g in gene_order if g in available_genes]
    gene_order += [g for g in available_genes if g not in gene_order]
    positive = plot_df.loc[plot_df["mean_expr"] > 0, "mean_expr"]
    color_max = float(positive.quantile(0.95)) if not positive.empty else 1.0

    traces = []
    annotations = []
    layout_axes = {}
    panel_gap = 0.025
    panel_height = (1.0 - panel_gap * (len(cell_types) - 1)) / len(cell_types)

    for index, cell_type in enumerate(cell_types):
        panel = plot_df[plot_df["cell_type"] == cell_type].copy()
        axis_number = index + 1
        axis_suffix = "" if axis_number == 1 else str(axis_number)
        top = 1.0 - index * (panel_height + panel_gap)
        bottom = max(0.0, top - panel_height)
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": panel["_region_full"].tolist(),
            "y": panel["gene"].tolist(),
            "xaxis": f"x{axis_suffix}",
            "yaxis": f"y{axis_suffix}",
            "hovertext": matrix_plot_hover(panel, comparison_cols),
            "hoverinfo": "text",
            "marker": matrix_plot_marker(
                panel,
                showscale=index == 0,
                color_max=color_max,
            ),
            "showlegend": False,
        })
        layout_axes[f"xaxis{axis_suffix}"] = {
            "domain": [0.0, 1.0],
            "anchor": f"y{axis_suffix}",
            "categoryorder": "array",
            "categoryarray": region_order,
            "tickmode": "array",
            "tickvals": region_order,
            "ticktext": tick_text,
            "tickangle": -40,
            "tickfont": {"size": 10},
            "showticklabels": index == len(cell_types) - 1,
            "showgrid": True,
            "gridcolor": "#edf2f7",
            "automargin": True,
        }
        layout_axes[f"yaxis{axis_suffix}"] = {
            "domain": [bottom, top],
            "anchor": f"x{axis_suffix}",
            "categoryorder": "array",
            "categoryarray": list(reversed(gene_order)),
            "tickfont": {"size": 11},
            "showgrid": True,
            "gridcolor": "#edf2f7",
            "automargin": True,
        }
        annotations.append({
            "text": f"<b>{cell_type}</b>",
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": min(1.0, top + 0.008),
            "xanchor": "left",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 14, "color": "#32175a"},
        })

    layout = {
        "title": {
            "text": (
                "<b>VasQ expression by cell type</b>"
                "<br><span style='font-size:12px;color:#64748b'>"
                "Each panel is one cell type · x = region · y = gene · "
                "Color = mean expression · Size = expressing-cell fraction"
                f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        "height": max(760, 180 + len(cell_types) * max(190, 34 * len(gene_order))),
        "autosize": True,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Satoshi, Arial, sans-serif", "color": "#32175a", "size": 13},
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#8ab4c4", "font": {"color": "#32175a"}},
        "annotations": annotations,
        "margin": {"l": 115, "r": 115, "t": 125, "b": 190},
    }
    layout.update(layout_axes)
    return json.dumps({"data": traces, "layout": layout}, allow_nan=False)


def build_matrix_expression_plot(
    stats_df,
    gene_order=None,
    comparison_cols=None,
):
    """Build a gene-by-comparison-group dot plot.

    The x-axis can represent any requested combination of brain region,
    region layer, cell class, and cell type. Groups below the shared cell-count
    threshold are removed defensively even if the caller already filtered them.
    """
    base_required = {"gene", "mean_expr", "pct_expr", "n_cells"}
    comparison_cols = [
        col
        for col in (
            comparison_cols
            or ["brain_region", "region_layer", "cell_type"]
        )
        if col in stats_df.columns
    ]
    required = base_required.union(comparison_cols)
    if (
        stats_df.empty
        or not comparison_cols
        or not required.issubset(stats_df.columns)
    ):
        return None

    plot_df = stats_df.dropna(
        subset=["gene"] + comparison_cols
    ).copy()
    plot_df["n_cells"] = pd.to_numeric(
        plot_df["n_cells"], errors="coerce"
    ).fillna(0).astype(int)
    plot_df = plot_df[
        plot_df["n_cells"] >= MIN_CELLS_PER_GROUP
    ].copy()
    if plot_df.empty:
        return None

    plot_df["gene"] = plot_df["gene"].astype(str)
    for col in comparison_cols:
        plot_df[col] = plot_df[col].astype(str)
    plot_df["mean_expr"] = pd.to_numeric(
        plot_df["mean_expr"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    plot_df["pct_expr"] = pd.to_numeric(
        plot_df["pct_expr"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    available_genes = plot_df["gene"].drop_duplicates().tolist()
    gene_order = [g for g in (gene_order or []) if g in available_genes]
    gene_order += [g for g in available_genes if g not in gene_order]

    # Purpose-built cell-type layouts avoid the unreadable flattened x-axis
    # produced by concatenating region, layer, and cell type into one label.
    if "cell_type" in comparison_cols and len(gene_order) == 1:
        specialized = build_single_gene_cell_type_matrix(
            plot_df,
            comparison_cols,
        )
        if specialized:
            return specialized

    if "cell_type" in comparison_cols and len(gene_order) > 2:
        specialized = build_cell_type_gene_panels(
            plot_df,
            gene_order,
            comparison_cols,
        )
        if specialized:
            return specialized

    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_class": "Cell class",
        "cell_type": "Cell type",
    }

    def comparison_label(row):
        values = []
        for col in comparison_cols:
            value = str(row[col])
            if col == "brain_region":
                value = pretty_region_name(value)
            values.append(value)
        return "<br>".join(wrap_plot_label(value) for value in values)

    plot_df["_comparison_label"] = plot_df.apply(
        comparison_label,
        axis=1,
    )

    comparison_order = sorted(
        plot_df["_comparison_label"].drop_duplicates().tolist()
    )

    gene_rank = {gene: i for i, gene in enumerate(gene_order)}
    comparison_rank = {
        label: i for i, label in enumerate(comparison_order)
    }
    plot_df["_gene_order"] = plot_df["gene"].map(gene_rank)
    plot_df["_comparison_order"] = plot_df["_comparison_label"].map(
        comparison_rank
    )
    plot_df = plot_df.sort_values(
        ["_gene_order", "_comparison_order"]
    )

    x_values = plot_df["_comparison_label"].tolist()
    y_values = plot_df["gene"].tolist()
    marker_sizes = (
        6.0 + 18.0 * np.sqrt(plot_df["pct_expr"].to_numpy())
    ).round(1).tolist()
    marker_opacity = [
        0.45 if n_cells < 20 else 0.9
        for n_cells in plot_df["n_cells"]
    ]

    color_values = plot_df["mean_expr"].tolist()
    positive_colors = plot_df.loc[
        plot_df["mean_expr"] > 0,
        "mean_expr",
    ]
    color_max = (
        float(positive_colors.quantile(0.95))
        if not positive_colors.empty
        else 1.0
    )
    color_max = max(color_max, 0.001)

    hover_text = []
    for _, row in plot_df.iterrows():
        details = [f"<b>{row['gene']}</b>"]
        for col in comparison_cols:
            value = row[col]
            if col == "brain_region":
                value = pretty_region_name(value)
            details.append(f"{display_names.get(col, col)}: {value}")
        details.extend([
            f"Mean expression: {float(row['mean_expr']):.3f}",
            f"Expressing cells: {100.0 * float(row['pct_expr']):.1f}%",
            f"Cells analyzed: {int(row['n_cells']):,}",
        ])
        hover_text.append("<br>".join(details))

    dimension_title = " / ".join(
        display_names.get(col, col)
        for col in comparison_cols
    )
    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": x_values,
                "y": y_values,
                "hovertext": hover_text,
                "hoverinfo": "text",
                "marker": {
                    "size": marker_sizes,
                    "sizemode": "diameter",
                    "color": color_values,
                    "cmin": 0,
                    "cmax": color_max,
                    "colorscale": [
                        [0.00, "#eef6f8"],
                        [0.25, "#8ab4c4"],
                        [0.60, "#5b3d8b"],
                        [1.00, "#32175a"],
                    ],
                    "opacity": marker_opacity,
                    "line": {"color": "#ffffff", "width": 1},
                    "colorbar": {
                        "title": {"text": "Mean<br>expression"},
                        "thickness": 14,
                        "len": 0.72,
                        "outlinewidth": 0,
                    },
                },
                "showlegend": False,
            }
        ],
        "layout": {
            "title": {
                "text": (
                    f"<b>VasQ expression by {dimension_title}</b>"
                    "<br><span style='font-size:12px;color:#64748b'>"
                    "Color = mean expression · Size = expressing-cell fraction"
                    f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells"
                    "</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            "height": max(650, 220 + 42 * len(gene_order)),
            "autosize": True,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#ffffff",
            "font": {
                "family": "Satoshi, Arial, sans-serif",
                "color": "#32175a",
                "size": 15,
            },
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": "#8ab4c4",
                "font": {"color": "#32175a"},
            },
            "xaxis": {
                "title": {"text": dimension_title, "standoff": 18},
                "categoryorder": "array",
                "categoryarray": comparison_order,
                "tickangle": -40,
                "tickfont": {"size": 11},
                "showticklabels": True,
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1,
                "ticks": "outside",
                "ticklen": 6,
                "tickwidth": 1,
                "tickcolor": "#32175a",
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
                "automargin": True,
            },
            "yaxis": {
                "title": {"text": "Gene", "standoff": 12},
                "categoryorder": "array",
                "categoryarray": list(reversed(gene_order)),
                "showticklabels": True,
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1,
                "ticks": "outside",
                "ticklen": 6,
                "tickwidth": 1,
                "tickcolor": "#32175a",
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
                "automargin": True,
            },
            "margin": {"l": 110, "r": 115, "t": 105, "b": 190},
        },
    }

    return json.dumps(fig, allow_nan=False)


### Search Functions ###

# Search Google
def search_google(query):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")

    if not google_api_key:
        logger.error("Google search failed: GOOGLE_API_KEY is missing")
        return ""

    if not search_engine_id:
        logger.error("Google search failed: SEARCH_ENGINE_ID is missing")
        return ""

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 5,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        logger.info("Google status: %s", response.status_code)
        logger.info("Google body: %s", response.text[:1000])
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Google search request failed")
        return ""

    items = data.get("items", [])
    if not items:
        logger.warning("Google search returned no items")
        return ""

    formatted_results = ""
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "No Title")
        link = item.get("link", "No Link")
        snippet = item.get("snippet", "")
        formatted_results += f"{idx}. {title}\n{link}\n{snippet}\n\n"

    return formatted_results

import os
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine


import json
import os
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account



from google.oauth2 import service_account

def search_vertex_ai(query):
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "global")
    engine_id = os.getenv("VERTEX_ENGINE_ID")
    serving_config_id = os.getenv("VERTEX_SERVING_CONFIG", "default_search")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    logger.info(
        "Vertex config present? project=%s engine=%s sa_json_present=%s",
        bool(project_id), bool(engine_id), bool(sa_json)
    )

    if not project_id or not engine_id:
        logger.error("Missing Vertex config: VERTEX_PROJECT_ID or VERTEX_ENGINE_ID")
        return ""

    if not sa_json:
        logger.error("Missing GCP_SERVICE_ACCOUNT_JSON")
        return ""

    try:
        sa_info = json.loads(sa_json)
        credentials = service_account.Credentials.from_service_account_info(sa_info)
    except Exception as e:
        logger.exception("Failed to parse GCP_SERVICE_ACCOUNT_JSON: %s", e)
        return ""

    client_options = ClientOptions(
        api_endpoint=(
            "discoveryengine.googleapis.com"
            if location == "global"
            else f"{location}-discoveryengine.googleapis.com"
        )
    )

    try:
        client = discoveryengine.SearchServiceClient(
            credentials=credentials,
            client_options=client_options,
        )
    except Exception as e:
        logger.exception("Failed to create Vertex client: %s", e)
        return ""

    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection/"
        f"engines/{engine_id}/servingConfigs/{serving_config_id}"
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=5,
    )

    try:
        response = client.search(request=request)
        logger.info("Vertex result count=%d", len(response.results))
        for i, result in enumerate(response.results, 1):
            doc = result.document
            logger.info("doc[%d].id=%r", i, getattr(doc, "id", None))
            logger.info(
                "doc[%d].derived_struct_data type=%s value=%r",
                i,
                type(getattr(doc, "derived_struct_data", None)),
                getattr(doc, "derived_struct_data", None),
            )
    except Exception as e:
        logger.exception("Vertex AI Search failed: %s", e)
        return ""

    formatted_results = []
    for i, result in enumerate(response.results, start=1):
        doc = result.document
        derived = getattr(doc, "derived_struct_data", None)

        title = ""
        link = ""
        snippet = ""

        if derived:
            try:
                derived = dict(derived)
            except Exception:
                derived = {}

        if isinstance(derived, dict):
            title = derived.get("title", "") or ""
            link = derived.get("link", "") or ""

            snippets = derived.get("snippets", []) or []
            if snippets:
                first_snippet = snippets[0]
                try:
                    first_snippet = dict(first_snippet)
                except Exception:
                    pass
                if isinstance(first_snippet, dict):
                    snippet = first_snippet.get("snippet", "") or ""

        if not title:
            title = getattr(doc, "id", "No Title")

        formatted_results.append(f"{i}. {title}\n{link}\n{snippet}")

    return "\n\n".join(formatted_results).strip()


def wants_web_search(user_input: str) -> bool:
    lowered = user_input.lower()
    web_terms = [
        "search google",
        "google",
        "google it",
        "find articles",
        "articles",
        "for papers",
        "find papers",
        "find paper",
        "find papers on",
        "search pubmed",
        "pubmed",
        "google scholar",
        "scholar",
        "look up papers",
        "find studies",
        "search the web",
        "web search",
    ]
    return any(term in lowered for term in web_terms)


def explicitly_requests_vasq_matrix(user_input: str) -> bool:
    """Return True when the user explicitly limits the answer to VasQ data."""
    text = normalize_text(user_input)
    vasq_only_terms = [
        "using the vasq matrix",
        "using only the vasq matrix",
        "use the vasq matrix",
        "use only the vasq matrix",
        "from the vasq matrix",
        "from the vasq matrix only",
        "based on the vasq matrix",
        "using vasq matrix data",
        "using only vasq data",
        "vasq matrix only",
    ]
    return any(term in text for term in vasq_only_terms)


def parse_json_object(raw_text):
    """Parse a helper-model JSON object without trusting markdown fences."""
    if not raw_text:
        return None

    text = str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def recent_conversation_context(history, max_messages=8, max_chars=5000):
    messages = []
    for message in (history or [])[-max_messages:]:
        if message.get("role") not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            messages.append(f"{message['role']}: {content}")
    return "\n".join(messages)[-max_chars:]


def is_simple_conversational_message(user_input):
    """Fast path that guarantees greetings/thanks do not trigger retrieval."""
    text = normalize_text(user_input)
    text = re.sub(r"[^\w\s]", "", text).strip()
    phrases = {
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "ok", "okay",
        "ä½ å¥½", "æ‚¨å¥½", "å—¨", "è°¢è°¢", "å¥½çš„", "æ”¶åˆ°", "å†è§",
    }
    return text in phrases


def fallback_query_intent(user_input):
    """Conservative rule-based intent used only when helper parsing fails."""
    text = user_input or ""
    lowered = text.lower()
    excluded_symbols = {
        "THE", "AND", "FOR", "WITH", "WHAT", "HOW", "WHY", "CAN",
        "DO", "IS", "ARE", "WEB", "DNA", "RNA", "KG", "RAG",
    }
    genes = [
        token for token in re.findall(r"\b[A-Z][A-Z0-9-]{1,11}\b", text)
        if token not in excluded_symbols
    ]
    genes = list(dict.fromkeys(genes))

    biomedical_terms = [
        "gene", "protein", "cell", "brain", "vascular", "disease",
        "drug", "molecule", "pathway", "expression", "expressed",
        "mutation", "receptor", "enzyme", "neuron", "astrocyte",
        "endothelial", "cancer", "syndrome", "alzheimer", "parkinson",
    ]
    asks_markers = wants_marker_query(text) or wants_top_genes(text)
    asks_expression = asks_markers or wants_matrix_expression_query(text)
    drug_terms = [
        "drug", "drugs", "treatment", "treatments", "therapy", "therapies",
        "therapeutic", "therapeutics", "compound", "compounds",
        "small molecule", "small molecules", "modulator", "modulators",
        "inhibitor", "inhibitors", "agonist", "agonists",
        "antagonist", "antagonists", "clinical trial", "clinical trials",
    ]
    asks_drugs = any(term in lowered for term in drug_terms)

    return {
        "is_scientific": bool(
            genes
            or any(term in lowered for term in biomedical_terms)
            or looks_like_expression_query(text)
        ),
        "asks_expression": asks_expression,
        "asks_markers": asks_markers,
        "asks_drugs": asks_drugs,
        "use_vasq": asks_expression and not any(
            term in lowered
            for term in [
                "not in vasculature", "non-vascular", "nonvascular",
                "liver", "kidney", "lung", "heart", "blood plasma",
                "gray matter", "grey matter", "parenchyma",
            ]
        ),
        "genes": genes,
        "diseases": [],
        "resolved_question": text.strip(),
    }


def analyze_query_intent(user_input, history=None):
    """Resolve scientific intent and entities once for all retrieval branches."""
    if is_simple_conversational_message(user_input):
        return {
            "is_scientific": False,
            "asks_expression": False,
            "asks_markers": False,
            "asks_drugs": False,
            "use_vasq": False,
            "genes": [],
            "diseases": [],
            "resolved_question": user_input.strip(),
        }

    system_prompt = (
        "Classify a conversation turn for a biomedical/neuroscience research "
        "assistant. Return JSON only with keys: is_scientific (boolean), "
        "asks_expression (boolean), asks_markers (boolean), asks_drugs "
        "(boolean), genes (array of human gene symbols), diseases (array of "
        "disease/condition names), use_vasq (boolean), and resolved_question "
        "(string). Set asks_drugs only when the current question explicitly "
        "asks about drugs, compounds, treatments, therapies, modulators, or "
        "clinical candidates; the mere presence of a disease or gene is not "
        "enough. A greeting, "
        "thanks, casual chat, or "
        "app/meta question is not scientific. Set asks_expression only when "
        "the user is asking about measured gene expression, expression "
        "differences, expressing-cell percentage, regional/cell-type "
        "distribution, or marker genes. Set asks_markers for marker/rank/top-"
        "gene questions. Set use_vasq when asks_expression is true and the "
        "question concerns brain vasculature, vascular cell types/regions, or "
        "does not specify a different tissue; set it false when the user "
        "explicitly asks about a non-vascular or other-organ tissue. Resolve "
        "short follow-ups from recent context, but do "
        "not invent a gene, disease, cell type, or brain region. Include genes "
        "or diseases inherited from context only when the reference is "
        "unambiguous. Normalize gene symbols to uppercase."
    )
    user_prompt = (
        f"Recent conversation:\n{recent_conversation_context(history)}\n\n"
        f"Current user message:\n{user_input}"
    )

    try:
        response = call_helper_api(system_prompt, user_prompt)
        raw = response.choices[0].message.content
        parsed = parse_json_object(raw)
        if not parsed:
            raise ValueError("Intent helper did not return a JSON object")

        genes = [
            str(x).upper().strip()
            for x in parsed.get("genes", [])
            if str(x).strip()
        ]
        diseases = [
            str(x).strip()
            for x in parsed.get("diseases", [])
            if str(x).strip()
        ]
        return {
            "is_scientific": bool(parsed.get("is_scientific", False)),
            "asks_expression": bool(parsed.get("asks_expression", False)),
            "asks_markers": bool(parsed.get("asks_markers", False)),
            "asks_drugs": bool(parsed.get("asks_drugs", False)),
            "use_vasq": bool(parsed.get("use_vasq", False)),
            "genes": list(dict.fromkeys(genes)),
            "diseases": list(dict.fromkeys(diseases)),
            "resolved_question": str(
                parsed.get("resolved_question") or user_input
            ).strip(),
        }
    except Exception:
        logger.exception("Scientific intent analysis failed; using rules")
        return fallback_query_intent(user_input)


def assess_kg_relevance(user_input, kg_result):
    """Separate a useful KG hit from a non-empty but irrelevant response."""
    if not kg_result or not str(kg_result).strip():
        return {
            "relevant": False,
            "has_function_or_pathway": False,
            "reason": "no KG result",
        }

    system_prompt = (
        "Assess whether biomedical knowledge-graph content is relevant to a "
        "question. Return JSON only with keys: relevant (boolean), "
        "has_function_or_pathway (boolean), and reason (short string). Content "
        "is relevant only if it addresses the question's entities or their "
        "biological relationships. Do not treat generic biomedical text as a "
        "relevant hit."
    )
    prompt = (
        f"Question:\n{user_input}\n\n"
        f"Knowledge-graph result:\n{str(kg_result)[:5000]}"
    )

    try:
        response = call_helper_api(system_prompt, prompt)
        parsed = parse_json_object(response.choices[0].message.content)
        if not parsed:
            raise ValueError("KG assessment helper returned invalid JSON")
        return {
            "relevant": bool(parsed.get("relevant", False)),
            "has_function_or_pathway": bool(
                parsed.get("has_function_or_pathway", False)
            ),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except Exception:
        logger.exception("KG relevance assessment failed")
        lowered = str(kg_result).lower()
        return {
            "relevant": True,
            "has_function_or_pathway": any(
                term in lowered
                for term in ["pathway", "function", "participates", "process"]
            ),
            "reason": "fallback assessment",
        }


def ensure_chat_initialized(history):
    """Initialize each conversation independently; avoid process-global routing."""
    if not any(message.get("role") == "system" for message in history):
        initialize(history)


def retrieved_text_and_graph(result):
    if isinstance(result, dict):
        return str(result.get("text", "") or ""), result.get("graph_json")
    if result is None:
        return "", None
    return str(result), None


def cap_source_text(text, limit):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[Source text truncated]"


def build_partial_response(
    genes,
    vasq_text,
    scientific_web_result,
    drug_result=None,
):
    """Return recovered evidence if the final model synthesis is unavailable."""
    sections = [
        "I recovered partial results, but the final evidence synthesis was "
        "not available before the request deadline."
    ]
    if genes:
        sections.append("Associated genes:\n" + ", ".join(genes))
    if vasq_text:
        sections.append(
            "VasQ expression results:\n" + cap_source_text(vasq_text, 6000)
        )
    if scientific_web_result:
        sections.append(
            "Scientific evidence:\n"
            + cap_source_text(scientific_web_result, 4000)
        )
    if drug_result:
        sections.append(
            "Drug and small-molecule evidence:\n"
            + cap_source_text(drug_result, 3000)
        )
    if len(sections) == 1:
        sections.append(
            "No reliable partial evidence was available. Please retry the request."
        )
    return "\n\n".join(sections)


### Function Descriptions ###

functions = [
    {
        "name": "marker_gene_expression",
        "description": "Queries precomputed VasQ marker-gene rankings by \
        cell type and brain region. Use it for top-marker, marker-rank, or \
        enriched-gene questions; a top-marker query does not require the user \
        to provide genes. These marker statistics are not matrix mean \
        expression.",
        "parameters": {
            "type": "object",
            "properties": 
                {"user_input":{
                    "type":"string","description":"Full text of user input."}
                },
            "required": ["user_input"],
        }
    },
    {
        "name": "query_kg_rag",
        "description": "Collects biomedical information related to diseases \
        mentioned in user queries.",
        "parameters": {
            "type": "object",
            "properties": 
                {"user_input":{
                    "type":"string","description":"user input"}
                },
            "required": ["user_input"],
        }
    },
    {
        "name": "matrix_expression",
        "description": (
            "Returns log-normalized gene expression summaries from the sparse "
            "HVG matrix. It can filter and compare brain regions, region layers, "
            "cell classes, cell types, and sex while keeping requested dimensions "
            "separate. Groups with fewer than 10 cells are not returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                     "type": "string",
                     "description": "Full text of user input."
                }
            },
            "required": ["user_input"]
        }
     }
]

def looks_like_expression_query(user_input):
    text = user_input.lower()

    expression_terms = [
        "expression", "expressed", "marker", "markers",
        "top genes", "highest expressed", "most expressed",
        "enriched", "upregulated", "gene expression"
    ]

    anatomy_terms = [
        "endothelial", "arterial", "arteriole", "artery", "capillary",
        "fenestrated endothelial", "pericyte", "smooth muscle", "venule",
        "vein", "astrocyte", "neuron", "fibroblast", "epithelial",
        "oligodendrocyte", "microglia", "macrophage", "t cell",
        "choroid plexus", "hippocampus", "pons", "amygdala",
        "thalamus", "midbrain", "cerebellum", "insula"
    ]

    return (
        any(term in text for term in expression_terms) or
        any(term in text for term in anatomy_terms)
    )


### Main Chat Function ###

# Chat between user and chatbot




#def chat(user_input, history):
#    global func_flag, init_flag
#
#    if init_flag:
#        history.clear()
#        initialize(history)
#
#    update_history(history, "user", user_input)
#
#    retrieved_info = None
#    chat_message = call_api(history, functions)
#    logger.info("First model content: %s", getattr(chat_message, "content", None))
#    logger.info("Function call present: %s", bool(chat_message.function_call))
#    
#    if chat_message.function_call:
#        try:
#            retrieved_info = func_call(user_input, chat_message, history)
#        except Exception as e:
#            logger.exception("func_call failed: %s", e)
#            retrieved_info = None

#    # If function calling missed it or returned nothing useful, try local gene-expression data first
#    if not retrieved_info and looks_like_expression_query(user_input):

#        logger.info("Heuristic routing to gene_expression first")
#        try:
#            retrieved_info = gene_expression(user_input)
#        except Exception as e:
#            logger.exception("gene_expression heuristic failed: %s", e)
#            retrieved_info = None
#
#    # KG-RAG before Google for graph-style biomedical questions
#    if not retrieved_info:
#        lowered = user_input.lower()
#        kg_terms = [
#            "drug", "drugs", "target", "targets", "disease",
#            "association", "associated", "pathway", "pathways",
#            "implicated", "implication"
#        ]
#        if any(term in lowered for term in kg_terms):
#            try:
#                retrieved_info = query_kg_rag(user_input)
#            except Exception as e:
#                logger.exception("KG query failed: %s", e)
#                retrieved_info = None
#
#    if not retrieved_info:
#        logger.info("Calling Google Vertex AI API...")
#        try:
#            retrieved_info = search_openai_web(user_input)
#            if not retrieved_info:
#                logger.warning("Google search returned no usable results")
#        except Exception as e:
#            logger.exception("Google search failed")
#            retrieved_info = None
    
#    graph_json = None

#    if not retrieved_info:
#        retrieved_info = ""

#    if isinstance(retrieved_info, dict):
#        graph_json = retrieved_info.get("graph_json")
#        retrieved_text = retrieved_info.get("text", "")
#    else:
#        retrieved_text = retrieved_info

#    if not isinstance(retrieved_text, str):
#        retrieved_text = str(retrieved_text)

#    retrieved_text = retrieved_text[:4000]


#    synthesis_messages = history[:] + [
#        {
#            "role": "system",
#            "content": (
#                "Answer the user's question directly using the retrieved information below. "
#                "Do not mention search tools or internal routing."
#
#            )
#        },
#        {
#            "role": "user",
#            "content": f"Retrieved information:\n{retrieved_text}"
#        }
#    ]



#    final_message = call_api(synthesis_messages).content
#    logger.info("Final message returned to UI: %r", final_message)
#    update_history(history, "assistant", final_message)

#    logger.info("Retrieved text going into history: %s", retrieved_text)
#    logger.info("Final message returned to UI: %s", final_message)
#    return final_message, history, graph_json

### new chat


class ChatCancelled(Exception):
    """Stop a chat turn without converting it into an error response."""


def _raise_if_cancelled(should_stop):
    if should_stop is not None and should_stop():
        raise ChatCancelled("Chat request stopped by user.")



def _chat_impl(user_input, history, should_stop=None):
    _raise_if_cancelled(should_stop)

    if history is None:
        history = []

    ensure_chat_initialized(history)

    # Resolve the current turn against prior context before adding it to history.
    intent = analyze_query_intent(user_input, history)
    _raise_if_cancelled(should_stop)
    update_history(history, "user", user_input)

    # Greetings, thanks, casual conversation, and meta questions do not search.
    if not intent.get("is_scientific"):
        _raise_if_cancelled(should_stop)
        direct_message = call_api(
            history,
            stage_name="direct_answer",
            timeout_seconds=_env_float("OPENAI_SYNTHESIS_TIMEOUT_SECONDS", 45),
        )
        _raise_if_cancelled(should_stop)
        final_message = getattr(direct_message, "content", None) or (
            "I'm sorry, but I couldn't generate a response."
        )
        update_history(history, "assistant", final_message)
        return final_message, history, None

    resolved_question = intent.get("resolved_question") or user_input
    user_supplied_genes = list(intent.get("genes") or [])
    genes = user_supplied_genes[:]
    diseases = intent.get("diseases") or []

    # An explicit VasQ-matrix-only request should not spend several minutes on
    # KG-RAG or Web Search. An equally explicit request for web/literature
    # search takes precedence and keeps the normal external-evidence route.
    direct_vasq_only = (
        explicitly_requests_vasq_matrix(user_input)
        and not wants_web_search(user_input)
    )
    if direct_vasq_only:
        # Do not let an intent-classifier miss undo the user's explicit source
        # instruction. These flags ensure the request reaches Branch B.
        intent["asks_expression"] = True
        intent["use_vasq"] = True
    logger.info("Direct VasQ-only routing=%s", direct_vasq_only)

    kg_result = None
    kg_assessment = {
        "relevant": False,
        "reason": "KG-RAG was not run for a VasQ-only request.",
    }
    scientific_web_result = None

    # Branch A: unless this is a direct VasQ-only request, run KG-RAG followed
    # by a function/pathway-oriented Web Search.
    if direct_vasq_only:
        logger.info(
            "Direct VasQ matrix request detected; skipping KG-RAG and "
            "OpenAI Web Search"
        )
    else:
        _raise_if_cancelled(should_stop)
        try:
            kg_result = query_kg_rag(resolved_question)
        except Exception:
            logger.exception("KG-RAG branch failed")
            kg_result = None

        _raise_if_cancelled(should_stop)
        kg_assessment = assess_kg_relevance(resolved_question, kg_result)

        _raise_if_cancelled(should_stop)
        try:
            logger.info("Web Search stage 1/2: scientific knowledge and genes")
            scientific_web_result = search_scientific_web(
                resolved_question,
                kg_context=kg_result,
                kg_assessment=kg_assessment,
            )
        except Exception:
            logger.exception("Scientific Web Search branch failed")
            scientific_web_result = None

        _raise_if_cancelled(should_stop)

        # The first search can discover genes that were not explicitly written
        # in the user's question. Use that evidence-derived list for matrix
        # expression and drug searches.
        genes = derive_genes_from_first_search(
            resolved_question,
            scientific_web_result,
            kg_result=(
                kg_result
                if kg_assessment.get("relevant")
                else None
            ),
            existing_genes=genes,
        )

        logger.info(
            "Gene list after primary scientific search: %s",
            genes,
        )
    
    # Run a smaller, focused Web Search when a matrix-expression question
    # needs genes but the primary search did not resolve any.
    needs_gene_fallback = (
        not direct_vasq_only
        and intent.get("asks_expression")
        and not intent.get("asks_markers")
        and not user_supplied_genes
        and not genes
    )
    
    if needs_gene_fallback:
        _raise_if_cancelled(should_stop)
        logger.warning(
            "Primary evidence resolved no genes; "
            "starting focused gene fallback search"
        )
    
        try:
            fallback_web_result = search_gene_fallback(
                resolved_question
            )
        except Exception:
            logger.exception(
                "Focused gene fallback search failed"
            )
            fallback_web_result = None

        _raise_if_cancelled(should_stop)
    
        if fallback_web_result:
            fallback_genes = derive_genes_from_first_search(
                resolved_question,
                fallback_web_result,
                kg_result=None,
                existing_genes=[],
            )

            genes = list(
                dict.fromkeys(
                    genes + fallback_genes
                )
            )[:20]
    
            # Preserve fallback evidence for final synthesis and citations.
            evidence_parts = [
                part
                for part in [
                    scientific_web_result,
                    (
                        "Focused fallback gene evidence:\n"
                        + fallback_web_result
                    ),
                ]
                if part
            ]
    
            scientific_web_result = "\n\n".join(
                evidence_parts
            )
    
            logger.info(
                "Gene list after fallback search: %s",
                genes,
            )
        else:
            logger.warning(
                "Focused gene fallback search returned no evidence"
            )

    # Branch B: calculate expression for the explicit or first-search-derived
    # gene list. Marker/rank questions use the ranked marker table; measured
    # expression questions use the VasQ matrix.
    vasq_result = None
    vasq_note = ""
    graph_json = None

    if intent.get("asks_expression") and intent.get("use_vasq"):
        _raise_if_cancelled(should_stop)
        if genes or intent.get("asks_markers"):
            try:
                if intent.get("asks_markers"):
                    vasq_result = marker_gene_expression(
                        resolved_question,
                        genes_override=user_supplied_genes,
                    )
                    if isinstance(vasq_result, dict):
                        marker_genes = vasq_result.get("genes") or []
                        genes = list(dict.fromkeys(marker_genes + genes))[:20]
                else:
                    vasq_result = matrix_expression(
                        resolved_question,
                        genes_override=genes,
                    )
                logger.info("VasQ analysis completed for genes: %s", genes)
            except Exception:
                logger.exception("VasQ branch failed")
                vasq_result = None
                vasq_note = "The VasQ query failed and returned no usable data."
        else:
            vasq_note = (
                "The question concerns expression, but no gene was resolved "
                "from the current turn or recent context, so the VasQ matrix "
                "was not queried."
            )

    _raise_if_cancelled(should_stop)
    vasq_text, graph_json = retrieved_text_and_graph(vasq_result)

    # Branch C: optional second Web Search. Only run it when the user actually
    # asked for drugs/therapeutics; a disease or gene alone is not sufficient.
    drug_result = None
    asks_drugs = bool(intent.get("asks_drugs"))
    if not direct_vasq_only and asks_drugs and (genes or diseases):
        _raise_if_cancelled(should_stop)
        try:
            logger.info(
                "Web Search stage 2/2: drugs for genes=%s diseases=%s",
                genes,
                diseases,
            )
            drug_result = search_drugs_and_small_molecules(
                resolved_question,
                genes=genes,
                diseases=diseases,
            )
        except Exception:
            logger.exception("Drug/small-molecule Web Search branch failed")
            drug_result = None

    _raise_if_cancelled(should_stop)

    # Cap each source independently so a long Web result cannot erase VasQ data.
    evidence_parts = []

    if direct_vasq_only:
        evidence_parts.append(
            "SOURCE SCOPE:\n"
            "The user explicitly requested VasQ matrix data only. KG-RAG "
            "and Web Search were not run."
        )

    if genes:
        evidence_parts.append(
            (
                "GENES USED FOR VASQ ANALYSIS:\n"
                if direct_vasq_only
                else "GENE LIST DERIVED FROM THE FIRST SEARCH:\n"
            )
            + ", ".join(genes)
        )

    if vasq_text:
        evidence_parts.append(
            "VASQ EXPRESSION OR MARKER DATA:\n"
            + cap_source_text(vasq_text, 15000)
        )
    elif vasq_note:
        evidence_parts.append("VASQ STATUS:\n" + vasq_note)

    if not direct_vasq_only:
        if kg_result and kg_assessment.get("relevant"):
            evidence_parts.append(
                "RELEVANT BIOMEDICAL KNOWLEDGE-GRAPH CONTEXT:\n"
                + cap_source_text(kg_result, 3000)
            )
        else:
            evidence_parts.append(
                "KNOWLEDGE-GRAPH STATUS:\n"
                "No sufficiently relevant knowledge-graph content was available."
            )

        if scientific_web_result:
            evidence_parts.append(
                "SCIENTIFIC WEB/LITERATURE EVIDENCE:\n"
                + cap_source_text(scientific_web_result, 7000)
            )
        else:
            evidence_parts.append(
                "SCIENTIFIC WEB STATUS:\nNo usable web evidence was returned."
            )

    if not direct_vasq_only and asks_drugs and (genes or diseases):
        if drug_result:
            evidence_parts.append(
                "DRUG AND SMALL-MOLECULE EVIDENCE:\n"
                + cap_source_text(drug_result, 7000)
            )
        else:
            evidence_parts.append(
                "DRUG SEARCH STATUS:\n"
                "No usable drug or small-molecule search result was returned."
            )

    evidence_package = "\n\n".join(evidence_parts)

    synthesis_instruction = {
        "role": "system",
        "content": (
            "Answer the user's scientific question directly using the evidence "
            "package supplied after the conversation. Integrate only relevant "
            "evidence. When SOURCE SCOPE says VasQ matrix data only, do not "
            "introduce literature or knowledge-graph claims and do not imply "
            "that external sources were searched. Treat knowledge-graph "
            "relationships as associations, "
            "not proof of causality. Use web/literature evidence for current "
            "function, pathway, mechanism, clinical-stage, and regulatory "
            "claims, and preserve its citations. Use VasQ only for measured "
            "brain-vasculature expression claims; distinguish matrix mean "
            "expression from marker rank/score. In the drug section, clearly "
            "separate direct gene/protein modulators, pathway-related "
            "compounds, and disease-directed treatments, and distinguish "
            "approved, clinical, preclinical, and research-tool status. Never "
            "claim that a disease drug directly targets a gene without direct "
            "support. For disease questions that ask about genes and their "
            "expression, organize the answer into two explicit sections: "
            "(1) associated genes and supporting knowledge and (2) VasQ "
            "matrix expression by the requested brain region, region layer, "
            "cell class, and/or cell type dimensions with the supplied measured "
            "values. Preserve every comparison dimension present in the VasQ "
            "table and never merge distinct region layers, cell types, or brain "
            "regions. Treat the supplied 'Applied matrix filters' line as the "
            "authoritative record of which filters were actually used. Treat "
            "the supplied 'Observed comparison values' list as authoritative "
            "coverage: never claim that a value such as White Matter Tracts is "
            "absent when it appears in that list, even if a capped detail table "
            "shows only some rows. Use reader-facing dimension columns named Brain region, "
            "Region layer, Cell class, and Cell type when each is present, "
            "followed by Mean expression (log-normalized), Expressing cells, "
            "and Cells analyzed (n). Never reconstruct, infer, or display a "
            f"group with fewer than {MIN_CELLS_PER_GROUP} cells. Convert "
            "pct_expr fractions to percentages (for "
            "example, 0.15 becomes 15.0%); never expose pct_expr or n as "
            "unexplained technical headers. Briefly state that mean "
            "expression includes zero-valued cells, Expressing cells is the "
            "nonzero-expression percentage, and Cells analyzed (n) is the "
            "group size. Cover every analyzed gene when evidence is "
            "available. When the user explicitly asks for drugs or "
            "therapeutics, add a third section for compounds related to those "
            "genes; otherwise "
            "do not add or search for drug information. If a source reports "
            "no result, state the limitation briefly rather "
            "than inventing content. Do not mention internal routing or "
            "implementation details."
        ),
    }

    if history and history[0].get("role") == "system":
        synthesis_messages = (
            [history[0], synthesis_instruction]
            + history[1:]
        )
    else:
        synthesis_messages = [synthesis_instruction] + history[:]

    synthesis_messages.append({
        "role": "user",
        "content": (
            "Evidence package for the current question:\n\n"
            + evidence_package
        ),
    })

    _raise_if_cancelled(should_stop)
    try:
        final_message_obj = call_api(
            synthesis_messages,
            stage_name="final_synthesis",
            timeout_seconds=_env_float("OPENAI_SYNTHESIS_TIMEOUT_SECONDS", 45),
            reserve_seconds=2,
        )
        final_message = getattr(final_message_obj, "content", None) or (
            "I'm sorry, but I couldn't synthesize the available evidence."
        )
    except Exception:
        logger.exception(
            "Final synthesis failed; returning recovered partial evidence"
        )
        final_message = build_partial_response(
            genes,
            vasq_text,
            scientific_web_result,
            drug_result=drug_result,
        )
    _raise_if_cancelled(should_stop)
    logger.info("Final message generated: %r", final_message)
    update_history(history, "assistant", final_message)

    return final_message, history, graph_json


def chat(user_input, history, should_stop=None):
    """Run one synchronous chat turn inside a hard wall-clock budget."""
    budget_seconds = _env_float("VASQ_TURN_BUDGET_SECONDS", 240)
    started_at = time.monotonic()
    deadline_token = _TURN_DEADLINE.set(started_at + budget_seconds)
    logger.info("VasQ turn started budget=%.1fs", budget_seconds)

    try:
        return _chat_impl(user_input, history, should_stop=should_stop)
    except ChatCancelled:
        logger.info("VasQ turn stopped by user")
        raise
    except Exception:
        logger.exception("VasQ turn failed before a normal response was produced")
        safe_history = history if history is not None else []
        if not any(
            message.get("role") == "user"
            and message.get("content") == user_input
            for message in safe_history[-2:]
        ):
            update_history(safe_history, "user", user_input)
        fallback = (
            "The request could not be completed within the available time. "
            "Please retry; any optional evidence source that is slow will be skipped."
        )
        update_history(safe_history, "assistant", fallback)
        return fallback, safe_history, None
    finally:
        elapsed = time.monotonic() - started_at
        logger.info("VasQ turn finished elapsed=%.1fs", elapsed)
        _TURN_DEADLINE.reset(deadline_token)
