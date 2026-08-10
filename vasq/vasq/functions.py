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
        func_name == "gene_expression"
        and wants_matrix_expression_query(user_input)
        and not wants_marker_query(user_input)
    ):
        logger.info("Overriding model-selected gene_expression -> matrix_expression")
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


def resolve_matrix_entities(user_input):
    ensure_matrix_expression_data_loaded()

    cell_type_matches = resolve_entities_from_text(user_input, MATRIX_CELL_TYPE_ALIAS_MAP)
    cell_class_matches = resolve_entities_from_text(user_input, MATRIX_CELL_CLASS_ALIAS_MAP)
    region_matches = resolve_entities_from_text(user_input, MATRIX_REGION_ALIAS_MAP)
    region_layer_matches = resolve_entities_from_text(user_input, MATRIX_REGION_LAYER_ALIAS_MAP)

    gpt_cell_type_matches, gpt_region_matches = resolve_dataset_entities_with_gpt(
        user_input,
        MATRIX_AVAILABLE_CELL_TYPES,
        MATRIX_AVAILABLE_REGIONS
    )

    cell_type_matches = list(dict.fromkeys(cell_type_matches + gpt_cell_type_matches))
    region_matches = list(dict.fromkeys(region_matches + gpt_region_matches))

    return cell_type_matches, cell_class_matches, region_matches, region_layer_matches

def extract_sex_filters(user_input):
    text = normalize_text(user_input)
    out = []

    if re.search(r"\bfemale\b|\bfemales\b|\bwoman\b|\bwomen\b", text):
        out.append("f")
    if re.search(r"\bmale\b|\bmales\b|\bman\b|\bmen\b", text):
        out.append("m")

    return out


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

    present_genes = [g for g in genes if g in MATRIX_GENE_TO_IDX]
    missing_genes = [g for g in genes if g not in MATRIX_GENE_TO_IDX]

    notes = [
        "This answer uses log-normalized values from the HVG-filtered expression matrix."
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

    all_sections = []
    regional_plot_frames = []

    if len(cell_indices) > 0:
        effective_cell_indices = cell_indices
    else:
        effective_cell_indices = MATRIX_META.index.to_numpy()

    for gene in present_genes:
        # Keep the detailed region/cell-type table used by the text answer.
        stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            ["brain_region", "cell_class", "cell_type"]
        )

        all_sections.append(format_matrix_expression_summary(stats, gene, max_rows=5))

        # Build a stable region-level summary for the visual. Aggregating at
        # the region level prevents n=1 cell subgroups from dominating the
        # chart and lets all requested genes appear in one figure.
        region_stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            ["brain_region"]
        )
        if not region_stats.empty:
            regional_plot_frames.append(region_stats)

    plot_json = None
    if regional_plot_frames:
        plot_stats = pd.concat(regional_plot_frames, ignore_index=True)
        plot_json = build_matrix_expression_plot(
            plot_stats,
            gene_order=present_genes,
            region_order=regions,
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


def summarize_group_expression(gene, cell_indices, group_cols):
    if len(cell_indices) == 0:
        return pd.DataFrame()

    obs = MATRIX_META.iloc[cell_indices][group_cols].copy()
    rows = []

    for key, g in obs.groupby(group_cols):
        idx = g.index.to_numpy()
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


def format_matrix_expression_summary(stats_df, gene, max_rows=5):
    if stats_df.empty:
        return f"No matching cells found for {gene} after applying the requested filters."

    work = stats_df.sort_values(
        ["mean_expr", "pct_expr", "n_cells"],
        ascending=[False, False, False]
    ).head(max_rows)

    lines = [f"Top expression contexts for {gene} (log-normalized mean expression):"]

    for _, row in work.iterrows():
        parts = []
        for col in ["brain_region", "region_layer", "cell_class", "cell_type"]:
            if col in work.columns and pd.notna(row.get(col)):
                parts.append(str(row[col]))

        label = " | ".join(parts) if parts else "all matched cells"
        lines.append(
            f"- {label}: mean_expr {row['mean_expr']:.3f}, "
            f"pct_expr {row['pct_expr']:.3f}, n {int(row['n_cells'])}"
        )

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


def build_expression_plot(df, gene_name=None, max_rows=8, metric="score"):
    plot_df = df.copy()

    if plot_df.empty:
        return None

    if metric not in plot_df.columns:
        metric = "rank"

    if metric == "rank":
        plot_df = plot_df.sort_values("rank", ascending=True).head(max_rows)
        y_vals = plot_df["rank"].tolist()
        y_title = "Rank"
    else:
        plot_df = plot_df.sort_values([metric, "rank"], ascending=[False, True]).head(max_rows)
        y_vals = plot_df[metric].tolist()
        y_title = metric

    labels = [
        f"{pretty_region_name(row['region'])} | {row['cell_type']}"
        for _, row in plot_df.iterrows()
    ]

    hover_text = []
    for _, row in plot_df.iterrows():
        text = f"{row['gene']}<br>{pretty_region_name(row['region'])}<br>{row['cell_type']}"
        text += f"<br>rank {row['rank']}"
        if "score" in plot_df.columns and pd.notna(row.get("score")):
            text += f"<br>score {row['score']:.2f}"
        if "logFC" in plot_df.columns and pd.notna(row.get("logFC")):
            text += f"<br>logFC {row['logFC']:.2f}"
        if "pct_expr" in plot_df.columns and pd.notna(row.get("pct_expr")):
            text += f"<br>pct_expr {row['pct_expr']:.2f}"
        hover_text.append(text)

    title = f"{gene_name} expression across matched regions/cell types" if gene_name else "Expression across matched regions/cell types"

    fig = {
        "data": [
            {
                "type": "bar",
                "x": labels,
                "y": y_vals,
                "text": hover_text,
                "hoverinfo": "text"
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "Region | Cell type"},
            "yaxis": {"title": y_title},
            "margin": {"l": 60, "r": 20, "t": 60, "b": 160}
        }
    }

    return json.dumps(fig)



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

def resolve_dataset_entities_with_gpt(user_input, available_cell_types, available_regions):
    system_prompt = (
        "You are helping map a biology question onto a fixed dataset schema. "
        "Choose the closest matching dataset labels from the provided lists. "
        "Return JSON only with keys: "
        '{"cell_types": [], "regions": []}. '
        "Only use labels that appear in the provided lists. "
        "Do not invent labels."
    )

    user_prompt = (
        f"User query: {user_input}\n\n"
        f"Available cell types: {available_cell_types}\n\n"
        f"Available regions: {available_regions}"
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
        regions = parsed.get("regions", [])

        cell_types = [x for x in cell_types if x in available_cell_types]
        regions = [x for x in regions if x in available_regions]

        return cell_types, regions

    except Exception as e:
        logger.exception("GPT dataset entity resolution failed: %s", e)
        return [], []



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

    # first try deterministic alias matching
    cell_matches = resolve_entities_from_text(user_input, CELL_TYPE_ALIAS_MAP)
    region_matches = resolve_entities_from_text(user_input, REGION_ALIAS_MAP)

    # then let GPT broaden or refine using actual dataset labels
    gpt_cell_matches, gpt_region_matches = resolve_dataset_entities_with_gpt(
        user_input,
        AVAILABLE_CELL_TYPES,
        AVAILABLE_REGIONS
    )

    combined_cell_matches = list(dict.fromkeys(cell_matches + gpt_cell_matches))
    combined_region_matches = list(dict.fromkeys(region_matches + gpt_region_matches))

    logger.info(
        "extract_entities user_input=%s cell_matches=%s region_matches=%s",
        user_input,
        combined_cell_matches,
        combined_region_matches
    )

    return combined_cell_matches, combined_region_matches



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

def format_single_gene_expression_rows(df, max_rows=20):
    lines = []

    for _, row in df.head(max_rows).iterrows():
        cell_type = row.get("cell_type", "Unknown cell type")
        region = row.get("region", "Unknown region")
        gene = row.get("gene", "Unknown gene")

        line = f"{cell_type} in {region}: {gene} (rank {row['rank']}"

        if "score" in df.columns and pd.notna(row.get("score")):
            line += f", score {row['score']:.2f}"
        if "logFC" in df.columns and pd.notna(row.get("logFC")):
            line += f", logFC {row['logFC']:.2f}"
        if "pct_expr" in df.columns and pd.notna(row.get("pct_expr")):
            line += f", pct_expr {row['pct_expr']:.2f}"

        line += ")"
        lines.append(line)

    return "\n".join(lines)


def wants_top_genes(user_input):
    text = user_input.lower()
    triggers = [
        "top 5 genes",
        "top 10 genes",
        "top 20 genes",
        "top markers",
        "top expressed", "highest expressed", "highest expression",
        "top genes", "marker genes", "markers", "most expressed"
    ]
    return any(t in text for t in triggers)


def wants_specific_gene(user_input, genes):
    return len(genes) > 0


def format_gene_rows(df, max_rows=20):
    lines = []
    for _, row in df.head(max_rows).iterrows():
        line = f"{row['gene']} (rank {row['rank']}"
        if "score" in df.columns and pd.notna(row.get("score")):
            line += f", score {row['score']:.2f}"
        if "logFC" in df.columns and pd.notna(row.get("logFC")):
            line += f", logFC {row['logFC']:.2f}"
        if "pct_expr" in df.columns and pd.notna(row.get("pct_expr")):
            line += f", pct_expr {row['pct_expr']:.2f}"
        line += ")"
        lines.append(line)
    return "\n".join(lines)


def pretty_region_name(region):
    region_map = {
        "CP": "choroid plexus",
        "Hip-EC": "hippocampal-entorhinal vasculature",
        "ACA": "anterior cerebral artery",
        "BA.CoW": "basilar artery / circle of Willis",
    }
    return region_map.get(region, region)

def gene_expression(user_input):
    ensure_expression_data_loaded()
    all_regions_flag = all_regions(user_input)
    cell_types, regions = extract_entities(user_input)
    gene_names = extract_genes(user_input)

    base_df = EXPR_DF.copy()

    requested_cell_types = list(cell_types) if cell_types else []
    requested_regions = regions if (regions and not all_regions_flag) else None

    # 1. exact match first
    df = base_df.copy()

    if requested_cell_types:
        df = df[df["cell_type"].isin(requested_cell_types)]

    if requested_regions:
        df = df[df["region"].isin(requested_regions)]

    if gene_names:
        df = df[df["gene"].isin(gene_names)]

    match_note = "exact match"

    # 2. broaden vascular/endothelial labels if exact match is empty
    if df.empty and requested_cell_types:
        expanded_cell_types = list(requested_cell_types)

        if "Endothelial" in requested_cell_types:
            for extra in [
                "Fenestrated Endothelial",
                "Fenestrated Capillary",
                "Fenestrated Capillaries",
                "Capillary",
                "Capillaries",
            ]:
                if extra in AVAILABLE_CELL_TYPES and extra not in expanded_cell_types:
                    expanded_cell_types.append(extra)

        if any(x in requested_cell_types for x in ["Capillary", "Capillaries"]):
            for extra in [
                "Fenestrated Capillary",
                "Fenestrated Capillaries",
                "Fenestrated Endothelial",
                "Endothelial",
                "Pericyte",
            ]:
                if extra in AVAILABLE_CELL_TYPES and extra not in expanded_cell_types:
                    expanded_cell_types.append(extra)

        df = base_df.copy()
        df = df[df["cell_type"].isin(expanded_cell_types)]

        if requested_regions:
            df = df[df["region"].isin(requested_regions)]

        if gene_names:
            df = df[df["gene"].isin(gene_names)]

        if not df.empty:
            match_note = f"expanded cell type match: {expanded_cell_types}"

    logger.info(
        "gene_expression cell_types=%s regions=%s genes=%s rows=%s match_note=%s",
        cell_types, regions, gene_names, len(df), match_note
    )

    if df.empty:
        return "No matching gene expression data found for the specified query."

    prefix = ""
    if match_note != "exact match":
        prefix = f"Using {match_note} because no exact dataset match was found.\n\n"

    # case 1: user asked about specific gene(s)

    if wants_specific_gene(user_input, gene_names):
        if "score" in df.columns:
            df = df.sort_values(["score", "rank"], ascending=[False, True])
        else:
            df = df.sort_values("rank", ascending=True)

        gene_name = gene_names[0] if len(gene_names) == 1 else ", ".join(gene_names)

        text = summarize_single_gene_expression(df, gene_name, max_rows=5)
        graph_json = build_expression_plot(df, gene_name=gene_name, max_rows=8, metric="score")

        return {
           "text": prefix + text,
           "graph_json": graph_json
        }


    # case 2: top genes / markers
    if wants_top_genes(user_input) or not gene_names:
        sort_cols = ["rank"]
        if "score" in df.columns:
            sort_cols = ["rank", "score"]

        df = df.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True)

        group_cols = []
        if cell_types:
            group_cols.append("cell_type")
        if regions and not all_regions_flag:
            group_cols.append("region")

        if not group_cols:
            if cell_types:
                group_cols = ["cell_type"]
            elif regions and not all_regions_flag:
                group_cols = ["region"]
            else:
                group_cols = ["cell_type", "region"]

        sections = []
        grouped = df.groupby(group_cols)

        for key, g in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            key_map = dict(zip(group_cols, key))

            if "cell_type" in key_map and "region" in key_map:
                header = f"Top marker genes for {key_map['cell_type']} in {key_map['region']}"
            elif "cell_type" in key_map:
                header = f"Top marker genes for {key_map['cell_type']}"
            elif "region" in key_map:
                header = f"Top marker genes in {key_map['region']}"
            else:
                header = "Top marker genes"

            sections.append(header)
            sections.append(format_gene_rows(g, max_rows=15))
            sections.append("")

        return prefix + "\n".join(sections).strip()

    return "No matching gene expression data found for the specified query."


def summarize_single_gene_expression(df, gene_name, max_rows=5):
    if df.empty:
        return f"No expression data found for {gene_name}."

    work_df = df.copy()

    if "score" in work_df.columns:
        work_df = work_df.sort_values(["score", "rank"], ascending=[False, True])
    else:
        work_df = work_df.sort_values("rank", ascending=True)

    top_rows = work_df.head(max_rows)

    top = top_rows.iloc[0]
    top_region = pretty_region_name(top.get("region", "unknown region"))
    top_cell_type = top.get("cell_type", "unknown cell type")

    summary = (
        f"{gene_name} is highest in {top_cell_type.lower()} beds, especially in {top_region}. "
    )

    if len(top_rows) > 1:
        second = top_rows.iloc[1]
        second_region = pretty_region_name(second.get("region", "another region"))
        second_cell_type = second.get("cell_type", "another vascular compartment")
        summary += (
            f"In this dataset, the strongest signal appears in {top_cell_type.lower()} of {top_region}, "
            f"with another notable signal in {second_region} {second_cell_type.lower()} compartments"
        )

        if len(top_rows) > 2:
            summary += ", while other vascular compartments are weaker."
        else:
            summary += "."
    else:
        summary += "in this dataset."

    details = []
    for _, row in top_rows.iterrows():
        line = f"- {pretty_region_name(row['region'])}, {row['cell_type']}: rank {row['rank']}"
        if "score" in work_df.columns and pd.notna(row.get("score")):
            line += f", score {row['score']:.2f}"
        if "logFC" in work_df.columns and pd.notna(row.get("logFC")):
            line += f", logFC {row['logFC']:.2f}"
        if "pct_expr" in work_df.columns and pd.notna(row.get("pct_expr")):
            line += f", pct_expr {row['pct_expr']:.2f}"
        details.append(line)

    return summary + "\n\n" + "\n".join(details)




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
        response = requests.post(
            url,
            json={"query": user_input},
            timeout=timeout_seconds,
        )
        response.raise_for_status()

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

    except Exception:
        logger.exception("KG-RAG request failed")
        return None


def build_matrix_expression_plot(stats_df, gene_order=None, region_order=None):
    """Build a gene-by-region dot plot from region-level matrix summaries.

    Marker color represents mean log-normalized expression, while marker size
    represents the fraction of cells with non-zero expression.
    """
    required = {"gene", "brain_region", "mean_expr", "pct_expr", "n_cells"}
    if stats_df.empty or not required.issubset(stats_df.columns):
        return None

    plot_df = stats_df.dropna(subset=["gene", "brain_region"]).copy()
    if plot_df.empty:
        return None

    plot_df["gene"] = plot_df["gene"].astype(str)
    plot_df["brain_region"] = plot_df["brain_region"].astype(str)
    plot_df["mean_expr"] = pd.to_numeric(
        plot_df["mean_expr"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    plot_df["pct_expr"] = pd.to_numeric(
        plot_df["pct_expr"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    plot_df["n_cells"] = pd.to_numeric(
        plot_df["n_cells"], errors="coerce"
    ).fillna(0).astype(int)

    available_genes = plot_df["gene"].drop_duplicates().tolist()
    gene_order = [g for g in (gene_order or []) if g in available_genes]
    gene_order += [g for g in available_genes if g not in gene_order]

    available_regions = plot_df["brain_region"].drop_duplicates().tolist()
    region_order = [r for r in (region_order or []) if r in available_regions]
    region_order += sorted(r for r in available_regions if r not in region_order)

    gene_rank = {gene: i for i, gene in enumerate(gene_order)}
    region_rank = {region: i for i, region in enumerate(region_order)}
    plot_df["_gene_order"] = plot_df["gene"].map(gene_rank)
    plot_df["_region_order"] = plot_df["brain_region"].map(region_rank)
    plot_df = plot_df.sort_values(["_gene_order", "_region_order"])

    def wrap_axis_label(value, width=18):
        words = str(value).split()
        lines = []
        current = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and len(candidate) > width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return "<br>".join(lines)

    region_labels = {
        region: wrap_axis_label(region)
        for region in region_order
    }
    x_values = [region_labels[region] for region in plot_df["brain_region"]]
    y_values = plot_df["gene"].tolist()

    # Square-root scaling keeps low expressing fractions visible without
    # letting highly expressed groups overwhelm the figure.
    marker_sizes = (
        9.0 + 31.0 * np.sqrt(plot_df["pct_expr"].to_numpy())
    ).round(1).tolist()
    marker_opacity = [
        0.45 if n_cells < 20 else 0.9
        for n_cells in plot_df["n_cells"]
    ]

    color_values = plot_df["mean_expr"].tolist()
    positive_colors = plot_df.loc[plot_df["mean_expr"] > 0, "mean_expr"]
    color_max = (
        float(positive_colors.quantile(0.95))
        if not positive_colors.empty
        else 1.0
    )
    color_max = max(color_max, 0.001)

    customdata = [
        [
            row["gene"],
            row["brain_region"],
            float(row["mean_expr"]),
            float(row["pct_expr"]) * 100.0,
            int(row["n_cells"]),
        ]
        for _, row in plot_df.iterrows()
    ]

    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": x_values,
                "y": y_values,
                "customdata": customdata,
                "hovertemplate": (
                    "<b>%{customdata[0]}</b>"
                    "<br>Brain region: %{customdata[1]}"
                    "<br>Mean expression: %{customdata[2]:.3f}"
                    "<br>Expressing cells: %{customdata[3]:.1f}%"
                    "<br>Cells: %{customdata[4]:,}"
                    "<extra></extra>"
                ),
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
                    "<b>VasQ expression across brain regions</b>"
                    "<br><span style='font-size:12px;color:#64748b'>"
                    "Color = mean expression · Size = expressing-cell fraction"
                    "</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            "height": max(440, 170 + 32 * len(gene_order)),
            "autosize": True,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#ffffff",
            "font": {
                "family": "Satoshi, Arial, sans-serif",
                "color": "#32175a",
                "size": 12,
            },
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": "#8ab4c4",
                "font": {"color": "#32175a"},
            },
            "xaxis": {
                "title": {"text": "Brain region", "standoff": 18},
                "categoryorder": "array",
                "categoryarray": [region_labels[r] for r in region_order],
                "tickangle": 0,
                "tickfont": {"size": 10},
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
                "automargin": True,
            },
            "yaxis": {
                "title": {"text": "Gene", "standoff": 12},
                "categoryorder": "array",
                # Plotly orders categorical y values from bottom to top.
                "categoryarray": list(reversed(gene_order)),
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "zeroline": False,
                "automargin": True,
            },
            "margin": {"l": 90, "r": 100, "t": 90, "b": 125},
        },
    }

    return json.dumps(fig)


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
        "name": "gene_expression",
        "description": "Collects information on gene expression rankings by \
        cell type and region in brain vasculature.",
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
            "HVG matrix, with optional filters for brain region, region layer, "
            "cell class, cell type, age at death, and sex."
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



def _chat_impl(user_input, history):
    if history is None:
        history = []

    ensure_chat_initialized(history)

    # Resolve the current turn against prior context before adding it to history.
    intent = analyze_query_intent(user_input, history)
    update_history(history, "user", user_input)

    # Greetings, thanks, casual conversation, and meta questions do not search.
    if not intent.get("is_scientific"):
        direct_message = call_api(
            history,
            stage_name="direct_answer",
            timeout_seconds=_env_float("OPENAI_SYNTHESIS_TIMEOUT_SECONDS", 45),
        )
        final_message = getattr(direct_message, "content", None) or (
            "I'm sorry, but I couldn't generate a response."
        )
        update_history(history, "assistant", final_message)
        return final_message, history, None

    resolved_question = intent.get("resolved_question") or user_input
    genes = intent.get("genes") or []
    diseases = intent.get("diseases") or []

    # Branch A: every scientific question goes through KG-RAG, followed by a
    # function/pathway-oriented Web Search. If KG-RAG is empty or irrelevant,
    # the Web Search uses the resolved original question instead.
    try:
        kg_result = query_kg_rag(resolved_question)
    except Exception:
        logger.exception("KG-RAG branch failed")
        kg_result = None

    kg_assessment = assess_kg_relevance(resolved_question, kg_result)

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

    # The first search can discover genes that were not explicitly written in
    # the user's question (for example, an Alzheimer's disease question). Use
    # that evidence-derived list for both VasQ and the second drug search.
    genes = derive_genes_from_first_search(
        resolved_question,
        scientific_web_result,
        kg_result=kg_result if kg_assessment.get("relevant") else None,
        existing_genes=genes,
    )
    logger.info("Gene list after first search: %s", genes)

    # Branch B: calculate expression for the explicit or first-search-derived
    # gene list. Marker/rank questions use the ranked marker table; measured
    # expression questions use the VasQ matrix.
    vasq_result = None
    vasq_note = ""
    graph_json = None

    if intent.get("asks_expression") and intent.get("use_vasq"):
        if genes or intent.get("asks_markers"):
            vasq_query = resolved_question
            if genes:
                vasq_query += "\nExplicit genes for the VasQ query: " + ", ".join(genes)

            try:
                if intent.get("asks_markers"):
                    vasq_result = gene_expression(vasq_query)
                else:
                    vasq_result = matrix_expression(
                        vasq_query,
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

    vasq_text, graph_json = retrieved_text_and_graph(vasq_result)

    # Branch C: optional second Web Search. Only run it when the user actually
    # asked for drugs/therapeutics; a disease or gene alone is not sufficient.
    drug_result = None
    asks_drugs = bool(intent.get("asks_drugs"))
    if asks_drugs and (genes or diseases):
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

    # Cap each source independently so a long Web result cannot erase VasQ data.
    evidence_parts = []

    if genes:
        evidence_parts.append(
            "GENE LIST DERIVED FROM THE FIRST SEARCH:\n"
            + ", ".join(genes)
        )

    if vasq_text:
        evidence_parts.append(
            "VASQ EXPRESSION OR MARKER DATA:\n"
            + cap_source_text(vasq_text, 15000)
        )
    elif vasq_note:
        evidence_parts.append("VASQ STATUS:\n" + vasq_note)

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

    if asks_drugs and (genes or diseases):
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
            "evidence. Treat knowledge-graph relationships as associations, "
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
            "matrix expression by brain region/cell type with the supplied "
            "measured values. Cover every analyzed gene when evidence is "
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
    logger.info("Final message returned to UI: %r", final_message)
    update_history(history, "assistant", final_message)

    return final_message, history, graph_json


def chat(user_input, history):
    """Run one synchronous chat turn inside a hard wall-clock budget."""
    budget_seconds = _env_float("VASQ_TURN_BUDGET_SECONDS", 240)
    started_at = time.monotonic()
    deadline_token = _TURN_DEADLINE.set(started_at + budget_seconds)
    logger.info("VasQ turn started budget=%.1fs", budget_seconds)

    try:
        return _chat_impl(user_input, history)
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
