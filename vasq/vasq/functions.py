import concurrent.futures
import contextvars
import json
import threading
import openai
import os
import pandas as pd
import re
import requests
import time
import ast
import logging
import numpy as np
from scipy import sparse
from urllib.parse import urlparse

from openai import OpenAI

from .entity_aliases import (
    build_brain_region_alias_map as build_matrix_brain_region_alias_map,
    build_cell_class_alias_map as build_matrix_cell_class_alias_map,
    build_cell_type_alias_map as build_matrix_cell_type_alias_map,
    build_region_layer_alias_map as build_matrix_region_layer_alias_map,
    exclude_classes_implied_by_cell_types,
    exclude_normalized_duplicates,
    merge_hybrid_matches,
    normalize_text,
    resolve_entities_from_text,
    validate_controlled_vocabulary,
)


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
    timeout_seconds=None,
    max_retries=None,
):
    """Call the helper model with parameters compatible with GPT-4o and GPT-5.6.

    `timeout_seconds`/`max_retries` let a specific caller opt into its own
    budget instead of sharing OPENAI_HELPER_TIMEOUT_SECONDS with every other
    helper call in the app. Omit them to keep the previous shared behavior.
    """
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

    requested_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else _env_float("OPENAI_HELPER_TIMEOUT_SECONDS", 30)
    )
    resolved_timeout = _stage_timeout(
        stage_name,
        requested_seconds,
        # Preserve enough time for the final answer even if an optional helper
        # is reached late in the request.
        reserve_seconds=_env_float("VASQ_SYNTHESIS_RESERVE_SECONDS", 50),
    )
    resolved_max_retries = (
        max_retries
        if max_retries is not None
        else _env_int("OPENAI_HELPER_MAX_RETRIES", 1)
    )
    attempt_timeout = max(1.0, resolved_timeout / (resolved_max_retries + 1))
    return client.with_options(
        timeout=attempt_timeout,
        max_retries=resolved_max_retries,
    ).chat.completions.create(**request_args)


logger = logging.getLogger(__name__)


def _source_label(url):
    """A link label mechanically derived from the URL itself -- domain plus
    enough of the path to tell two different pages on the same domain
    apart (e.g. two different NCBI Bookshelf chapters, both under
    ncbi.nlm.nih.gov). Still derived only from the URL string itself, with
    no separate piece of data (a title, an author name) that could drift
    out of sync with the link target -- that drift is exactly what made an
    author/year or page-title label unreliable even when it started out
    correctly paired with its URL.
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc or url
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.strip("/")
    except Exception:
        return url
    if not path:
        return host
    # Keep the label compact: enough of the path to distinguish pages,
    # not the entire thing for a long or deeply nested URL.
    segments = [s for s in path.split("/") if s]
    short_path = "/".join(segments[:2])
    if len(short_path) > 40:
        short_path = short_path[:40].rstrip("/") + "\u2026"
    return f"{host}/{short_path}"


def _splice_web_search_citations(response):
    """Rewrite the model's own web-search text so each url_citation
    annotation becomes a complete, ready-to-use markdown link -- inserted
    immediately after the exact claim it supports, at the position OpenAI's
    own annotation says it applies to.

    Earlier versions of this fix progressively narrowed the failure mode:
    (1) appending a flat URL list at the end of the text still left a
    downstream model to re-derive which URL matched which claim from a
    disconnected list; (2) splicing a bare "[SOURCE: url]" marker inline
    fixed that pairing but still let the model write its own free-text
    citation label (e.g. "Smith et al., 2020"); (3) using the annotation's
    own page title as the label removed the model's free-text step, but a
    title is still a separate piece of data alongside the URL that a
    rewriting model could still touch or replace; (4) using just the URL's
    domain made two different pages on the same host (e.g. two different
    NCBI Bookshelf chapters) look identical.

    This version uses the domain plus a short piece of the URL's own path
    as the visible text -- still mechanically extracted from the URL
    string itself, so the visible text and the link target can never be
    sourced from two different places, but specific enough to tell two
    same-domain sources apart. Less informative than an author/year
    citation, but there is nothing left for a rewriting model to get wrong
    about the pairing.

    Returns (text_with_inline_sources, fallback_urls) -- fallback_urls are
    sources the tool fetched but never tied to a specific claim (from
    web_search_call.action.sources), for a short "also consulted" note.
    Falls back to the unmodified `response.output_text` with no inline
    links if the expected structure isn't present; never raises.
    """
    fallback_urls = []
    pieces = []
    try:
        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for content in getattr(item, "content", None) or []:
                    if getattr(content, "type", None) != "output_text":
                        continue
                    text = getattr(content, "text", None) or ""
                    annotations = [
                        a
                        for a in (getattr(content, "annotations", None) or [])
                        if getattr(a, "type", None) == "url_citation"
                        and getattr(a, "url", None)
                        and getattr(a, "start_index", None) is not None
                        and getattr(a, "end_index", None) is not None
                    ]
                    # Splice from the last position backwards so an
                    # earlier annotation's start/end offsets are never
                    # invalidated by a later insertion shifting the text.
                    annotations.sort(key=lambda a: a.end_index, reverse=True)
                    spliced = text
                    for annotation in annotations:
                        end = annotation.end_index
                        url = annotation.url
                        # OpenAI's web_search generation often already
                        # embeds its own markdown link to the cited URL
                        # directly in the text, right around where the
                        # annotation says it applies -- unconditionally
                        # splicing another one in would duplicate it next
                        # to the model's own. Check a window of the
                        # *original* text (not `spliced`, which may already
                        # have later-position markers inserted) around this
                        # position for an existing link to this exact URL.
                        window_start = max(0, end - 300)
                        window_end = min(len(text), end + 50)
                        if f"]({url})" in text[window_start:window_end]:
                            continue
                        marker = f" [{_source_label(url)}]({url})"
                        spliced = spliced[:end] + marker + spliced[end:]
                    pieces.append(spliced)

            elif item_type == "web_search_call":
                action = getattr(item, "action", None)
                if getattr(action, "type", None) == "search":
                    for source in getattr(action, "sources", None) or []:
                        url = getattr(source, "url", None)
                        if url:
                            fallback_urls.append(url)
    except Exception:
        logger.exception("Could not splice inline web-search citations")
        pieces = []

    if pieces:
        text_with_inline_sources = "\n".join(p for p in pieces if p.strip())
    else:
        text_with_inline_sources = str(getattr(response, "output_text", "") or "").strip()

    seen = set()
    unique_fallback = []
    for url in fallback_urls:
        if url in seen:
            continue
        seen.add(url)
        unique_fallback.append(url)

    return text_with_inline_sources, unique_fallback


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

        result, fallback_urls = _splice_web_search_citations(response)

        if not result:
            logger.warning("OpenAI Web Search returned no text")
            return None

        if fallback_urls:
            result += (
                "\n\nOther sources the search tool consulted, not tied to a "
                "specific claim above (use only if you independently need "
                "one; do not attach these to a claim that already has its "
                "own inline markdown link):\n"
                + "\n".join(f"- {url}" for url in fallback_urls)
            )

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


_STRICT_SOURCE_INSTRUCTION = (
    "Only cite peer-reviewed literature, PubMed/PMC, official gene or "
    "protein databases (e.g. Human Protein Atlas, NCBI Gene, UniProt), "
    "FDA, ClinicalTrials.gov, or other primary/authoritative scientific "
    "sources. Do not cite blogs, forums, news aggregators, general "
    "encyclopedias, or any page whose authority you cannot verify from "
    "its content, even if it appears in search results. If you cannot "
    "identify a specific, reliable source for a claim, state the claim "
    "as uncited (e.g. \"widely described in the literature\") rather "
    "than attaching an approximate, uncertain, or best-guess source to "
    "it -- an unlabeled claim is preferable to a confidently wrong "
    "citation. Every citation must trace to a page you actually "
    "retrieved in this search, never one recalled from general "
    "training knowledge. "
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
        "is not available, say so instead of inferring it. For each gene "
        "discussed, also state which cell type(s) the literature identifies "
        "it with -- for example as a canonical marker gene, from single-cell "
        "or tissue-atlas expression data, or from a cell-type-specific "
        "functional role -- and cite the supporting source. This will be "
        "cross-checked against measured single-cell expression data, so be "
        "explicit about which cell type(s) are reported and say so plainly "
        "if the literature does not establish a cell-type association. "
    )

    return run_openai_web_search(
        "Search the live web to answer this biomedical question. "
        + pathway_instruction
        + _STRICT_SOURCE_INSTRUCTION
        + "Provide source citations and distinguish established evidence from "
        "hypotheses.\n\n"
        + context_block
        + f"Question: {user_input}",
        stage_name="scientific_web_search",
        search_context_size=os.getenv(
            "OPENAI_SCIENTIFIC_SEARCH_CONTEXT_SIZE", "high"
        ),
    )


def search_gene_literature_evidence(user_input, genes, diseases=None):
    """Retrieve compact, cited evidence for every prioritized expression gene.

    The primary search runs before the final gene list exists, so it is good at
    discovering candidate genes but cannot reliably guarantee equal literature
    coverage for every gene ultimately selected. This second, bounded search
    uses the finalized list and asks for one compact evidence record per gene.
    """
    genes = list(dict.fromkeys(
        str(gene).upper().strip()
        for gene in (genes or [])
        if str(gene).strip()
    ))
    diseases = list(dict.fromkeys(
        str(disease).strip()
        for disease in (diseases or [])
        if str(disease).strip()
    ))

    if not genes:
        return None

    disease_context = (
        ", ".join(diseases)
        if diseases
        else "the disease or biological context in the original question"
    )

    return run_openai_web_search(
        (
            "Search the live scientific web for gene-by-gene evidence for "
            "the exact prioritized human genes below. Cover every listed "
            "gene in the supplied order and do not add unrelated genes. For "
            "each gene, use a heading exactly formatted as 'GENE: SYMBOL', "
            "followed by two concise evidence bullets: (1) its relationship "
            f"to {disease_context}, clearly labeled as causal, genetic-risk, "
            "associated, mechanistic, or uncertain; and (2) any reliable "
            "brain or cell-type association relevant to interpreting a "
            "single-nucleus expression matrix. Put at least one inline "
            "source citation immediately after the supported statement for "
            "each gene, preferably a gene-specific peer-reviewed paper, "
            "PubMed/PMC record, major genetics consortium, or authoritative "
            "atlas/database entry. Use no more than two sources per gene. "
            "If reliable gene-specific evidence or cell-type evidence cannot "
            "be found, state that limitation under that gene instead of "
            "omitting the gene or borrowing a citation from another gene. "
            "Keep the result compact enough that all genes retain coverage. "
            + _STRICT_SOURCE_INSTRUCTION
            + "\n\nExact genes: "
            + ", ".join(genes)
            + f"\n\nOriginal question: {user_input}"
        ),
        stage_name="gene_literature_web_search",
        search_context_size=os.getenv(
            "OPENAI_GENE_LITERATURE_CONTEXT_SIZE", "high"
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
            + _STRICT_SOURCE_INSTRUCTION
            + f"\nQuestion: {user_input}"
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
        "and authoritative company trial records. "
        + _STRICT_SOURCE_INSTRUCTION
        + "Use current information, "
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


### Gene Expression Functions ###
DATA_DIR = "/data"
MATRIX_NPZ_PATH = os.path.join(DATA_DIR, "VasQ_adata_X_sparse.npz")
CELL_META_PATH = os.path.join(DATA_DIR, "VasQ_cell_meta_table.csv")
GENE_NAMES_PATH = os.path.join(DATA_DIR, "VasQ_gene_names.csv")

# Guards the lazy-load below so two concurrent requests hitting a cold
# cache at the same time can't both start loading the (large) matrix data.
_MATRIX_LOAD_LOCK = threading.Lock()

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

# Keep multi-gene answers readable and inexpensive to render. Every gene still
# receives a text summary; only the three strongest overall expression
# profiles receive Plotly figures.
MAX_PLOTTED_GENES = 3


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

    with _MATRIX_LOAD_LOCK:
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
            MATRIX_CELL_TYPE_ALIAS_MAP = build_matrix_cell_type_alias_map(
                MATRIX_AVAILABLE_CELL_TYPES
            )

        if MATRIX_CELL_CLASS_ALIAS_MAP is None:
            MATRIX_CELL_CLASS_ALIAS_MAP = build_matrix_cell_class_alias_map(
                MATRIX_AVAILABLE_CELL_CLASSES
            )

        if MATRIX_REGION_ALIAS_MAP is None:
            MATRIX_REGION_ALIAS_MAP = build_matrix_brain_region_alias_map(
                MATRIX_AVAILABLE_REGIONS
            )

        if MATRIX_REGION_LAYER_ALIAS_MAP is None:
            MATRIX_REGION_LAYER_ALIAS_MAP = build_matrix_region_layer_alias_map(
                MATRIX_AVAILABLE_REGION_LAYERS
            )


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


def resolve_matrix_entities(
    user_input,
    *,
    include_interpretations=False,
    interpretation_input=None,
):
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

    # `user_input` may be the intent helper's context-resolved rewrite. Keep
    # the literal current-turn wording separately for disclosure, otherwise a
    # hidden rewrite from "memory-related region" to "Hippocampus" makes the
    # mapping look as though the user explicitly requested Hippocampus.
    disclosure_input = interpretation_input or user_input
    if disclosure_input == user_input:
        disclosure_cell_type_matches = local_cell_type_matches
        disclosure_cell_class_matches = local_cell_class_matches
        disclosure_region_matches = local_region_matches
        disclosure_region_layer_matches = local_region_layer_matches
    else:
        disclosure_cell_type_matches = resolve_entities_from_text(
            disclosure_input, MATRIX_CELL_TYPE_ALIAS_MAP
        )
        disclosure_cell_class_matches = resolve_entities_from_text(
            disclosure_input, MATRIX_CELL_CLASS_ALIAS_MAP
        )
        disclosure_region_matches = resolve_entities_from_text(
            disclosure_input, MATRIX_REGION_ALIAS_MAP
        )
        disclosure_region_layer_matches = resolve_entities_from_text(
            disclosure_input, MATRIX_REGION_LAYER_ALIAS_MAP
        )

    # cell_type/cell_class are one axis (a specific type already implies its
    # class); brain_region/region_layer are the other axis (a named region
    # already implies its coarse layer bucket, and vice versa). GPT is only
    # worth consulting for an axis where *neither* member resolved locally.
    # If either member of a pair already hit, that whole axis is trusted and
    # GPT's opinion on it is discarded even when the shared call still
    # returns one -- local never gets overridden or duplicated.
    cell_axis_resolved = bool(local_cell_type_matches) or bool(local_cell_class_matches)
    region_axis_resolved = bool(local_region_matches) or bool(local_region_layer_matches)
    disclosure_cell_axis_resolved = (
        bool(disclosure_cell_type_matches)
        or bool(disclosure_cell_class_matches)
    )
    disclosure_region_axis_resolved = (
        bool(disclosure_region_matches)
        or bool(disclosure_region_layer_matches)
    )

    gpt_filter_interpretations = []

    if cell_axis_resolved and region_axis_resolved:
        gpt_cell_type_matches = None
        gpt_cell_class_matches = None
        gpt_region_matches = None
        gpt_region_layer_matches = None
    else:
        (
            gpt_cell_type_matches,
            gpt_cell_class_matches,
            gpt_region_matches,
            gpt_region_layer_matches,
            gpt_filter_interpretations,
        ) = resolve_dataset_entities_with_gpt(
            user_input,
            MATRIX_AVAILABLE_CELL_TYPES,
            MATRIX_AVAILABLE_REGIONS,
            available_cell_classes=MATRIX_AVAILABLE_CELL_CLASSES,
            available_region_layers=MATRIX_AVAILABLE_REGION_LAYERS,
            include_interpretations=True,
        )
        if cell_axis_resolved:
            gpt_cell_type_matches = None
            gpt_cell_class_matches = None
            gpt_filter_interpretations = [
                item
                for item in gpt_filter_interpretations
                if isinstance(item, dict)
                and item.get("dimension") not in {"cell_type", "cell_class"}
            ]
        if region_axis_resolved:
            gpt_region_matches = None
            gpt_region_layer_matches = None
            gpt_filter_interpretations = [
                item
                for item in gpt_filter_interpretations
                if isinstance(item, dict)
                and item.get("dimension") not in {"brain_region", "region_layer"}
            ]

    # Hybrid resolution: curated deterministic aliases protect common dataset
    # terminology from an LLM omission, while the LLM covers unenumerated
    # natural-language variants. Every merged value is validated against the
    # current matrix vocabulary before it can become a filter.
    cell_type_matches = merge_hybrid_matches(
        local_cell_type_matches,
        gpt_cell_type_matches,
        MATRIX_AVAILABLE_CELL_TYPES,
        dimension_name="cell_type",
    )
    cell_class_matches = merge_hybrid_matches(
        local_cell_class_matches,
        gpt_cell_class_matches,
        MATRIX_AVAILABLE_CELL_CLASSES,
        dimension_name="cell_class",
    )
    region_matches = merge_hybrid_matches(
        local_region_matches,
        gpt_region_matches,
        MATRIX_AVAILABLE_REGIONS,
        dimension_name="brain_region",
    )
    region_layer_matches = merge_hybrid_matches(
        local_region_layer_matches,
        gpt_region_layer_matches,
        MATRIX_AVAILABLE_REGION_LAYERS,
        dimension_name="region_layer",
    )

    # Cross-dimension reconciliation. merge_hybrid_matches above only
    # combines local+GPT matches within a single dimension; it cannot see
    # that a resolved cell_type already pins down (and makes redundant) a
    # cell_class match found independently in the same query.
    cell_class_matches = exclude_normalized_duplicates(
        cell_class_matches, cell_type_matches
    )
    cell_class_matches = exclude_classes_implied_by_cell_types(
        cell_class_matches, cell_type_matches
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

    filter_interpretations = normalize_filter_interpretations(
        disclosure_input,
        gpt_filter_interpretations,
        cell_types=cell_type_matches,
        cell_classes=cell_class_matches,
        regions=region_matches,
        region_layers=region_layer_matches,
        semantic_dimensions={
            *(
                []
                if disclosure_cell_axis_resolved
                else ["cell_type", "cell_class"]
            ),
            *(
                []
                if disclosure_region_axis_resolved
                else ["brain_region", "region_layer"]
            ),
        },
    )

    logger.info(
        "Matrix filters resolved cell_types=%s cell_classes=%s "
        "brain_regions=%s region_layers=%s",
        cell_type_matches,
        cell_class_matches,
        region_matches,
        region_layer_matches,
    )

    result = (
        cell_type_matches,
        cell_class_matches,
        region_matches,
        region_layer_matches,
    )
    if include_interpretations:
        return (*result, filter_interpretations)
    return result


def _semantic_source_phrase(user_input, dimension):
    """Recover a concise functional phrase when the helper omitted it."""
    if dimension in {"brain_region", "region_layer"}:
        match = re.search(
            r"\b(?:memory|learning|emotion|motor|visual|auditory|language|"
            r"executive|reward|sleep)[-\s]+related(?:\s+(?:brain\s+)?regions?)?\b",
            user_input,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0).strip()
    return ""


def _default_filter_interpretation_reason(source_phrase, dimension, labels):
    phrase_norm = normalize_text(source_phrase)
    label_norms = {normalize_text(label) for label in labels}
    if (
        dimension == "brain_region"
        and "memory related" in phrase_norm
        and "hippocampus" in label_norms
    ):
        return (
            "the hippocampus is a canonical memory-related structure and "
            "is the closest matching brain-region label available in VasQ"
        )
    dimension_label = {
        "brain_region": "brain-region",
        "region_layer": "region-layer",
        "cell_type": "cell-type",
        "cell_class": "cell-class",
    }.get(dimension, "dataset")
    return (
        f"this was the closest supported match in the available VasQ "
        f"{dimension_label} vocabulary"
    )


def normalize_filter_interpretations(
    user_input,
    raw_interpretations,
    *,
    cell_types=None,
    cell_classes=None,
    regions=None,
    region_layers=None,
    semantic_dimensions=None,
):
    """Validate semantic query-to-dataset mappings returned by the helper.

    Only GPT-only matches need disclosure: literal labels and curated aliases
    have already been resolved deterministically and do not represent a hidden
    biological interpretation. A fallback entry is created when the helper
    resolved a functional phrase but omitted its explanation.
    """
    allowed_by_dimension = {
        "cell_type": list(cell_types or []),
        "cell_class": list(cell_classes or []),
        "brain_region": list(regions or []),
        "region_layer": list(region_layers or []),
    }
    semantic_dimensions = set(
        allowed_by_dimension
        if semantic_dimensions is None
        else semantic_dimensions
    )
    dimension_aliases = {
        "region": "brain_region",
        "brain region": "brain_region",
        "brain_region": "brain_region",
        "region layer": "region_layer",
        "region_layer": "region_layer",
        "cell type": "cell_type",
        "cell_type": "cell_type",
        "cell class": "cell_class",
        "cell_class": "cell_class",
    }
    query_norm = normalize_text(user_input)
    cleaned = []

    for raw in raw_interpretations or []:
        if not isinstance(raw, dict):
            continue
        raw_dimension = normalize_text(str(raw.get("dimension", "")))
        dimension = dimension_aliases.get(raw_dimension)
        if dimension not in semantic_dimensions:
            continue

        allowed = allowed_by_dimension.get(dimension, [])
        raw_labels = raw.get("labels") or []
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        labels = validate_controlled_vocabulary(
            raw_labels,
            allowed,
            dimension_name=dimension,
        )
        if not labels:
            continue

        source_phrase = str(raw.get("source_phrase", "")).strip()
        if source_phrase and normalize_text(source_phrase) not in query_norm:
            source_phrase = ""
        source_phrase = source_phrase or _semantic_source_phrase(
            user_input, dimension
        )
        if not source_phrase:
            source_phrase = "the functional description in the question"

        reason = re.sub(
            r"\s+",
            " ",
            str(raw.get("rationale", "")).strip(),
        )[:360]
        reason = re.sub(r"^(?:because|as)\s+", "", reason, flags=re.IGNORECASE)
        reason = re.sub(r"^The\s+", "the ", reason)
        reason = re.sub(r"^This\s+", "this ", reason)
        reason = reason.rstrip(". ") or _default_filter_interpretation_reason(
            source_phrase, dimension, labels
        )
        cleaned.append({
            "dimension": dimension,
            "source_phrase": source_phrase,
            "labels": labels,
            "rationale": reason,
        })

    # The structured explanation is optional model output, but disclosing a
    # semantic filter is not optional. Create a deterministic fallback for a
    # resolved semantic axis when necessary.
    covered_dimensions = {item["dimension"] for item in cleaned}
    for dimension in ["brain_region", "region_layer", "cell_type", "cell_class"]:
        labels = allowed_by_dimension.get(dimension, [])
        if (
            dimension not in semantic_dimensions
            or not labels
            or dimension in covered_dimensions
        ):
            continue
        source_phrase = _semantic_source_phrase(user_input, dimension)
        if not source_phrase:
            source_phrase = "the functional description in the question"
        cleaned.append({
            "dimension": dimension,
            "source_phrase": source_phrase,
            "labels": labels,
            "rationale": _default_filter_interpretation_reason(
                source_phrase, dimension, labels
            ),
        })

    return cleaned


def format_filter_interpretation_evidence(interpretations):
    """Create compact, explicit evidence for the final synthesis model."""
    if not interpretations:
        return ""
    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_type": "Cell type",
        "cell_class": "Cell class",
    }
    lines = [
        "SEMANTIC FILTER INTERPRETATION — explain this before expression results:"
    ]
    for item in interpretations:
        labels = ", ".join(item["labels"])
        lines.append(
            f"- {display_names.get(item['dimension'], item['dimension'])}: "
            f'The user phrase "{item["source_phrase"]}" was operationalized '
            f"as {labels} because {item['rationale']}."
        )
    lines.append(
        "This is an operational mapping to the available VasQ vocabulary, "
        "not a claim that the selected value is the only biological structure "
        "related to the broader concept."
    )
    return "\n".join(lines)


def build_filter_interpretation_section(interpretations):
    """Render the user-facing explanation guaranteed to precede results."""
    if not interpretations:
        return ""
    display_names = {
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_type": "Cell type",
        "cell_class": "Cell class",
    }
    lines = ["### How the request was interpreted", ""]
    has_region_mapping = False
    for item in interpretations:
        labels = ", ".join(f"**{label}**" for label in item["labels"])
        dimension_name = display_names.get(item["dimension"], item["dimension"])
        lines.append(
            f'- **{dimension_name}:** I interpreted '
            f'“{item["source_phrase"]}” as {labels} because '
            f'{item["rationale"]}. The VasQ calculation therefore used '
            f'`{dimension_name} = {", ".join(item["labels"])}`.'
        )
        has_region_mapping |= item["dimension"] in {"brain_region", "region_layer"}
    if has_region_mapping:
        lines.extend([
            "",
            (
                "This is an operational mapping to the regions available in "
                "VasQ; it does not mean that the selected region is the only "
                "brain region involved in the broader function."
            ),
        ])
    return "\n".join(lines)


def ensure_filter_interpretation_visible(final_message, interpretations):
    """Prepend the mapping if synthesis omitted the required explanation."""
    if not interpretations:
        return final_message
    first_part = normalize_text((final_message or "")[:1800])
    has_explanation_language = any(
        term in first_part
        for term in ["interpreted", "operationalized", "mapped", "closest matching"]
    )
    mentions_every_phrase = all(
        normalize_text(item["source_phrase"]) in first_part
        for item in interpretations
        if item.get("source_phrase")
    )
    mentions_every_label = all(
        all(normalize_text(label) in first_part for label in item.get("labels", []))
        for item in interpretations
    )
    if has_explanation_language and mentions_every_phrase and mentions_every_label:
        return final_message
    section = build_filter_interpretation_section(interpretations)
    return section + "\n\n" + str(final_message or "").lstrip()

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

    if not group_cols:
        # Nothing was asked about at all; keep the original two-axis default.
        group_cols = ["brain_region", "cell_type"]
    elif not asks_cell_types and not asks_cell_classes:
        # Region and/or region_layer was asked about, but the cell dimension
        # was never mentioned either way. Still break results out by
        # cell_type -- a bare "which regions is this expressed in" question
        # should not silently pool every cell type into one number.
        group_cols.append("cell_type")

    return group_cols


def matrix_expression(
    user_input,
    genes_override=None,
    web_evidence_text=None,
    kg_evidence_text=None,
    original_user_input=None,
):
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

    present_genes = [g for g in genes if g in MATRIX_GENE_TO_IDX]
    missing_genes = [g for g in genes if g not in MATRIX_GENE_TO_IDX]

    (
        cell_types,
        cell_classes,
        regions,
        region_layers,
        filter_interpretations,
    ) = resolve_matrix_entities(
        user_input,
        include_interpretations=True,
        interpretation_input=original_user_input,
    )

    # Literature evidence must not silently narrow the matrix scope. The
    # literature and VasQ comparison is only meaningful when VasQ is allowed to
    # evaluate every cell type/region inside the user's explicit filters. In an
    # earlier implementation, a literature statement such as "APP is neuronal"
    # could become a Neuron-only matrix filter, making it impossible to test
    # whether Neuron was actually the highest VasQ cell type. Keep literature
    # available for final synthesis, but derive matrix filters only from the
    # user's question.
    web_filter_notes = []

    group_cols = requested_matrix_group_columns(
        user_input,
        cell_types=cell_types,
        cell_classes=cell_classes,
        regions=regions,
        region_layers=region_layers,
    )

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

    filter_interpretation_evidence = format_filter_interpretation_evidence(
        filter_interpretations
    )
    if filter_interpretation_evidence:
        notes.append(filter_interpretation_evidence)

    notes.extend(web_filter_notes)

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

    gene_stats = {}
    gene_plot_scores = {}
    gene_sections = {}
    regional_plot_frames = []

    if len(cell_indices) > 0:
        effective_cell_indices = cell_indices
    else:
        effective_cell_indices = MATRIX_META.index.to_numpy()

    # With many genes queried together, the per-gene text below is later
    # joined and hard-capped by length downstream (cap_source_text on the
    # full result["text"], applied in _chat_impl before synthesis). A fixed
    # 40-row table per gene lets the first few genes consume that entire
    # budget, silently cutting every later gene's section down to nothing
    # -- the synthesis model then has no way to know that gene's data
    # exists at all and reports it as "not supplied", even though it was
    # measured and is sitting right here. Shrinking the per-gene budget as
    # the gene count grows keeps every gene represented, even if each one's
    # detail is smaller.
    gene_count = max(1, len(present_genes))
    if gene_count <= 2:
        rows_per_gene, cell_types_per_gene, coverage_values_per_gene = 12, 12, 12
    elif gene_count <= 5:
        rows_per_gene, cell_types_per_gene, coverage_values_per_gene = 6, 8, 8
    elif gene_count <= 10:
        rows_per_gene, cell_types_per_gene, coverage_values_per_gene = 3, 6, 6
    else:
        rows_per_gene, cell_types_per_gene, coverage_values_per_gene = 1, 4, 4

    for gene in present_genes:
        # Rank plot candidates by expression across every matched cell,
        # including zero values. This is more stable than selecting genes by
        # one potentially extreme region x cell-type group.
        gene_values = get_gene_vector(effective_cell_indices, gene)
        gene_plot_scores[gene] = (
            float(gene_values.mean()) if len(gene_values) else 0.0,
            float((gene_values > 0).mean()) if len(gene_values) else 0.0,
        )

        # Keep every requested comparison dimension separate. This prevents,
        # for example, Layer 2 and Layer 3 from being pooled into one mean.
        stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            group_cols,
            min_cells=MIN_CELLS_PER_GROUP,
        )

        gene_stats[gene] = stats
        # This is a raw-cell-weighted aggregation across all matched regions,
        # not an unweighted average of region-level means. It is therefore the
        # correct VasQ quantity to compare with a literature statement about a
        # gene's overall cell-type association.
        cell_type_stats = summarize_group_expression(
            gene,
            effective_cell_indices,
            ["cell_type"],
            min_cells=MIN_CELLS_PER_GROUP,
        )
        gene_sections[gene] = format_matrix_expression_summary(
            stats,
            gene,
            group_cols=group_cols,
            cell_type_stats=cell_type_stats,
            max_rows=rows_per_gene,
            max_cell_type_rows=cell_types_per_gene,
            max_coverage_values=coverage_values_per_gene,
        )

        # Preserve every eligible group for plotting. The compact text table
        # below is capped for readability, but the graph payload is separate
        # from the LLM evidence text and therefore does not consume its token
        # budget.
        if not stats.empty:
            regional_plot_frames.append(stats)

    plot_json = None
    all_stats = (
        pd.concat(regional_plot_frames, ignore_index=True)
        if regional_plot_frames
        else pd.DataFrame()
    )
    if not all_stats.empty:
        plot_group_cols = list(group_cols)
        plot_genes = select_top_expression_genes(
            present_genes,
            gene_plot_scores,
            max_genes=MAX_PLOTTED_GENES,
        )

        if "cell_type" in group_cols:
            # One region x cell-type figure per gene is much easier to read
            # than flattening gene x region x cell type into one axis. Keep
            # the complete eligible statistics in each figure; only the text
            # summary is capped.
            plot_group_cols = ["brain_region"]
            if "region_layer" in group_cols:
                plot_group_cols.append("region_layer")
            plot_group_cols.append("cell_type")

            plot_payloads = []
            for gene in plot_genes:
                if plot_group_cols == list(group_cols):
                    plot_stats = gene_stats.get(gene, pd.DataFrame())
                else:
                    plot_stats = summarize_group_expression(
                        gene,
                        effective_cell_indices,
                        plot_group_cols,
                        min_cells=MIN_CELLS_PER_GROUP,
                    )
                if plot_stats.empty:
                    continue
                gene_plot_json = build_matrix_expression_plot(
                    plot_stats,
                    gene_order=[gene],
                    comparison_cols=plot_group_cols,
                )
                if gene_plot_json:
                    plot_payloads.append(json.loads(gene_plot_json))

            if plot_payloads:
                plot_json = (
                    plot_payloads[0]
                    if len(plot_payloads) == 1
                    else plot_payloads
                )
                notes.append(
                    "Separate region × cell type plots are provided for the "
                    f"top {len(plot_genes)} genes by overall mean expression "
                    "across all matched cells (including zeros): "
                    + ", ".join(plot_genes)
                    + ". Every analyzed gene remains in the text summary; "
                    "plot data are not included in the LLM text budget."
                )
        else:
            plot_stats = all_stats[all_stats["gene"].isin(plot_genes)].copy()
            plot_json = build_matrix_expression_plot(
                plot_stats,
                gene_order=plot_genes,
                comparison_cols=plot_group_cols,
            )
            if plot_json:
                notes.append(
                    f"The plot is limited to the top {len(plot_genes)} genes "
                    "by overall mean expression across all matched cells "
                    "(including zeros): "
                    + ", ".join(plot_genes)
                    + ". Every analyzed gene remains in the text summary."
                )

    expression_table = build_expression_table_payload(
        all_stats,
        gene_order=present_genes,
        group_cols=group_cols,
    )

    # Cap each gene independently so no early gene can consume the entire
    # synthesis allowance and erase later genes. The deterministic global
    # maximum and cross-region cell-type ranking are placed first in each
    # section, so optional detail rows are what get trimmed first.
    per_gene_text_limit = max(600, 12000 // gene_count)
    all_sections = [
        cap_text_at_line_boundary(
            gene_sections[g],
            per_gene_text_limit,
            suffix="[Additional per-gene detail omitted from synthesis text]",
        )
        for g in present_genes
        if g in gene_sections
    ]

    return {
        "text": "\n\n".join(notes + [""] + all_sections),
        "graph_json": plot_json,
        "expression_table": expression_table,
        "filter_interpretations": filter_interpretations,
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


def select_top_expression_genes(gene_order, gene_scores, max_genes=3):
    """Select plot genes by stable, whole-subset expression measurements."""
    eligible = [gene for gene in gene_order if gene in gene_scores]
    ranked = sorted(
        eligible,
        key=lambda gene: gene_scores[gene],
        reverse=True,
    )
    return ranked[:max(0, int(max_genes))]


def build_expression_table_payload(stats_df, gene_order, group_cols):
    """Serialize every eligible expression group for UI paging and CSV.

    This payload is returned separately from the compact evidence text, so the
    complete table never consumes synthesis tokens. The view stores it under
    the request ID and serves only one filtered page to the browser at a time.
    """
    if stats_df is None or stats_df.empty:
        return None

    display_names = {
        "gene": "Gene",
        "brain_region": "Brain region",
        "region_layer": "Region layer",
        "cell_class": "Cell class",
        "cell_type": "Cell type",
        "mean_expr": "Mean expression (log-normalized)",
        "pct_expr": "Expressing cells (%)",
        "n_cells": "Cells analyzed (n)",
    }
    dimension_cols = [
        col
        for col in (group_cols or [])
        if col in stats_df.columns
    ]
    column_keys = [
        "gene",
        *dimension_cols,
        "mean_expr",
        "pct_expr",
        "n_cells",
    ]
    gene_rank = {
        str(gene): rank
        for rank, gene in enumerate(gene_order or [])
    }
    work = stats_df.copy()
    work["_gene_rank"] = work["gene"].map(gene_rank).fillna(len(gene_rank))
    work = work.sort_values(
        ["_gene_rank", "mean_expr", "pct_expr", "n_cells"],
        ascending=[True, False, False, False],
    )

    rows = []
    for _, row in work.iterrows():
        record = {"gene": str(row.get("gene", ""))}
        for col in dimension_cols:
            value = row.get(col, "")
            record[col] = "" if pd.isna(value) else str(value)
        record["mean_expr"] = round(float(row["mean_expr"]), 6)
        record["pct_expr"] = round(100.0 * float(row["pct_expr"]), 2)
        record["n_cells"] = int(row["n_cells"])
        rows.append(record)

    filter_keys = [
        key
        for key in [
            "gene",
            "brain_region",
            "region_layer",
            "cell_class",
            "cell_type",
        ]
        if key in column_keys
    ]
    filters = {
        key: sorted({str(row[key]) for row in rows if str(row.get(key, ""))})
        for key in filter_keys
    }

    return {
        "columns": [
            {"key": key, "label": display_names[key]}
            for key in column_keys
        ],
        "filter_keys": filter_keys,
        "filters": filters,
        "rows": rows,
        "total_rows": len(rows),
        "minimum_cells_per_group": MIN_CELLS_PER_GROUP,
    }


def select_balanced_expression_rows(
    stats_df,
    group_cols,
    max_rows=40,
):
    """Keep requested comparison values represented in a capped table.

    A global top-N by mean expression can accidentally retain only Cortex
    and hide White Matter Tracts -- or, just as importantly, retain only
    whichever single cell type happens to have the single highest value in
    almost every region, silently excluding every other cell type despite
    the table nominally being broken out by both.

    An earlier version filled the row budget column by column: every
    observed value of the first priority column, then every value of the
    second, and so on. A column with many observed values (brain_region
    routinely has ~40) could exhaust the entire budget before a column
    with fewer values (cell_type, cell_class) ever got a single guaranteed
    row -- e.g. a gene whose single highest-expressing cell type in nearly
    every region is the same one would fill all 40 rows with just that one
    cell type, silently dropping every other cell type from the table even
    though the surrounding text discusses them by name.

    This version round-robins across columns instead: each column
    contributes at most one new row per pass before moving to the next
    column, cycling until the budget is full. Every requested dimension
    gets a fair, interleaved share of the row budget regardless of how
    many distinct values it has.
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

    chosen_set = set()
    value_queues = {
        col: list(ranked[col].drop_duplicates()) for col in priority_cols
    }

    progressed = True
    while len(chosen_set) < max_rows and progressed:
        progressed = False
        for col in priority_cols:
            queue = value_queues[col]
            picked = None
            while queue:
                value = queue.pop(0)
                candidates = ranked[ranked[col] == value]
                for idx in candidates.index:
                    if idx not in chosen_set:
                        picked = idx
                        break
                if picked is not None:
                    break
                # Every row for this value was already claimed by another
                # column's earlier turn; try this column's next value
                # within the same round instead of skipping its turn.
            if picked is not None:
                chosen_set.add(picked)
                progressed = True
            if len(chosen_set) >= max_rows:
                break

    if len(chosen_set) < max_rows:
        for idx in ranked.index:
            if idx not in chosen_set:
                chosen_set.add(idx)
            if len(chosen_set) >= max_rows:
                break

    # Return in the original rank order, not round-robin insertion order.
    chosen_rank_order = [idx for idx in ranked.index if idx in chosen_set]
    return ranked.loc[chosen_rank_order]


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
    cell_type_stats=None,
    max_rows=40,
    max_cell_type_rows=8,
    max_coverage_values=30,
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

    ranked = stats_df.sort_values(
        ["mean_expr", "pct_expr", "n_cells"],
        ascending=[False, False, False],
    )
    peak = ranked.iloc[0]

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

    peak_dimensions = []
    for col in group_cols:
        peak_dimensions.append(
            f"{display_names.get(col, col)}="
            f"{str(peak.get(col, 'All matched values')).replace('|', '/')}"
        )

    lines = [
        f"VasQ deterministic summary for {gene}:",
        "",
        (
            f"Global maximum among all {len(stats_df)} eligible requested "
            "comparison groups: "
            + "; ".join(peak_dimensions)
            + f"; Mean expression={peak['mean_expr']:.3f}; "
            f"Expressing cells={100.0 * peak['pct_expr']:.1f}%; "
            f"Cells analyzed (n)={int(peak['n_cells'])}."
        ),
    ]

    if cell_type_stats is not None and not cell_type_stats.empty:
        ranked_cell_types = cell_type_stats.sort_values(
            ["mean_expr", "pct_expr", "n_cells"],
            ascending=[False, False, False],
        )
        shown_cell_types = ranked_cell_types.head(max_cell_type_rows)
        lines.extend([
            "",
            (
                "Overall cell-type ranking across all matched regions "
                "(computed directly from all matched cells, so region-level "
                "means are not averaged equally):"
            ),
            "",
            (
                "| Rank | Cell type | Mean expression (log-normalized) | "
                "Expressing cells | Cells analyzed (n) |"
            ),
            "| ---: | --- | ---: | ---: | ---: |",
        ])
        for rank, (_, row) in enumerate(shown_cell_types.iterrows(), start=1):
            cell_type = str(row.get("cell_type", "Unknown")).replace("|", "/")
            lines.append(
                f"| {rank} | {cell_type} | {row['mean_expr']:.3f} | "
                f"{100.0 * row['pct_expr']:.1f}% | "
                f"{int(row['n_cells'])} |"
            )
        if len(ranked_cell_types) > len(shown_cell_types):
            lines.append(
                f"Top {len(shown_cell_types)} of "
                f"{len(ranked_cell_types)} eligible cell types shown."
            )

    lines.extend([
        "",
        (
            "Representative detailed comparison groups, selected from the "
            "complete ranked result while balancing the requested dimensions "
            "(groups with fewer than "
            f"{MIN_CELLS_PER_GROUP} cells are not displayed):"
        ),
        "",
    ])
    lines.extend(format_comparison_coverage(stats_df, group_cols, max_values=max_coverage_values))
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


def resolve_dataset_entities_with_gpt(
    user_input,
    available_cell_types,
    available_regions,
    available_cell_classes=None,
    available_region_layers=None,
    include_interpretations=False,
):
    available_cell_types = available_cell_types or []
    available_regions = available_regions or []
    available_cell_classes = available_cell_classes or []
    available_region_layers = available_region_layers or []
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
        '"region_layers": [], "filter_interpretations": []}. Only use exact '
        "labels from the supplied lists. When a functional or conceptual "
        "phrase rather than a literal dataset label is mapped to one or more "
        "labels (for example, 'memory-related region' to 'Hippocampus'), add "
        "one filter_interpretations object with keys dimension, source_phrase, "
        "labels, and rationale. source_phrase must be an exact short span from "
        "the user query; labels must be the exact selected dataset labels; and "
        "rationale must briefly explain both the biological connection and why "
        "those labels are the closest available dataset match. Do not imply "
        "that one selected region is the only structure involved in a broad "
        "function. Do not add an interpretation object for a literal label or "
        "a deterministic spelling/abbreviation alias."
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
        filter_interpretations = parsed.get("filter_interpretations", [])

        cell_types = validate_controlled_vocabulary(
            cell_types, available_cell_types, dimension_name="cell_type"
        )
        cell_classes = validate_controlled_vocabulary(
            cell_classes, available_cell_classes, dimension_name="cell_class"
        )
        regions = validate_controlled_vocabulary(
            regions, available_regions, dimension_name="region"
        )
        region_layers = validate_controlled_vocabulary(
            region_layers, available_region_layers, dimension_name="region_layer"
        )

        result = (cell_types, cell_classes, regions, region_layers)
        if include_interpretations:
            return (
                *result,
                filter_interpretations
                if isinstance(filter_interpretations, list)
                else [],
            )
        return result

    except Exception as e:
        logger.exception("GPT dataset entity resolution failed: %s", e)
        # None means the helper failed and allows the caller to use local
        # matching. A successful helper response can intentionally return [].
        result = (None, None, None, None)
        if include_interpretations:
            return (*result, [])
        return result

def _unanimous_value(per_gene_map, genes):
    """A cell-type/region value only counts if EVERY gene in `genes` maps
    to the exact same non-empty value (single value, not a set) in
    `per_gene_map`. This check runs in Python rather than trusting the
    model to self-police cross-gene consistency in one aggregate answer --
    the same lesson learned from the citation-URL fix: when correctness
    depends on comparing several independent pieces of the model's own
    output against each other, do that comparison in code, not prompt
    instructions alone. One gene missing a value, having a different
    value, or having more than one candidate value all count as "not
    unanimous" and yield [].
    """
    values = []
    for gene in genes:
        gene_values = per_gene_map.get(gene) or per_gene_map.get(gene.upper()) or []
        if len(gene_values) != 1:
            return []
        values.append(gene_values[0])
    if not values or len(set(values)) != 1:
        return []
    return [values[0]]


def infer_matrix_hints_from_web_evidence(
    genes,
    web_result_text,
    available_cell_types,
    available_cell_classes,
    available_regions,
    available_region_layers,
):
    """Read literature evidence to see if it establishes cell-type and/or
    brain-region associations for the given genes -- used only as a
    fallback for whichever dimensions the user's own query left
    unresolved, so the VasQ matrix would otherwise be reported across
    everything on that axis. Returns
    (cell_types, cell_classes, regions, region_layers, cell_reason,
    region_reason). Every list is empty, and each reason is "", when the
    evidence does not clearly and unambiguously support a specific value --
    this function never guesses. `cell_reason`/`region_reason` are each
    scoped to their own axis only, so a caller that applies just one axis
    never has to show rationale text about the other, unapplied one.

    When multiple genes are queried together, the model judges each gene
    independently (so one gene's strong signal, e.g. "APP -> Neuron", can't
    silently get applied to a different gene, e.g. PSEN1, for which the
    literature never established that association) and this function only
    returns a value if every gene agrees on the exact same one -- checked
    in Python, not left to the model's own aggregate judgment.
    """
    genes = genes or []
    web_result_text = (web_result_text or "").strip()
    if not genes or not web_result_text:
        return [], [], [], [], "", ""

    system_prompt = (
        "Read biomedical literature evidence and decide, for EACH gene "
        "independently, whether it establishes: (1) a specific cell type "
        "or cell class that gene is known to be associated with -- for "
        "example as a canonical marker gene, from single-cell/tissue-atlas "
        "expression data, or a cell-type-specific functional role; and/or "
        "(2) a specific brain region or region layer that gene is known to "
        "be relevant to -- for example a region implicated in the disease "
        "discussed. Judge every gene separately; do not let one gene's "
        "strong signal influence another gene's judgment, even if they are "
        "discussed in the same disease context. Return JSON only with "
        "keys: "
        '{"gene_cell_types": {}, "gene_cell_classes": {}, "gene_regions": '
        '{}, "gene_region_layers": {}, "cell_reason": "", '
        '"region_reason": ""}. Each of the first four keys maps a gene '
        "symbol (exactly as given below) to a list of at most one value "
        "for that gene alone -- an empty list if that gene's own evidence "
        "does not clearly and unambiguously establish one. Only use exact "
        "labels from the supplied lists, and never invent one. Every gene "
        "given below must appear as a key in all four maps, even if its "
        "value is an empty list. \"cell_reason\" is a short (1-2 sentence) "
        "plain-language summary of what the evidence establishes about "
        "cell type/class, gene by gene -- leave it empty if every gene's "
        "cell-type and cell-class lists are empty. \"region_reason\" is "
        "the same, but for the region/region_layer judgment only -- leave "
        "it empty if every gene's region and region_layer lists are "
        "empty. Never mention the cell-type judgment inside region_reason "
        "or vice versa."
    )
    user_prompt = (
        f"Genes: {genes}\n\n"
        f"Available cell types: {available_cell_types}\n\n"
        f"Available cell classes: {available_cell_classes}\n\n"
        f"Available brain regions: {available_regions}\n\n"
        f"Available region layers: {available_region_layers}\n\n"
        f"Literature evidence:\n{cap_source_text(web_result_text, 6000)}"
    )

    try:
        response = call_helper_api(
            system_prompt,
            user_prompt,
            stage_name="web_matrix_hint_inference",
            # This task judges four dimensions plus two rationale fields in
            # one call -- more for the model to do than the other, simpler
            # helper calls that share OPENAI_HELPER_TIMEOUT_SECONDS. Give it
            # its own, more generous budget so it isn't starved by whatever
            # that shared setting happens to be tuned to.
            timeout_seconds=_env_float(
                "OPENAI_MATRIX_HINT_TIMEOUT_SECONDS", 45
            ),
        )
        parsed = parse_json_object(response.choices[0].message.content)
        if not parsed:
            return [], [], [], [], "", ""

        def _clean_map(key):
            raw_map = parsed.get(key) or {}
            if not isinstance(raw_map, dict):
                return {}
            return {
                str(gene): [str(v) for v in (values or []) if str(v).strip()]
                for gene, values in raw_map.items()
            }

        gene_cell_types = _clean_map("gene_cell_types")
        gene_cell_classes = _clean_map("gene_cell_classes")
        gene_regions = _clean_map("gene_regions")
        gene_region_layers = _clean_map("gene_region_layers")

        cell_types = validate_controlled_vocabulary(
            _unanimous_value(gene_cell_types, genes),
            available_cell_types,
            dimension_name="cell_type",
        )
        cell_classes = validate_controlled_vocabulary(
            _unanimous_value(gene_cell_classes, genes),
            available_cell_classes,
            dimension_name="cell_class",
        )
        regions = validate_controlled_vocabulary(
            _unanimous_value(gene_regions, genes),
            available_regions,
            dimension_name="region",
        )
        region_layers = validate_controlled_vocabulary(
            _unanimous_value(gene_region_layers, genes),
            available_region_layers,
            dimension_name="region_layer",
        )
        cell_reason = str(parsed.get("cell_reason", "")).strip() if (cell_types or cell_classes) else ""
        region_reason = str(parsed.get("region_reason", "")).strip() if (regions or region_layers) else ""
        return cell_types, cell_classes, regions, region_layers, cell_reason, region_reason
    except Exception:
        logger.exception("Web-evidence matrix-hint inference failed")
        return [], [], [], [], "", ""



def extract_genes(user_input):
    system_prompt = (
        "You are an expert molecular biologist. Extract all human gene symbols "
        "or gene names explicitly mentioned in the user's message. "
        "Return only a Python list, e.g. ['APOE', 'SLC2A1', 'CLDN5']. "
        "If no genes are explicitly mentioned, return []."
    )

    try:
        response = call_helper_api(system_prompt, user_input)
        raw_text = response.choices[0].message.content.strip()
    except Exception:
        # Match the degrade-gracefully pattern used by the other helper/web
        # calls: a slow or failed API call should skip gene extraction, not
        # bubble up and cost the whole turn (which would otherwise only be
        # caught by chat()'s top-level fallback).
        logger.exception("extract_genes: call_helper_api failed")
        return []

    try:
        genes = ast.literal_eval(raw_text)
        if isinstance(genes, list):
            return [str(g).upper().strip() for g in genes if str(g).strip()]
    except Exception:
        logger.warning("extract_genes: could not parse model output: %r", raw_text)

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


def pretty_region_name(region):
    region_map = {
        "CP": "choroid plexus",
        "Hip-EC": "hippocampal-entorhinal vasculature",
        "ACA": "anterior cerebral artery",
        "BA.CoW": "basilar artery / circle of Willis",
    }
    return region_map.get(region, region)


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


def marker_size_from_cell_counts(n_cells_series, *, min_size=4.0, max_size=17.0):
    """Map cell counts to dot diameters on a log scale, normalized to the
    min/max actually present in this plot. Cell counts routinely span two
    to three orders of magnitude (tens to tens of thousands) in a single
    plot, so a linear or sqrt scale would make everything but the single
    largest group look identically tiny; log compresses that range into a
    readable size gradient instead.
    """
    counts = np.clip(n_cells_series.to_numpy(dtype=float), 1.0, None)
    log_counts = np.log10(counts)
    log_min, log_max = log_counts.min(), log_counts.max()
    if log_max > log_min:
        normalized = (log_counts - log_min) / (log_max - log_min)
    else:
        # Every group in this plot has (about) the same cell count.
        normalized = np.ones_like(log_counts)
    return (min_size + (max_size - min_size) * normalized).round(1).tolist()


def matrix_plot_marker(plot_df, *, showscale=True, color_max=None):
    """Shared dot-matrix encoding: color=mean expression, size=cells analyzed."""
    if color_max is None:
        positive = plot_df.loc[plot_df["mean_expr"] > 0, "mean_expr"]
        color_max = float(positive.quantile(0.95)) if not positive.empty else 1.0
    color_max = max(float(color_max), 0.001)
    return {
        "size": marker_size_from_cell_counts(plot_df["n_cells"]),
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
        "line": {"color": "#ffffff", "width": 1},
        "showscale": showscale,
        "colorbar": {
            "title": {
                "text": "Mean<br>expression",
                "font": {"size": 16},
            },
            "tickfont": {"size": 14},
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
    tick_text = [
        wrap_plot_label(value, max_chars=28, max_lines=2)
        for value in region_order
    ]
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
                    "Color = mean expression · Size = cells analyzed"
                    f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            # Leave enough vertical room for every cell-type tick. When this
            # was compressed more aggressively, Plotly automatically skipped
            # alternating y tick labels, making correctly positioned dots look
            # as though they belonged to the wrong cell type.
            "height": max(650, 210 + 30 * len(cell_order)),
            "autosize": True,
            # Without this, hovering can trigger every point sharing the
            # nearest x (region) at once instead of just the point under the
            # cursor -- e.g. every cell type present in that region popping
            # up together.
            "hovermode": "closest",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#ffffff",
            "font": {"family": "Satoshi, Arial, sans-serif", "color": "#32175a", "size": 16},
            "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#8ab4c4", "font": {"color": "#32175a"}},
            "xaxis": {
                "title": {
                    "text": "Region",
                    "standoff": 18,
                    "font": {"size": 20},
                },
                "categoryorder": "array",
                "categoryarray": region_order,
                "tickmode": "array",
                "tickvals": region_order,
                "ticktext": tick_text,
                "tickangle": -90,
                "tickfont": {"size": 14},
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1.5,
                "ticks": "outside",
                "ticklen": 7,
                "tickwidth": 1.5,
                "tickcolor": "#32175a",
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "automargin": True,
            },
            "yaxis": {
                "title": {
                    "text": "Cell type",
                    "standoff": 14,
                    "font": {"size": 20},
                },
                "categoryorder": "array",
                "categoryarray": list(reversed(cell_order)),
                "tickmode": "array",
                "tickvals": cell_order,
                "ticktext": [
                    str(cell_type).replace("_", " ")
                    for cell_type in cell_order
                ],
                "tickfont": {"size": 14},
                "showline": True,
                "linecolor": "#32175a",
                "linewidth": 1.5,
                "ticks": "outside",
                "ticklen": 7,
                "tickwidth": 1.5,
                "tickcolor": "#32175a",
                "showgrid": True,
                "gridcolor": "#edf2f7",
                "automargin": True,
            },
            "margin": {"l": 205, "r": 125, "t": 95, "b": 210},
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
            "showline": True,
            "linecolor": "#32175a",
            "linewidth": 1.25,
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.25,
            "tickcolor": "#32175a",
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
            "showline": True,
            "linecolor": "#32175a",
            "linewidth": 1.25,
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.25,
            "tickcolor": "#32175a",
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
                "Color = mean expression · Size = cells analyzed"
                f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        "height": max(760, 180 + len(cell_types) * max(190, 34 * len(gene_order))),
        "autosize": True,
        "hovermode": "closest",
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
    marker_sizes = marker_size_from_cell_counts(
        plot_df["n_cells"], min_size=6.0, max_size=24.0
    )

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
                    "Color = mean expression · Size = cells analyzed"
                    f" · Groups require ≥{MIN_CELLS_PER_GROUP} cells"
                    "</span>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            "height": max(650, 220 + 42 * len(gene_order)),
            "autosize": True,
            "hovermode": "closest",
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


def explicitly_disables_web_search(user_input: str) -> bool:
    """Detect an instruction to skip external Web/literature retrieval."""
    text = normalize_text(user_input)
    patterns = [
        r"\bdo not use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdon't use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdont use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdo not (?:search|browse) (?:the )?web\b",
        r"\bdon't (?:search|browse) (?:the )?web\b",
        r"\bdo not (?:search|review) (?:the )?literature\b",
        r"\bdon't (?:search|review) (?:the )?literature\b",
        r"\bdont (?:search|review) (?:the )?literature\b",
        r"\b(?:no|without|skip|avoid) (?:the )?(?:literature search|literature review|literature evidence|external evidence|citations?|source links?)\b",
        r"\bdo not (?:use|include|provide) (?:any )?(?:literature evidence|external evidence|citations?|source links?)\b",
        r"\bdon't (?:use|include|provide) (?:any )?(?:literature evidence|external evidence|citations?|source links?)\b",
        r"\b(?:no|without|skip|avoid) (?:openai )?web search\b",
        r"\bdo not use (?:external |web |literature )?(?:evidence|sources)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def wants_web_search(user_input: str) -> bool:
    # Remove explicitly negated search phrases before looking for positive
    # search requests. Previously, "Do not use Web Search" matched the bare
    # term "web search" and incorrectly enabled external retrieval.
    lowered = normalize_text(user_input)
    negated_patterns = [
        r"\bdo not use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdon't use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdont use(?: kg rag,?)? (?:openai )?web search\b",
        r"\bdo not (?:search|browse) (?:the )?web\b",
        r"\bdon't (?:search|browse) (?:the )?web\b",
        r"\bdo not (?:search|review) (?:the )?literature\b",
        r"\bdon't (?:search|review) (?:the )?literature\b",
        r"\bdont (?:search|review) (?:the )?literature\b",
        r"\b(?:no|without|skip|avoid) (?:the )?(?:literature search|literature review|literature evidence|external evidence|citations?|source links?)\b",
        r"\bdo not (?:use|include|provide) (?:any )?(?:literature evidence|external evidence|citations?|source links?)\b",
        r"\bdon't (?:use|include|provide) (?:any )?(?:literature evidence|external evidence|citations?|source links?)\b",
        r"\b(?:no|without|skip|avoid) (?:openai )?web search\b",
    ]
    for pattern in negated_patterns:
        lowered = re.sub(pattern, " ", lowered)

    web_terms = [
        "search google",
        "google",
        "google it",
        "search literature",
        "search the literature",
        "literature search",
        "literature review",
        "literature evidence",
        "literature link",
        "literature links",
        "find articles",
        "articles",
        "for papers",
        "find papers",
        "find paper",
        "find papers on",
        "paper link",
        "paper links",
        "search pubmed",
        "pubmed",
        "google scholar",
        "scholar",
        "look up papers",
        "find studies",
        "provide citations",
        "include citations",
        "citation link",
        "citation links",
        "cite the source",
        "cite the sources",
        "source link",
        "source links",
        "source url",
        "source urls",
        "clickable link",
        "clickable links",
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
    if any(term in text for term in vasq_only_terms):
        return True

    # Accept natural variants such as "based only on the retrieved VasQ
    # matrix measurements" instead of depending on one exact word order.
    if "vasq matrix" in text and re.search(
        r"\b(?:only|solely|exclusively)\b",
        text,
    ):
        return True

    # An explicit VasQ matrix request paired with a no-Web-Search instruction
    # is also unambiguously matrix-only.
    return (
        "vasq matrix" in text
        and explicitly_disables_web_search(user_input)
    )


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


def recent_conversation_context(
    history, max_messages=8, max_chars_per_message=800, max_chars=5000
):
    """Build the short context window analyze_query_intent() uses to resolve
    follow-ups ("what about that gene").

    Each message is capped *individually* before joining. Capping only the
    joined blob (the previous behavior) let one long assistant reply --
    a full VasQ table plus a literature review can run several thousand
    characters -- eat the entire budget on its own, silently dropping the
    user question that prompted it and every earlier turn. Per-message
    capping guarantees several recent turns survive regardless of how long
    any single one of them was.
    """
    messages = []
    for message in (history or [])[-max_messages:]:
        if message.get("role") not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message].rstrip() + " […]"
        messages.append(f"{message['role']}: {content}")
    combined = "\n".join(messages)
    # Still cap the overall size as a final safety net, but from the front
    # now that no single message can dominate it -- keeps the most recent
    # turns rather than an arbitrary tail cut mid-message.
    if len(combined) > max_chars:
        combined = combined[-max_chars:]
    return combined


def summarize_turn_for_history(user_input, final_message, max_verbatim_chars=600):
    """Compact a turn before it goes into `history`.

    `update_history` used to store `final_message` verbatim -- a VasQ answer
    (a full expression table plus a literature review) is often several
    thousand characters. Every later turn's intent classification and final
    synthesis reread the *entire* growing history, so storing raw answers
    makes each turn's context bigger, and unrelated (long-ago) detail can
    crowd out the one thing a follow-up actually needs: what was just
    discussed. Store a short summary instead, so history grows by roughly a
    fixed, small amount per turn regardless of how detailed the answer was.

    Short replies (greetings, brief clarifications) are kept verbatim --
    they are already compact, so summarizing them would spend a call for no
    benefit.
    """
    text = str(final_message or "").strip()
    if len(text) <= max_verbatim_chars:
        return text

    system_prompt = (
        "Compress an assistant's answer into a short note for the "
        "conversation's own memory, written so a later turn can resolve a "
        "follow-up question (e.g. 'what about that gene', 'and in the "
        "hippocampus?') without rereading the full answer. In 2-4 sentences, "
        "state: what was asked, which genes, diseases, cell types, and brain "
        "regions were involved, and the single most important conclusion or "
        "measured finding. Do not restate full tables, citations, or "
        "caveats -- keep only what a future turn needs to stay oriented. "
        "Plain text, no markdown, no headers."
    )
    user_prompt = (
        f"User's question:\n{user_input}\n\n"
        f"Assistant's full answer:\n{cap_source_text(text, 12000)}"
    )
    try:
        response = call_helper_api(
            system_prompt,
            user_prompt,
            stage_name="history_summary",
            timeout_seconds=_env_float("OPENAI_HISTORY_SUMMARY_TIMEOUT_SECONDS", 20),
        )
        summary = str(
            getattr(response.choices[0].message, "content", None) or ""
        ).strip()
        if summary:
            return summary
    except Exception:
        logger.exception(
            "History summarization failed; falling back to a plain truncation"
        )

    # Fallback: keep the turn discoverable even if summarization fails,
    # rather than silently dropping it or storing the (huge) raw text.
    return text[:max_verbatim_chars].rstrip() + " […]"


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
    # Marker/rank/top-gene wording still counts as an expression question --
    # it used to route to a separate precomputed-marker-table pipeline, which
    # has been removed. It's no longer tracked as its own routing flag.
    asks_expression = (
        wants_marker_query(text)
        or wants_top_genes(text)
        or wants_matrix_expression_query(text)
    )
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
            "asks_drugs": False,
            "use_vasq": False,
            "genes": [],
            "diseases": [],
            "resolved_question": user_input.strip(),
        }

    system_prompt = (
        "Classify a conversation turn for a biomedical/neuroscience research "
        "assistant. Return JSON only with keys: is_scientific (boolean), "
        "asks_expression (boolean), asks_drugs "
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
        "distribution, or marker/rank/top-gene questions -- all of these are "
        "answered from the same measured expression matrix. Set use_vasq "
        "when asks_expression is true and the "
        "question concerns brain vasculature, vascular cell types/regions, or "
        "does not specify a different tissue; set it false when the user "
        "explicitly asks about a non-vascular or other-organ tissue. Resolve "
        "short follow-ups from recent context, but do "
        "not invent a gene, disease, cell type, or brain region. Include genes "
        "or diseases inherited from context only when the reference is "
        "unambiguous. Normalize gene symbols to uppercase. In resolved_question, "
        "preserve the current user's exact functional and qualitative wording "
        "and all qualifiers. Only add an unambiguous entity needed to resolve a "
        "short follow-up; do not translate a phrase such as 'memory-related "
        "region' into a named brain region, and do not silently replace a broad "
        "concept with a narrower dataset label. If the current message is "
        "standalone, copy it verbatim into resolved_question."
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


def retrieved_text_graph_and_table(result):
    if isinstance(result, dict):
        return (
            str(result.get("text", "") or ""),
            result.get("graph_json"),
            result.get("expression_table"),
        )
    if result is None:
        return "", None, None
    return str(result), None, None


def cap_source_text(text, limit):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[Source text truncated]"


def cap_text_at_line_boundary(text, limit, suffix="[Text truncated]"):
    """Cap text without cutting a Markdown row or consuming another gene."""
    text = str(text or "").strip()
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text

    suffix = str(suffix or "").strip()
    if len(suffix) >= limit:
        return suffix[:limit]

    content_budget = limit - len(suffix) - 1
    clipped = text[:content_budget]
    if "\n" in clipped:
        clipped = clipped.rsplit("\n", 1)[0]
    clipped = clipped.rstrip()
    if not clipped:
        return suffix
    return clipped + "\n" + suffix


_URL_PATTERN = re.compile(r'https?://[^\s\])"\'<>]+')
_MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')


def _normalize_url(url):
    return str(url or "").strip().rstrip('.,;:)]}\u00bb\u201d\'"').split("#", 1)[0]


def extract_known_urls(*texts):
    """Collect every URL that actually appeared somewhere in this turn's
    own evidence (web search, KG-RAG, drug search), regardless of stage.
    Used as a whitelist for `sanitize_uncited_urls` -- a hard, code-level
    backstop, since prompting alone cannot guarantee a rewriting model
    never invents or misattributes a URL when composing the final answer.
    """
    known = set()
    for text in texts:
        if not text:
            continue
        for match in _URL_PATTERN.findall(str(text)):
            known.add(_normalize_url(match))
    return known


def extract_gene_literature_links(evidence_text, genes, max_links_per_gene=2):
    """Map citations to exact gene sections emitted by the focused search."""
    text = str(evidence_text or "")
    if not text.strip():
        return {}

    # Fallback sources were consulted by the tool but are deliberately not
    # tied to a claim. Exclude that appendix so its URLs cannot accidentally
    # be assigned to the final gene section.
    text = text.split(
        "\n\nOther sources the search tool consulted",
        1,
    )[0]
    requested = {
        str(gene).upper().strip()
        for gene in (genes or [])
        if str(gene).strip()
    }
    heading_pattern = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:[-*][ \t]+)?"
        r"(?:\*\*)?GENE:[ \t]*([A-Z][A-Z0-9.-]{1,19})"
        r"(?:\*\*)?[^\n]*$"
    )
    headings = list(heading_pattern.finditer(text))
    links_by_gene = {}

    for index, match in enumerate(headings):
        gene = match.group(1).upper()
        if gene not in requested:
            continue
        section_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(text)
        )
        section = text[match.end():section_end]
        urls = []
        for raw_url in _URL_PATTERN.findall(section):
            url = _normalize_url(raw_url)
            if url and url not in urls:
                urls.append(url)
            if len(urls) >= max_links_per_gene:
                break
        if urls:
            links_by_gene[gene] = urls

    return links_by_gene


def append_missing_gene_literature_links(
    final_message,
    gene_literature_result,
    genes,
):
    """Keep gene-specific links visible even if synthesis drops citations."""
    links_by_gene = extract_gene_literature_links(
        gene_literature_result,
        genes,
    )
    if not links_by_gene:
        return final_message

    final_urls = extract_known_urls(final_message)
    missing_lines = []
    for gene in genes or []:
        gene = str(gene).upper().strip()
        urls = links_by_gene.get(gene, [])
        if not urls:
            continue
        if any(
            url in final_urls
            or any(
                url.startswith(existing) or existing.startswith(url)
                for existing in final_urls
            )
            for url in urls
        ):
            continue
        citations = " ".join(
            f"[{_source_label(url)}]({url})"
            for url in urls
        )
        missing_lines.append(f"- **{gene}:** {citations}")

    if not missing_lines:
        return final_message

    return (
        str(final_message or "").rstrip()
        + "\n\n### Gene-specific literature links\n\n"
        + "\n".join(missing_lines)
    )


def sanitize_uncited_urls(final_message, known_urls):
    """Strip any hyperlink in the final answer whose URL never appeared
    anywhere in this turn's own gathered evidence -- keep the visible
    label text, just drop the link itself.

    This does not guarantee a URL that *did* appear in the evidence is
    paired with the *correct* claim (that's a separate, prompting-level
    concern); it only guarantees the reader is never handed a clickable
    link to something that traces back to nothing in this turn's evidence
    at all -- e.g. a URL the model reconstructed from general training
    knowledge rather than from what was actually retrieved just now.
    """
    if not final_message or not known_urls:
        return final_message

    def _replace(match):
        label, url = match.group(1), match.group(2)
        normalized = _normalize_url(url)
        if normalized in known_urls:
            return match.group(0)
        # Tolerate minor formatting drift (trailing slash, query string)
        # the model may introduce while otherwise citing a real source.
        if any(
            normalized.startswith(known) or known.startswith(normalized)
            for known in known_urls
        ):
            return match.group(0)
        logger.warning(
            "Stripping a citation URL not found in this turn's evidence: %s",
            url,
        )
        return label

    return _MARKDOWN_LINK_PATTERN.sub(_replace, final_message)


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
        update_history(
            history, "assistant", summarize_turn_for_history(user_input, final_message)
        )
        return final_message, history, None, None

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
        # KG-RAG and the first Web Search are independent lookups. Running
        # them sequentially means paying for both wait times in full; in
        # observed logs KG-RAG alone has taken 160-250s. Overlapping them
        # turns the wall-clock cost into roughly max(KG-RAG, Web Search)
        # instead of their sum. The trade-off: Web Search can no longer wait
        # to see whether KG-RAG found anything relevant, so it always
        # searches from the question directly rather than being primed with
        # KG context -- KG's own content still reaches gene derivation and
        # the final synthesis evidence package below, unaffected.
        kg_result = None
        scientific_web_result = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            kg_future = executor.submit(query_kg_rag, resolved_question)
            logger.info("Web Search stage 1/2: scientific knowledge and genes")
            web_future = executor.submit(
                search_scientific_web,
                resolved_question,
                kg_context=None,
                kg_assessment=None,
            )
            try:
                kg_result = kg_future.result()
            except Exception:
                logger.exception("KG-RAG branch failed")
                kg_result = None
            try:
                scientific_web_result = web_future.result()
            except Exception:
                logger.exception("Scientific Web Search branch failed")
                scientific_web_result = None

        _raise_if_cancelled(should_stop)
        kg_assessment = assess_kg_relevance(resolved_question, kg_result)

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
    # (including marker/rank/top-gene questions, now that they route through
    # the same matrix_expression() pipeline) needs genes but the primary
    # search did not resolve any.
    needs_gene_fallback = (
        not direct_vasq_only
        and intent.get("asks_expression")
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

    # The discovery search above runs before the final prioritized gene list
    # exists. For disease-expression questions, retrieve a second, compact
    # evidence set that explicitly covers every selected gene with its own
    # literature statement and citation. This prevents the final answer from
    # collapsing ten genes into one generic disease-association paragraph.
    gene_literature_result = None
    needs_gene_literature = (
        not direct_vasq_only
        and bool(genes)
        and bool(
            diseases
            or wants_web_search(user_input)
            or not user_supplied_genes
        )
    )
    if needs_gene_literature:
        _raise_if_cancelled(should_stop)
        try:
            logger.info(
                "Web Search: gene-by-gene literature coverage for genes=%s",
                genes,
            )
            gene_literature_result = search_gene_literature_evidence(
                resolved_question,
                genes,
                diseases=diseases,
            )
        except Exception:
            logger.exception("Gene-by-gene literature search failed")
            gene_literature_result = None
        _raise_if_cancelled(should_stop)

    # Branch B: calculate expression for the explicit or first-search-derived
    # gene list. Marker/rank questions (previously served by a separate
    # precomputed-marker-table pipeline) now go through the same VasQ matrix
    # as every other expression question -- candidate genes for a bare
    # "top marker genes for X" question come from the web-search-derived
    # `genes` list (see needs_gene_fallback above), not a reverse lookup
    # against a precomputed ranking table.
    vasq_result = None
    vasq_note = ""
    graph_json = None
    filter_interpretations = []

    if intent.get("asks_expression") and intent.get("use_vasq"):
        _raise_if_cancelled(should_stop)
        if genes:
            try:
                vasq_result = matrix_expression(
                    resolved_question,
                    genes_override=genes,
                    web_evidence_text=scientific_web_result,
                    kg_evidence_text=(
                        kg_result if kg_assessment.get("relevant") else None
                    ),
                    original_user_input=user_input,
                )
                if isinstance(vasq_result, dict):
                    filter_interpretations = list(
                        vasq_result.get("filter_interpretations") or []
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
    vasq_text, graph_json, expression_table = retrieved_text_graph_and_table(
        vasq_result
    )

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

        if gene_literature_result:
            evidence_parts.append(
                "GENE-BY-GENE SCIENTIFIC LITERATURE EVIDENCE:\n"
                + cap_source_text(gene_literature_result, 18000)
            )
        else:
            evidence_parts.append(
                "GENE-BY-GENE LITERATURE STATUS:\n"
                "No usable gene-specific literature evidence was returned."
            )

        if scientific_web_result:
            evidence_parts.append(
                "PRIMARY SCIENTIFIC WEB EVIDENCE USED TO IDENTIFY GENES:\n"
                + cap_source_text(scientific_web_result, 5000)
            )
        elif not gene_literature_result:
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
            "claims, and preserve its citations. The evidence text may "
            "already contain complete markdown links, e.g. "
            "\"[pubmed.ncbi.nlm.nih.gov/12345](https://...)\", immediately after "
            "a specific claim -- keep that exact link exactly as given, "
            "character for character, including its visible text. Never "
            "rewrite, shorten, or replace the visible text with an "
            "author/year or paper-title guess, never change the URL, and "
            "never move a link to a different claim than the one it "
            "followed in the evidence. A claim may have more than one such "
            "link immediately after it (multiple sources supporting the "
            "same statement) -- when it does, keep all of them, separated "
            "by a single space exactly as given; do not wrap them in your "
            "own parentheses or brackets, and do not merge, reorder, or "
            "drop any of them. If a claim has no such link, do not "
            "invent one from memory -- state it without a citation. Use VasQ only for measured "
            "brain-vasculature expression claims; distinguish matrix mean "
            "expression from marker rank/score. When the web/literature "
            "evidence reports a cell type a gene is known to be associated "
            "with (a marker gene, atlas data, or cell-type-specific "
            "function) and the VasQ matrix also reports that gene's "
            "measured expression by cell type, use the supplied 'Overall "
            "cell-type ranking across all matched regions' to decide whether "
            "the VasQ result agrees or disagrees with the literature-reported "
            "cell type. That ranking is computed directly from all matched "
            "cells and is the authoritative cell-type comparison. Report the "
            "supplied 'Global maximum among all eligible requested comparison "
            "groups' separately as the most specific requested-group peak "
            "(region-specific when Brain region is one of the dimensions). "
            "When that peak is a single region x cell-type group, do not use "
            "it to decide overall cell-type agreement. Note a discrepancy "
            "plainly rather than silently "
            "picking one source. Do not infer that a cell type is absent just "
            "because it is outside a capped table. "
            "In the drug section, clearly "
            "separate direct gene/protein modulators, pathway-related "
            "compounds, and disease-directed treatments, and distinguish "
            "approved, clinical, preclinical, and research-tool status. Never "
            "claim that a disease drug directly targets a gene without direct "
            "support. For disease questions that ask about genes and their "
            "expression, organize the answer into two explicit sections: "
            "(1) associated genes and supporting knowledge and (2) VasQ "
            "matrix expression by the requested brain region, region layer, "
            "cell class, and/or cell type dimensions with the supplied measured "
            "values. In section (1), treat 'GENE LIST DERIVED FROM THE FIRST "
            "SEARCH' as the closed list to explain unless the user explicitly "
            "asks for additional genes. Cover every gene in that list in its "
            "own bullet or short paragraph, normally with one or two substantive "
            "sentences describing the type of disease evidence and relevant "
            "function or cell-type context. Preserve at least one exact inline "
            "literature link for each gene whenever its gene-by-gene evidence "
            "supplies one. Do not replace these gene-specific explanations "
            "with one generic knowledge-graph paragraph, and do not append a "
            "larger list of incidental knowledge-graph genes. If gene-specific "
            "evidence is unavailable for one gene, state that only for that "
            "gene. In section (2), do not jump directly from a generic matrix "
            "description to tables and plots. Before any detailed table, give "
            "every analyzed gene one concise interpretation bullet stating "
            "its top overall VasQ cell type, its most specific requested-group "
            "peak, and whether the cell-type result agrees with the cited "
            "literature when that comparison is supported. If you create the "
            "compact peak-summary table, label its two concepts unambiguously "
            "as 'Highest cell type pooled across regions' and 'Cell type in "
            "highest region x cell-type group'; never shorten them to the "
            "ambiguous labels 'Overall top cell type' and 'Cell type at peak'. "
            "Preserve every "
            "comparison dimension present in the VasQ "
            "table and never merge distinct region layers, cell types, or brain "
            "regions. Treat the supplied 'Applied matrix filters' line as the "
            "authoritative record of which filters were actually used. Treat "
            "a supplied 'SEMANTIC FILTER INTERPRETATION' block as mandatory "
            "reader-facing context. Before stating any expression conclusion, "
            "explicitly explain the user's functional phrase, which exact "
            "VasQ label or labels it was mapped to, why that biological and "
            "dataset-vocabulary mapping was made, and the supplied scope "
            "caveat. Do not jump directly from a phrase such as 'memory-related "
            "region' to 'hippocampal' without showing this reasoning. Use the "
            "heading 'How the request was interpreted' for that explanation. "
            "Treat "
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

    # Even summarized, unbounded history growth is still growth. Cap how many
    # past turns synthesis rereads every time -- a safety net for very long
    # sessions, on top of (not instead of) storing summaries rather than raw
    # answers.
    max_synthesis_history = _env_int("VASQ_SYNTHESIS_HISTORY_MESSAGES", 20)
    if history and history[0].get("role") == "system":
        past_turns = history[1:]
        if len(past_turns) > max_synthesis_history:
            past_turns = past_turns[-max_synthesis_history:]
        synthesis_messages = [history[0], synthesis_instruction] + past_turns
    else:
        past_turns = history[:]
        if len(past_turns) > max_synthesis_history:
            past_turns = past_turns[-max_synthesis_history:]
        synthesis_messages = [synthesis_instruction] + past_turns

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
            "\n\n".join(
                part
                for part in [gene_literature_result, scientific_web_result]
                if part
            ),
            drug_result=drug_result,
        )
    _raise_if_cancelled(should_stop)

    # The synthesis prompt asks the model to preserve inline citations, but a
    # rewriting step can still occasionally omit one. Because the focused
    # search uses deterministic GENE: SYMBOL sections, safely append only the
    # gene-to-URL pairs actually returned by that search when their URLs are
    # absent from the synthesized answer.
    final_message = append_missing_gene_literature_links(
        final_message,
        gene_literature_result,
        genes,
    )

    known_urls = extract_known_urls(
        gene_literature_result,
        scientific_web_result,
        kg_result,
        drug_result,
    )
    final_message = sanitize_uncited_urls(final_message, known_urls)
    final_message = ensure_filter_interpretation_visible(
        final_message,
        filter_interpretations,
    )

    logger.info("Final message generated: %r", final_message)
    update_history(
        history, "assistant", summarize_turn_for_history(user_input, final_message)
    )

    return final_message, history, graph_json, expression_table


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
        return fallback, safe_history, None, None
    finally:
        elapsed = time.monotonic() - started_at
        logger.info("VasQ turn finished elapsed=%.1fs", elapsed)
        _TURN_DEADLINE.reset(deadline_token)
