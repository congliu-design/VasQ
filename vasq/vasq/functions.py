import difflib
import json
import openai
import os
import pandas as pd
import re
import requests
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


logger = logging.getLogger(__name__)

#def query_kg_rag(user_input):
#    url = os.getenv("KG_RAG_URL", "http://kg-rag.railway.internal:8080/query")
#    logger.info("Calling KG_RAG_URL=%s", url)
#    response = requests.post(url, json={"query": user_input}, timeout=120)
#    logger.info("kg-rag status=%s body=%s", response.status_code, response.text[:500])
#    response.raise_for_status()
#    return response.json()


# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")



### Helper Functions ###

# Call OpenAI API
def call_api(history, functions=None):
    MAX_MESSAGES = 12

    trimmed_history = history[-MAX_MESSAGES:]

    chat_co = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=trimmed_history,
        functions=functions,
        temperature=0.2,
        top_p=0.4,
    )
    return chat_co.choices[0].message


# Update chat history
def update_history(history, role, content):
    message = {"role": role, "content": content}
    history.append(message)

# Initialize chat
def initialize(history):
    global init_flag
    
    system_prompt = "You are a neuroscience research assistant. You answer \
        scientific questions using multiple resources and should draw on prior \
        conversation history to maintain coherence. When details are \
        unspecified, infer them based on recent context (e.g., assume the same \
        cell type if the user referenced it most recently). You have access to \
        the following tools to support scientific inquiry: \n1. \
        gene_expression Function: Returns gene expression and protein \
        prevalence data for specific cell types and brain vasculature regions, \
        based on single-nucleus RNA sequencing (snRNA-seq) from the Brain \
        Resilience Laboratory at Stanford University. \n2. Biomedical \
        Knowledge Graph: A curated knowledge graph based on SPOKE (from UCSF), \
        containing molecular and disease biology relationships. \n3. Google \
        Search API: Allows web search for up-to-date biomedical information. \
        \n4. Pretrained Scientific Knowledge: You may also draw on your own \
        scientific knowledge acquired during pre-training. \nWHEN TO CALL \
        gene_expression: \n- If the user asks about gene expression or \
        protein prevalence in a cell type and/or brain region, and does NOT \
        specify a tissue type, ASSUME they mean brain vasculature and call the \
        function. \n- If the user asks: `Is gene X among the top Y expressed \
        genes in brain region Z?`, check all listed genes for each of the \
        specified brain regions using the function. \n- If the user explicitly \
        mentions vasculature (e.g., `in vasculature`, `vascular tissue`, \
        `blood vessels`), call the function. \nResponse Notes: The numbers \
        next to genes are their expression rank. Always include data numbers \
        in your response. \n- If the data returned includes some genes \
        mentioned in the user query, but not all, ASSUME the missing genes do \
        not appear in the top 1000 expressed genes for the given region and/or \
        cell type and state that in your answer. \n- If the data returned only \
        relates to a brain region, you MUST state in your response: `This \
        answer reflects all cell types across the specified brain region. \n- \
        If the data returned only relates to a cell type, you MUST state: \
        `This answer reflects the specified cell type across all brain \
        regions.` \n- If tissue type is unspecified but both cell type and \
        brain region are given, add: `Since you specified a cell type and \
        brain region but did not mention tissue type, I’ve assumed brain \
        vasculature.` \n- In all cases, include: `This answer is based on \
        single-nucleus data from the Brain Resilience Lab at Stanford \
        University.` \nWHEN NOT TO CALL gene_expression: \n- If the user \
        explicitly says `not in vasculature`, `non-vascular`, or specifies a \
        different tissue (e.g., `nervous tissue`, `gray matter`, \
        `parenchyma`), DO NOT call the function. \n- DO NOT call the function \
        for queries unrelated to gene expression levels in specific cell types \
        or brain regions, even if they mention particular genes. \n- Instead, \
        use the knowledge graph, Google web search, or your own pretrained \
        knowledge to answer. \nWHEN TO CALL query_kg_rag: \n- If the user \
        mentions any disease by name in their query you MUST call \
        query_kg_rag. \nResponse strategy: \n- Prioritize information from \
        tools in the following order: \n1. gene_expression \n2. Biomedical \
        knowledge graph \n3. Web results (Google Search API) \n4. Parametric \
        (pretrained) knowledge \n- Always include information from each tool \
        used in the response. \n- Summarize the findings from each tool so the \
        user can ask follow-up questions if needed. \n- Cite all sources when \
        using the knowledge graph or web search. (e.g., `This data is from \
        NCBI and ChEMBL` or `Visit this website for more info...`) \n- Format \
        all responses clearly and professionally for a scientific audience. \
        \nAdditional Expectations: \n- Reason through ambiguous queries. \n- \
        Clarify assumptions explicitly in your replies. \n- Clearly state the \
        origin of any scientific data used. \n- Keep your answers as concise \
        as possible."

    update_history(history, "system", system_prompt)
    init_flag = False

# Call function from chat

def func_call(user_input, chat_message, history):
    if wants_web_search(user_input):
        logger.info("func_call override: explicit web/literature intent -> Google")
        return search_vertex_ai(user_input)

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

    candidate_cols = ["gene", "genes", "gene_name", "gene_symbol", "symbol"]

    if genes_df.shape[1] == 1:
        col = genes_df.columns[0]
    else:
        col = next((c for c in candidate_cols if c in genes_df.columns), genes_df.columns[0])

    genes = (
        genes_df[col]
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

    meta = pd.read_csv(CELL_META_PATH, index_col=0)
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


def matrix_expression(user_input):
    ensure_matrix_expression_data_loaded()

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
    plot_json = None

    for i, gene in enumerate(present_genes):
        if len(cell_indices) > 0:
            stats = summarize_group_expression(
                gene,
                cell_indices,
                ["brain_region", "cell_class", "cell_type"]
            )
        else:
            stats = summarize_group_expression(
                gene,
                MATRIX_META.index.to_numpy(),
                ["brain_region", "cell_class", "cell_type"]
            )

        all_sections.append(format_matrix_expression_summary(stats, gene, max_rows=5))

        if i == 0:
            plot_json = build_matrix_expression_plot(stats, gene_name=gene, max_rows=8)

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
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_HELPER_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

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

    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_HELPER_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        genes = ast.literal_eval(raw_text)
        if isinstance(genes, list):
            return [str(g).upper().strip() for g in genes if str(g).strip()]
    except Exception:
        pass

    return []

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

#def all_regions(user_input):
#    system_prompt = (
#        "Decide whether the user is asking for a comparison against all regions "
#        "or across the whole dataset, rather than only filtering to explicitly "
#        "named regions. Return only True or False.\n\n"
#        "Return True for examples like:\n"
#        "- 'higher than other regions'\n"
#        "- 'specific to hippocampus compared to the rest of brain'\n"
#        "- 'across all regions'\n"
#        "- 'highest in the brain'\n"
#        "- 'unique to pons versus other regions'\n\n"
#        "Return False for examples like:\n"
#        "- 'in hippocampus and amygdala'\n"
#        "- 'show top genes in pons'\n"
#        "- 'expression in thalamus'\n"
#        "- 'compare hippocampus and amygdala only'"
#    )
#
#    response = openai.chat.completions.create(
#        model=os.getenv("OPENAI_HELPER_MODEL", "gpt-4o"),
#        messages=[
#            {"role": "system", "content": system_prompt},
#            {"role": "user", "content": user_input}
#        ],
#        temperature=0
#    )
#
#    raw_text = response.choices[0].message.content.strip().lower()
#    return "true" in raw_text

def is_cross_region_comparison(user_input):
    system_prompt = (
        "Determine whether the user is asking for comparison against other brain regions "
        "or across the whole dataset. Return only True or False."
    )

    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_HELPER_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

    return "true" in response.choices[0].message.content.strip().lower()


def is_region_filtered_query(user_input):
    system_prompt = (
        "Determine whether the user mainly wants results filtered to one or more explicitly "
        "named brain regions, rather than compared to all other regions. Return only True or False."
    )

    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_HELPER_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

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
    url = os.getenv("KG_RAG_URL", "http://kg-rag.railway.internal:8080/query")
    logger.info("Calling KG_RAG_URL=%s", url)
    try:
        response = requests.post(
            url,
            json={"query": user_input},
            timeout=180,
        )
        logger.info("kg-rag status=%s body=%s", response.status_code, response.text[:1000])
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.exception("KG query failed: %s", e)
        return None



def build_matrix_expression_plot(stats_df, gene_name=None, max_rows=8):
    plot_df = stats_df.copy()

    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(
        ["mean_expr", "pct_expr", "n_cells"],
        ascending=[False, False, False]
    ).head(max_rows)

    def make_label(row):
        parts = []
        if "brain_region" in plot_df.columns and pd.notna(row.get("brain_region")):
            parts.append(str(row["brain_region"]))
        if "cell_class" in plot_df.columns and pd.notna(row.get("cell_class")):
            parts.append(str(row["cell_class"]))
        if "cell_type" in plot_df.columns and pd.notna(row.get("cell_type")):
            parts.append(str(row["cell_type"]))
        return " | ".join(parts)

    labels = [make_label(row) for _, row in plot_df.iterrows()]
    y_vals = plot_df["mean_expr"].tolist()

    hover_text = []
    for _, row in plot_df.iterrows():
        text = f"{gene_name or row.get('gene', 'gene')}"
        if "brain_region" in plot_df.columns:
            text += f"<br>region: {row.get('brain_region', '')}"
        if "cell_class" in plot_df.columns:
            text += f"<br>cell class: {row.get('cell_class', '')}"
        if "cell_type" in plot_df.columns:
            text += f"<br>cell type: {row.get('cell_type', '')}"
        text += f"<br>mean_expr {row['mean_expr']:.3f}"
        text += f"<br>pct_expr {row['pct_expr']:.3f}"
        text += f"<br>n_cells {int(row['n_cells'])}"
        hover_text.append(text)

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
            "title": f"{gene_name} expression across matched groups" if gene_name else "Expression across matched groups",
            "xaxis": {"title": "Region | Cell class | Cell type"},
            "yaxis": {"title": "Mean log-normalized expression"},
            "margin": {"l": 70, "r": 20, "t": 60, "b": 180}
        }
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
#            retrieved_info = search_vertex_ai(user_input)
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



def chat(user_input, history):
    global func_flag, init_flag

    if init_flag:
        history.clear()
        initialize(history)

    update_history(history, "user", user_input)

    retrieved_info = None
    graph_json = None

    expression_query = looks_like_expression_query(user_input)
    top_gene_query = (
        wants_top_genes(user_input)
        or ("top" in user_input.lower() and "gene" in user_input.lower())
        or ("highly expressed" in user_input.lower() and "gene" in user_input.lower())
        or ("what are the top" in user_input.lower() and "genes" in user_input.lower())
    )

    if wants_web_search(user_input):
        logger.info("Explicit web/literature intent detected; routing directly to Google")
        try:
            retrieved_info = search_vertex_ai(user_input)
        except Exception:
            logger.exception("Google search failed")
            retrieved_info = None
    else:
        # 1. hard-route top-gene / marker questions to ranked backend first
        if expression_query and top_gene_query:
            try:
                retrieved_info = gene_expression(user_input)
            except Exception:
                logger.exception("top-gene pre-routing failed")
                retrieved_info = None

        # 2. hard-route true expression-value questions to matrix backend
        if not retrieved_info and expression_query:
            if wants_matrix_expression_query(user_input) and not wants_marker_query(user_input):
                try:
                    retrieved_info = matrix_expression(user_input)
                except Exception:
                    logger.exception("matrix pre-routing failed")
                    retrieved_info = None

        # 3. only ask the model to choose a function if nothing already succeeded
        if not retrieved_info:
            chat_message = call_api(history, functions)
            logger.info("First model content: %s", getattr(chat_message, "content", None))
            logger.info("Function call present: %s", bool(chat_message.function_call))

            if chat_message.function_call:
                try:
                    retrieved_info = func_call(user_input, chat_message, history)
                except Exception:
                    logger.exception("func_call failed")
                    retrieved_info = None
            else:
                direct_reply = getattr(chat_message, "content", None)

                # For non-expression conversational follow-ups, allow direct reply.
                # For expression/top-gene questions, prefer dataset/tool routing.
                if direct_reply and direct_reply.strip() and not expression_query:
                    logger.info("Returning direct model reply without tool call")
                    update_history(history, "assistant", direct_reply)
                    return direct_reply, history, None

        # 4. heuristic routing fallback
        if not retrieved_info and expression_query:
            if top_gene_query or wants_marker_query(user_input):
                logger.info("Routing to marker backend")
                try:
                    retrieved_info = gene_expression(user_input)
                except Exception:
                    logger.exception("marker backend failed")
                    retrieved_info = None

            if not retrieved_info and wants_matrix_expression_query(user_input):
                logger.info("Routing to matrix expression backend")
                try:
                    retrieved_info = matrix_expression(user_input)
                except Exception:
                    logger.exception("matrix backend failed")
                    retrieved_info = None

            if not retrieved_info:
                logger.info("Expression fallback: matrix first, then marker")
                try:
                    retrieved_info = matrix_expression(user_input)
                except Exception:
                    logger.exception("matrix fallback failed")
                    retrieved_info = None

                if not retrieved_info:
                    try:
                        retrieved_info = gene_expression(user_input)
                    except Exception:
                        logger.exception("marker fallback failed")
                        retrieved_info = None

        # 5. KG fallback
        if not retrieved_info:
            lowered = user_input.lower()
            kg_terms = [
                "drug", "drugs", "target", "targets", "disease",
                "association", "associated", "pathway", "pathways",
                "implicated", "implication"
            ]
            if any(term in lowered for term in kg_terms):
                try:
                    retrieved_info = query_kg_rag(user_input)
                except Exception:
                    logger.exception("KG query failed")
                    retrieved_info = None

        # 6. Google fallback
        if not retrieved_info:
            logger.info("Calling Google Vertex AI API...")
            try:
                retrieved_info = search_vertex_ai(user_input)
            except Exception:
                logger.exception("Google search failed")
                retrieved_info = None

    if not retrieved_info:
        retrieved_info = ""

    if isinstance(retrieved_info, dict):
        graph_json = retrieved_info.get("graph_json")
        retrieved_text = retrieved_info.get("text", "")
    else:
        retrieved_text = retrieved_info

    if not isinstance(retrieved_text, str):
        retrieved_text = str(retrieved_text)

    retrieved_text = retrieved_text[:4000]

    synthesis_messages = history[:] + [
        {
            "role": "system",
            "content": (
                "Answer the user's question directly using the retrieved information below. "
                "Do not mention search tools or internal routing."
            )
        },
        {
            "role": "user",
            "content": f"Retrieved information:\n{retrieved_text}"
        }
    ]

    final_message = call_api(synthesis_messages).content
    logger.info("Final message returned to UI: %r", final_message)
    update_history(history, "assistant", final_message)

    return final_message, history, graph_json
