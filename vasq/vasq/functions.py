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
    global func_flag
    func_flag = True
    content = None

    func_name = chat_message.function_call.name
    print("Calling", func_name, "...")
    args = {"user_input":user_input}
    content = globals()[func_name](**args)
        
    func_flag = False
    return content


### Gene Expression Functions ###
DATA_DIR = "/data"
EXPR_PATH = os.path.join(DATA_DIR, "expression_markers.csv")
REGION_META_PATH = os.path.join(DATA_DIR, "region_metadata.csv")

#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
#EXPR_PATH = os.path.join(DATA_DIR, "expression_markers.csv")
#REGION_META_PATH = os.path.join(DATA_DIR, "regioIn_metadata.csv")


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
            model="gpt-4o",
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
        model="gpt-4o",
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
#        model="gpt-4o",
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
        model="gpt-4o",
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
        model="gpt-4o",
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

        header = f"Expression of {', '.join(gene_names)}"
        if cell_types:
            header += f" in cell types related to {', '.join(cell_types)}"
        if regions and not all_regions_flag:
            header += f" in regions {', '.join(regions)}"

        sections = [
            header,
            format_single_gene_expression_rows(df, max_rows=20)
        ]

        return prefix + "\n".join(sections).strip()


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

def chat(user_input, history):
    global func_flag, init_flag

    if init_flag:
        history.clear()
        initialize(history)

    update_history(history, "user", user_input)

    retrieved_info = None
    chat_message = call_api(history, functions)

    if chat_message.function_call:
        try:
            retrieved_info = func_call(user_input, chat_message, history)
        except Exception as e:
            logger.exception("func_call failed: %s", e)
            retrieved_info = None

    # If function calling missed it or returned nothing useful, try local gene-expression data first
    if not retrieved_info and looks_like_expression_query(user_input):
        logger.info("Heuristic routing to gene_expression first")
        try:
            retrieved_info = gene_expression(user_input)
        except Exception as e:
            logger.exception("gene_expression heuristic failed: %s", e)
            retrieved_info = None

    # KG-RAG before Google for graph-style biomedical questions
    if not retrieved_info:
        lowered = user_input.lower()
        kg_terms = [
            "drug", "drugs", "target", "targets", "disease",
            "association", "associated", "pathway", "pathways"
        ]
        if any(term in lowered for term in kg_terms):
            try:
                retrieved_info = query_kg_rag(user_input)
            except Exception as e:
                logger.exception("KG query failed: %s", e)
                retrieved_info = None

    if not retrieved_info:
        logger.info("Calling Google Search API...")
        try:
            retrieved_info = search_google(user_input)
            if not retrieved_info:
                logger.warning("Google search returned no usable results")
        except Exception as e:
            logger.exception("Google search failed")
            retrieved_info = None

    if not retrieved_info:
        retrieved_info = ""

    if not isinstance(retrieved_info, str):
        retrieved_info = str(retrieved_info)

    retrieved_info = retrieved_info[:4000]

    if retrieved_info:
        update_history(history, "system", retrieved_info)

    final_message = call_api(history).content
    update_history(history, "assistant", final_message)
    return final_message, history

