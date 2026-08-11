import ast, json, openai, os, requests, sys, time, torch
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from kg_rag.config_loader import *
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential

config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    timeout_seconds = float(os.environ.get("SPOKE_API_TIMEOUT_SECONDS", "20"))
    return requests.get(uri, params=params, timeout=timeout_seconds)

@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(2))
def get_context_using_spoke_api(
    node_value,
    node_type="Disease",
    attribute="name",
    neighbor_node_types=None,
):
    """Fetch one-hop SPOKE context for a typed node.

    ``neighbor_node_types`` limits the returned neighborhood.  For example,
    a Gene query can request only Disease, Pathway, and Compound neighbors.
    """
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    result.raise_for_status()
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]

    if neighbor_node_types:
        requested_neighbor_types = set(neighbor_node_types)
        filtered_node_types = [
            candidate
            for candidate in filtered_node_types
            if candidate in requested_neighbor_types
        ]

    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    result.raise_for_status()
    node_context = result.json()

    if not isinstance(node_context, list) or not node_context:
        return "", pd.DataFrame(
            columns=["source", "edge_type", "target", "provenance", "evidence", "predicate", "context"]
        )

    nbr_nodes = []
    nbr_edges = []
    root_data = None

    for item in node_context:
        item_data = item.get("data", {})
        neo4j_type = item_data.get("neo4j_type", "")

        if item_data.get("neo4j_root") == 1:
            root_data = item_data

        if "_" not in neo4j_type:
            try:
                if neo4j_type == "Protein":
                    nbr_nodes.append((neo4j_type, item_data["id"], item_data["properties"]["description"]))
                else:
                    nbr_nodes.append((neo4j_type, item_data["id"], item_data["properties"]["name"]))
            except:
                nbr_nodes.append((neo4j_type, item_data["id"], item_data["properties"]["identifier"]))
        elif "_" in neo4j_type:
            try:
                provenance = ", ".join(item_data["properties"]["sources"])
            except:
                try:
                    provenance = item_data["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item_data["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item_data["properties"]["pmid_list"])
                            pmid_list = ["pubmedId:" + str(x) for x in pmid_list]
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item_data["properties"]
            except:
                evidence = None
            nbr_edges.append((item_data["source"], neo4j_type, item_data["target"], provenance, evidence))

    if not nbr_nodes or not nbr_edges:
        return "", pd.DataFrame(
            columns=["source", "edge_type", "target", "provenance", "evidence", "predicate", "context"]
        )

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')

    if root_data:
        root_properties = root_data.get("properties", {})
        root_source = root_properties.get("source") or root_properties.get("sources") or "SPOKE-KG"
        if isinstance(root_source, list):
            root_source = ", ".join(dict.fromkeys(map(str, root_source)))
        root_identifier = root_properties.get("identifier", node_value)
        context += (
            f" {node_type} {node_value} has identifier {root_identifier} "
            f"and provenance {root_source}."
        )

    return context, merge_2

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    response = openai.ChatCompletion.create(
        temperature=temperature,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'

def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)

def stream_out(output):
    CHUNK_SIZE = max(1, int(round(len(output)/50)))
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")

def get_gpt35():
    chat_model_id = 'gpt-4o' if openai.api_type == 'azure' else 'gpt-4o'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id


def biomedical_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()

    system_prompt = (
        "You are an expert biomedical entity extractor. "
        "Extract explicitly mentioned biomedical entities from the user's sentence. "
        "Return valid JSON with exactly these keys:\n"
        '{'
        '"Diseases": [], '
        '"Genes_Proteins": [], '
        '"Drugs": [], '
        '"Pathways": []'
        '}\n'
        "Rules:\n"
        "- Include genes and proteins like PLVAP, CLDN5, APOE, VEGFA.\n"
        "- Include diseases like Alzheimer's disease, glioblastoma, Parkinson's disease.\n"
        "- Include drugs if explicitly mentioned.\n"
        "- Do not invent entities.\n"
        "- If nothing is found for a category, return an empty list.\n"
        "- Return JSON only."
    )

    resp = get_GPT_response(
        text,
        system_prompt,
        chat_model_id,
        chat_deployment_id,
        temperature=0
    )

    try:
        entity_dict = json.loads(resp)
        return entity_dict
    except Exception:
        return {
            "Diseases": [],
            "Genes_Proteins": [],
            "Drugs": [],
            "Pathways": []
        }

def find_exact_node_matches(entity, node_context_df):
    if node_context_df is None or "node_name" not in node_context_df.columns:
        return []

    entity_norm = str(entity).strip().lower()
    names = node_context_df["node_name"].astype(str)

    exact = node_context_df[names.str.strip().str.lower() == entity_norm]
    if not exact.empty:
        print(f"Exact matches for {entity}: {exact['node_name'].tolist()[:10]}")
        return exact["node_name"].tolist()

    contains = node_context_df[names.str.strip().str.lower().str.contains(entity_norm, regex=False)]
    if not contains.empty:
        print(f"Substring matches for {entity}: {contains['node_name'].tolist()[:10]}")
        return contains["node_name"].tolist()[:5]

    print(f"No exact or substring node_name matches for {entity}")
    return []


def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)


def select_balanced_neighbor_rows(
    context_table,
    neighbor_node_types,
    max_per_type,
):
    """Keep a small, balanced set of direct SPOKE relationships."""
    if context_table is None or context_table.empty:
        return pd.DataFrame()

    selected_parts = []
    source_names = context_table["source"].fillna("").astype(str)
    target_names = context_table["target"].fillna("").astype(str)

    for neighbor_type in neighbor_node_types or []:
        prefix = f"{neighbor_type} "
        type_rows = context_table[
            source_names.str.startswith(prefix)
            | target_names.str.startswith(prefix)
        ].drop_duplicates(subset=["source", "edge_type", "target"])

        if type_rows.empty:
            continue

        # Preserve several edge types instead of allowing a single dense
        # relationship class to consume the whole quota.
        edge_groups = [
            group
            for _edge_type, group in type_rows.groupby("edge_type", sort=True)
        ]
        chosen_indices = []
        row_offset = 0
        while len(chosen_indices) < max_per_type:
            added_row = False
            for group in edge_groups:
                if row_offset < len(group):
                    chosen_indices.append(group.index[row_offset])
                    added_row = True
                    if len(chosen_indices) >= max_per_type:
                        break
            if not added_row:
                break
            row_offset += 1

        selected_parts.append(context_table.loc[chosen_indices])

    if not selected_parts:
        return context_table.head(max_per_type).copy()

    return (
        pd.concat(selected_parts, ignore_index=False)
        .drop_duplicates(subset=["source", "edge_type", "target"])
        .copy()
    )

def retrieve_context(
    question,
    vectorstore,
    embedding_function,
    node_context_df,
    context_volume,
    context_sim_threshold,
    context_sim_min_threshold,
    edge_evidence,
    api=True
):
    entity_dict = biomedical_entity_extractor(question)
    extracted_entities = {
        key: [str(item).strip() for item in entity_dict.get(key, []) if str(item).strip()]
        for key in ["Genes_Proteins", "Diseases", "Drugs", "Pathways"]
    }
    print(f"Biomedical entities extracted by type: {extracted_entities}")

    # Each lookup keeps the SPOKE node type.  Only Diseases use the existing
    # disease-only CSV/Chroma store for exact and fuzzy resolution.
    spoke_lookups = []

    for entity in extracted_entities["Genes_Proteins"]:
        spoke_lookups.append(
            {
                "value": entity,
                "node_type": "Gene",
                "attribute": "name",
                "neighbor_node_types": ["Disease", "Pathway", "Compound"],
                "source": "direct gene lookup",
            }
        )

    for entity in extracted_entities["Diseases"]:
        disease_names = find_exact_node_matches(entity, node_context_df)

        if not disease_names:
            disease_hits = vectorstore.similarity_search_with_score(entity, k=3)
            disease_names = [
                doc.page_content
                for doc, _score in disease_hits
                if getattr(doc, "page_content", None)
            ]
            if disease_names:
                print(f"Fuzzy disease vectorstore matches for {entity}: {disease_names[:3]}")

        if not disease_names:
            # An exact disease name may exist in SPOKE even when it is absent
            # from the local disease-only vectorstore.
            disease_names = [entity]

        for disease_name in disease_names[:3]:
            spoke_lookups.append(
                {
                    "value": disease_name,
                    "node_type": "Disease",
                    "attribute": "name",
                    "neighbor_node_types": None,
                    "source": "disease exact/fuzzy lookup",
                }
            )

    # These direct lookups avoid incorrectly sending drugs and pathways to the
    # disease-only vectorstore.  Some SPOKE Compound nodes require an identifier;
    # failures are logged and skipped rather than broadened to an unrelated disease.
    for entity in extracted_entities["Drugs"]:
        spoke_lookups.append(
            {
                "value": entity,
                "node_type": "Compound",
                "attribute": "name",
                "neighbor_node_types": None,
                "source": "direct compound lookup",
            }
        )

    for entity in extracted_entities["Pathways"]:
        spoke_lookups.append(
            {
                "value": entity,
                "node_type": "Pathway",
                "attribute": "name",
                "neighbor_node_types": None,
                "source": "direct pathway lookup",
            }
        )

    # If GPT extracted no explicit entity, the question-level fallback remains
    # disease-only because that is what the bundled Chroma collection contains.
    if not spoke_lookups:
        print("No explicit entity found; using disease question-level vectorstore fallback")
        question_hits = vectorstore.similarity_search_with_score(question, k=5)
        for doc, _score in question_hits:
            if getattr(doc, "page_content", None):
                spoke_lookups.append(
                    {
                        "value": doc.page_content,
                        "node_type": "Disease",
                        "attribute": "name",
                        "neighbor_node_types": None,
                        "source": "question-level disease fallback",
                    }
                )

    deduplicated_lookups = []
    seen_lookups = set()
    for lookup in spoke_lookups:
        lookup_key = (
            lookup["node_type"],
            lookup["attribute"],
            str(lookup["value"]).strip().lower(),
        )
        if lookup_key not in seen_lookups:
            seen_lookups.add(lookup_key)
            deduplicated_lookups.append(lookup)

    if not deduplicated_lookups:
        return "No relevant knowledge graph context found."

    print(
        "Typed SPOKE lookups: "
        + str(
            [
                f"{item['node_type']}/{item['attribute']}/{item['value']}"
                for item in deduplicated_lookups
            ]
        )
    )

    question_embedding = None
    max_context_per_node = max(1, int(context_volume / len(deduplicated_lookups)))
    extracted_context_parts = []

    for lookup in deduplicated_lookups:
        if not api and lookup["node_type"] == "Disease":
            matches = node_context_df[
                node_context_df.node_name == lookup["value"]
            ]
            if matches.empty:
                print(f"No local disease context match for node: {lookup['value']}")
                continue
            node_context = matches.node_context.values[0]
            context_table = pd.DataFrame()
        else:
            try:
                node_context, context_table = get_context_using_spoke_api(
                    lookup["value"],
                    node_type=lookup["node_type"],
                    attribute=lookup["attribute"],
                    neighbor_node_types=lookup["neighbor_node_types"],
                )
            except Exception as exc:
                print(
                    f"SPOKE lookup failed for {lookup['node_type']} "
                    f"{lookup['value']}: {exc}"
                )
                continue

        if context_table is not None and not context_table.empty and "context" in context_table.columns:
            node_context_list = context_table["context"].dropna().astype(str).tolist()
        else:
            node_context_list = [
                item.strip()
                for item in str(node_context).split(". ")
                if item.strip()
            ]

        if not node_context_list:
            print(f"No SPOKE relationships returned for {lookup['node_type']} {lookup['value']}")
            continue

        selected_table = pd.DataFrame()

        if (
            lookup["source"] == "direct gene lookup"
            and context_table is not None
            and not context_table.empty
        ):
            max_per_type = max(
                1,
                int(os.environ.get("SPOKE_MAX_RELATIONSHIPS_PER_TYPE", "20")),
            )
            selected_table = select_balanced_neighbor_rows(
                context_table,
                lookup["neighbor_node_types"],
                max_per_type,
            )
            selected_context = (
                selected_table["context"].dropna().astype(str).tolist()
            )
            selected_counts = {
                neighbor_type: int(
                    selected_table["source"].fillna("").astype(str).str.startswith(
                        f"{neighbor_type} "
                    ).sum()
                    + selected_table["target"].fillna("").astype(str).str.startswith(
                        f"{neighbor_type} "
                    ).sum()
                )
                for neighbor_type in lookup["neighbor_node_types"]
            }
            print(
                f"Direct Gene context selected for {lookup['value']}: "
                f"{selected_counts}"
            )
        else:
            if question_embedding is None:
                question_embedding = embedding_function.embed_query(question)

            node_context_embeddings = embedding_function.embed_documents(
                node_context_list
            )
            similarities = [
                float(
                    cosine_similarity(
                        np.array(question_embedding).reshape(1, -1),
                        np.array(node_context_embedding).reshape(1, -1),
                    )[0][0]
                )
                for node_context_embedding in node_context_embeddings
            ]

            ranked_similarities = sorted(
                [(score, index) for index, score in enumerate(similarities)],
                reverse=True,
            )
            percentile_threshold = float(
                np.percentile(similarities, context_sim_threshold)
            )
            selected_indices = [
                index
                for score, index in ranked_similarities
                if score >= percentile_threshold
                and score >= context_sim_min_threshold
            ][:max_context_per_node]

            selected_context = [
                node_context_list[index]
                for index in selected_indices
            ]
            if context_table is not None and not context_table.empty:
                selected_table = context_table[
                    context_table.context.isin(selected_context)
                ].copy()

        if not selected_context:
            print(
                f"No relationships passed the similarity threshold for "
                f"{lookup['node_type']} {lookup['value']}"
            )
            continue

        if edge_evidence and not selected_table.empty:
            selected_table.loc[:, "context"] = (
                selected_table.source
                + " "
                + selected_table.predicate.str.lower()
                + " "
                + selected_table.target
                + " and Provenance of this association is "
                + selected_table.provenance
                + " and attributes associated with this association is in the following JSON format:\n "
                + selected_table.evidence.astype("str")
                + "\n\n"
            )
            extracted_context_parts.append(
                selected_table.context.str.cat(sep=" ")
            )
        else:
            extracted_context_parts.append(" ".join(selected_context))

    final_context = " ".join(
        part for part in extracted_context_parts if str(part).strip()
    ).strip()
    return final_context or "No relevant knowledge graph context found."
