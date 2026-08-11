import argparse
import threading

from kg_rag.utility import *


SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(
    config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"]
)
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(
    config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]
)
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"
]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"
]
TEMPERATURE = config_data["LLM_TEMPERATURE"]


class LazyVectorStore:
    """Load the disease-only Chroma store only when a disease needs it."""

    def __init__(self, vector_db_path, sentence_embedding_model):
        self.vector_db_path = vector_db_path
        self.sentence_embedding_model = sentence_embedding_model
        self._value = None
        self._lock = threading.Lock()

    def _get(self):
        if self._value is None:
            with self._lock:
                if self._value is None:
                    print("Loading disease vectorstore", flush=True)
                    self._value = load_chroma(
                        self.vector_db_path,
                        self.sentence_embedding_model,
                    )
        return self._value

    def similarity_search_with_score(self, *args, **kwargs):
        return self._get().similarity_search_with_score(*args, **kwargs)


class LazyEmbeddingFunction:
    """Load the heavy context reranker only when semantic ranking is needed."""

    def __init__(self, sentence_embedding_model):
        self.sentence_embedding_model = sentence_embedding_model
        self._value = None
        self._lock = threading.Lock()

    def _get(self):
        if self._value is None:
            with self._lock:
                if self._value is None:
                    print("Loading context embedding model", flush=True)
                    self._value = load_sentence_transformer(
                        self.sentence_embedding_model
                    )
        return self._value

    def embed_query(self, *args, **kwargs):
        return self._get().embed_query(*args, **kwargs)

    def embed_documents(self, *args, **kwargs):
        return self._get().embed_documents(*args, **kwargs)


# These objects are created once by the Gunicorn worker and reused by requests.
# Direct Gene lookups do not cause either model to load.
vectorstore = LazyVectorStore(
    VECTOR_DB_PATH,
    SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL,
)
embedding_function_for_context_retrieval = LazyEmbeddingFunction(
    SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL
)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)


def generate_answer(
    question,
    chat_model_id="gpt-4o",
    edge_evidence=False,
):
    """Run one KG-RAG request without starting a second Python process."""
    chat_deployment_id = (
        chat_model_id
        if openai.api_type == "azure"
        else None
    )
    context = retrieve_context(
        question,
        vectorstore,
        embedding_function_for_context_retrieval,
        node_context_df,
        CONTEXT_VOLUME,
        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
        bool(edge_evidence),
    )
    enriched_prompt = f"Context: {context}\nQuestion: {question}"
    return get_GPT_response(
        enriched_prompt,
        SYSTEM_PROMPT,
        chat_model_id,
        chat_deployment_id,
        temperature=TEMPERATURE,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        type=str,
        default="gpt-4o",
        help="GPT model selection",
    )
    parser.add_argument(
        "-e",
        action="store_true",
        help="Show evidence of association from the graph",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="User question (non-interactive)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    question = args.query or input("Enter your question: ")
    output = generate_answer(
        question,
        chat_model_id=args.g,
        edge_evidence=args.e,
    )
    stream_out(output)


if __name__ == "__main__":
    main()
