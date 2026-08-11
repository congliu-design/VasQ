import logging
import threading
import time
import traceback

from flask import Flask, jsonify, request

from kg_rag.generation.text_generation import generate_answer


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

_inflight_queries = set()
_inflight_lock = threading.Lock()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/query", methods=["POST"])
def query_kg_rag():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get("query", "")).strip()
    if not user_input:
        return jsonify({"error": "A non-empty query is required."}), 400

    query_key = " ".join(user_input.lower().split())
    with _inflight_lock:
        if query_key in _inflight_queries:
            app.logger.warning(
                "Duplicate KG query rejected while first request is running: %s",
                user_input,
            )
            return jsonify(
                {"error": "An identical KG query is already running."}
            ), 409
        _inflight_queries.add(query_key)

    started_at = time.monotonic()
    app.logger.info("KG query received: %s", user_input)

    try:
        output = generate_answer(
            user_input,
            chat_model_id="gpt-4o",
            edge_evidence=False,
        )
        elapsed = time.monotonic() - started_at
        app.logger.info(
            "KG query succeeded elapsed=%.1fs output_length=%s output_tail:\n%s",
            elapsed,
            len(output),
            output[-4000:],
        )
        return jsonify({"result": output})

    except Exception:
        elapsed = time.monotonic() - started_at
        error_text = traceback.format_exc()
        app.logger.error(
            "KG query failed elapsed=%.1fs:\n%s",
            elapsed,
            error_text,
        )
        return jsonify({"error": error_text}), 500

    finally:
        with _inflight_lock:
            _inflight_queries.discard(query_key)
