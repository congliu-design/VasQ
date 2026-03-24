from flask import Flask, request, jsonify
import subprocess
import traceback
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/query", methods=["POST"])
def query_kg_rag():
    user_input = request.json.get("query", "")

    base_dir = "/app"
    script_path = "kg_rag.generation.text_generation"

    command = [
        "python", "-m", script_path,
        "-g", "gpt-4",
        "--query", user_input
    ]

    app.logger.info("KG query received: %s", user_input)
    app.logger.info("Running command: %s", " ".join(command))

    try:
        result = subprocess.check_output(
            command,
            cwd=base_dir,
            stderr=subprocess.STDOUT
        )
        output = result.decode("utf-8", errors="replace")
        app.logger.info("KG success output:\n%s", output[:4000])
        return jsonify({"result": output})

    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8", errors="replace")
        app.logger.error("KG subprocess failed:\n%s", output)
        return jsonify({"error": output}), 500

    except Exception:
        err = traceback.format_exc()
        app.logger.error("KG unexpected failure:\n%s", err)
        return jsonify({"error": err}), 500

