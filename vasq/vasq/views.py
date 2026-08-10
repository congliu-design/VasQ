import json
import logging
import uuid

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render

from .functions import chat


logger = logging.getLogger(__name__)

# Normal chat history is retained for 24 hours.
CHAT_HISTORY_TIMEOUT_SECONDS = 24 * 60 * 60


def index(request):
    return render(request, "vasq/index.html")


def normalize_chat_id(raw_chat_id):
    """Validate a browser-tab chat ID or generate a safe replacement."""

    try:
        return str(uuid.UUID(str(raw_chat_id)))
    except (TypeError, ValueError, AttributeError):
        return str(uuid.uuid4())


def parse_input(request):
    payload = json.loads(request.body)

    message = str(
        payload.get("message", "")
    ).strip()

    reset_history = bool(
        payload.get("reset_history", False)
    )

    chat_id = normalize_chat_id(
        payload.get("chat_id")
    )

    return message, reset_history, chat_id


def api_chat(request):
    try:
        user_input, reset_history, chat_id = parse_input(request)

        if not user_input:
            return JsonResponse(
                {
                    "response": "Please enter a question.",
                    "graph_json": None,
                    "chat_id": chat_id,
                },
                status=400,
            )

        logger.info(
            "Received message chat_id=%s reset_history=%s message=%s",
            chat_id,
            reset_history,
            user_input,
        )

        # Every browser tab has a different cache key.
        history_key = f"vasq:chat-history:{chat_id}"

        if reset_history:
            history = []
        else:
            history = cache.get(history_key, [])

        if not isinstance(history, list):
            history = []

        content, updated_history, graph_json = chat(
            user_input,
            history,
        )

        # Queue questions are independent and do not need to remain cached.
        if reset_history:
            cache.delete(history_key)
        else:
            cache.set(
                history_key,
                updated_history,
                timeout=CHAT_HISTORY_TIMEOUT_SECONDS,
            )

        return JsonResponse(
            {
                "response": content,
                "graph_json": graph_json,
                "chat_id": chat_id,
            }
        )

    except Exception:
        logger.exception("api_chat failed")

        return JsonResponse(
            {
                "response": (
                    "Sorry, something went wrong while "
                    "processing your message."
                ),
                "graph_json": None,
            },
            status=500,
        )
