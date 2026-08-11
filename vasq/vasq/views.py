import json
import logging
import uuid

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render

from .functions import ChatCancelled, chat


logger = logging.getLogger(__name__)

# Normal chat history is retained for 24 hours.
CHAT_HISTORY_TIMEOUT_SECONDS = 24 * 60 * 60
# Cancellation markers only need to outlive an in-flight request.
REQUEST_CANCEL_TIMEOUT_SECONDS = 10 * 60


def index(request):
    return render(request, "vasq/index.html")


def normalize_chat_id(raw_chat_id):
    """Validate a browser-tab chat ID or generate a safe replacement."""

    try:
        return str(uuid.UUID(str(raw_chat_id)))
    except (TypeError, ValueError, AttributeError):
        return str(uuid.uuid4())


def normalize_request_id(raw_request_id):
    """Return a valid request UUID, or None when cancellation input is bad."""

    try:
        return str(uuid.UUID(str(raw_request_id)))
    except (TypeError, ValueError, AttributeError):
        return None


def cancellation_key(request_id):
    return f"vasq:cancelled-request:{request_id}"


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

    request_id = normalize_request_id(
        payload.get("request_id")
    ) or str(uuid.uuid4())

    cancel_request = bool(
        payload.get("cancel_request", False)
    )

    return message, reset_history, chat_id, request_id, cancel_request


def api_chat(request):
    try:
        (
            user_input,
            reset_history,
            chat_id,
            request_id,
            cancel_request,
        ) = parse_input(request)

        cancel_key = cancellation_key(request_id)

        if cancel_request:
            cache.set(
                cancel_key,
                True,
                timeout=REQUEST_CANCEL_TIMEOUT_SECONDS,
            )
            logger.info(
                "Cancellation requested chat_id=%s request_id=%s",
                chat_id,
                request_id,
            )
            return JsonResponse({
                "response": "Request stopped.",
                "stopped": True,
                "request_id": request_id,
                "chat_id": chat_id,
            })

        if not user_input:
            return JsonResponse(
                {
                    "response": "Please enter a question.",
                    "graph_json": None,
                    "request_id": request_id,
                    "chat_id": chat_id,
                },
                status=400,
            )

        logger.info(
            (
                "Received message chat_id=%s request_id=%s "
                "reset_history=%s message=%s"
            ),
            chat_id,
            request_id,
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

        try:
            content, updated_history, graph_json = chat(
                user_input,
                history,
                should_stop=lambda: bool(cache.get(cancel_key)),
            )
        except ChatCancelled:
            cache.delete(cancel_key)
            logger.info(
                "Stopped request chat_id=%s request_id=%s",
                chat_id,
                request_id,
            )
            return JsonResponse(
                {
                    "response": "Request stopped.",
                    "graph_json": None,
                    "stopped": True,
                    "request_id": request_id,
                    "chat_id": chat_id,
                },
                status=409,
            )

        # The provider call may not be interruptible, but a cancelled result
        # must never reach the UI or overwrite the conversation history.
        if cache.get(cancel_key):
            cache.delete(cancel_key)
            logger.info(
                "Discarding cancelled result chat_id=%s request_id=%s",
                chat_id,
                request_id,
            )
            return JsonResponse(
                {
                    "response": "Request stopped.",
                    "graph_json": None,
                    "stopped": True,
                    "request_id": request_id,
                    "chat_id": chat_id,
                },
                status=409,
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
                "request_id": request_id,
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
