import csv
import json
import logging
import os
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor

from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse

from .functions import ChatCancelled, chat


logger = logging.getLogger(__name__)

# Normal chat history is retained for 24 hours.
CHAT_HISTORY_TIMEOUT_SECONDS = 24 * 60 * 60
# Cancellation markers only need to outlive an in-flight request.
REQUEST_CANCEL_TIMEOUT_SECONDS = 10 * 60
# Completed job results remain available long enough for a browser to poll them.
CHAT_JOB_TIMEOUT_SECONDS = 60 * 60
# Full expression rows are cached separately from the small polling payload.
EXPRESSION_TABLE_TIMEOUT_SECONDS = 60 * 60
EXPRESSION_TABLE_DEFAULT_PAGE_SIZE = 50
EXPRESSION_TABLE_MAX_PAGE_SIZE = 100


def _background_worker_count():
    try:
        return max(1, int(os.getenv("VASQ_BACKGROUND_WORKERS", "1")))
    except (TypeError, ValueError):
        return 1


# Railway runs one Gunicorn worker for VasQ. A small in-process executor lets the
# HTTP request return immediately while the long scientific workflow continues.
_CHAT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_background_worker_count(),
    thread_name_prefix="vasq-chat",
)


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


def chat_job_key(request_id):
    return f"vasq:chat-job:{request_id}"


def expression_table_key(request_id):
    return f"vasq:expression-table:{request_id}"


def cache_chat_job(request_id, payload):
    cache.set(
        chat_job_key(request_id),
        payload,
        timeout=CHAT_JOB_TIMEOUT_SECONDS,
    )


def chat_status_url(request_id):
    return reverse(
        "api_chat_status",
        kwargs={"request_id": request_id},
    )


def expression_table_url(request_id):
    return reverse(
        "api_expression_table",
        kwargs={"request_id": request_id},
    )


def expression_table_download_url(request_id):
    return reverse(
        "api_expression_table_download",
        kwargs={"request_id": request_id},
    )


def encode_expression_table(payload):
    """Compress large deterministic tables before placing them in cache."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return zlib.compress(serialized, level=6)


def get_cached_expression_table(request_id):
    cached = cache.get(expression_table_key(str(request_id)))
    if isinstance(cached, dict):
        return cached
    if not isinstance(cached, (bytes, bytearray)):
        return None
    try:
        return json.loads(zlib.decompress(cached).decode("utf-8"))
    except (ValueError, TypeError, zlib.error, UnicodeDecodeError):
        logger.exception(
            "Could not decode expression table request_id=%s",
            request_id,
        )
        return None


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


def run_chat_job(
    user_input,
    reset_history,
    chat_id,
    request_id,
):
    """Run a long chat turn outside the request/response lifecycle."""

    cancel_key = cancellation_key(request_id)
    history_key = f"vasq:chat-history:{chat_id}"
    started_at = time.time()

    cache_chat_job(
        request_id,
        {
            "status": "running",
            "request_id": request_id,
            "chat_id": chat_id,
            "started_at": started_at,
        },
    )
    logger.info(
        "Background chat started chat_id=%s request_id=%s",
        chat_id,
        request_id,
    )

    try:
        if cache.get(cancel_key):
            raise ChatCancelled("Chat request stopped before it started.")

        if reset_history:
            history = []
        else:
            history = cache.get(history_key, [])

        if not isinstance(history, list):
            history = []

        content, updated_history, graph_json, expression_table = chat(
            user_input,
            history,
            should_stop=lambda: bool(cache.get(cancel_key)),
        )

        # Some provider calls cannot be interrupted. Never publish a result
        # that finished after the browser requested cancellation.
        if cache.get(cancel_key):
            raise ChatCancelled("Chat request stopped while processing.")

        if reset_history:
            cache.delete(history_key)
        else:
            cache.set(
                history_key,
                updated_history,
                timeout=CHAT_HISTORY_TIMEOUT_SECONDS,
            )

        table_url = None
        table_download_url = None
        if isinstance(expression_table, dict) and expression_table.get("rows"):
            cache.set(
                expression_table_key(request_id),
                encode_expression_table(expression_table),
                timeout=EXPRESSION_TABLE_TIMEOUT_SECONDS,
            )
            table_url = expression_table_url(request_id)
            table_download_url = expression_table_download_url(request_id)

        cache_chat_job(
            request_id,
            {
                "status": "completed",
                "response": content,
                "graph_json": graph_json,
                "expression_table_url": table_url,
                "expression_table_download_url": table_download_url,
                "request_id": request_id,
                "chat_id": chat_id,
                "started_at": started_at,
                "finished_at": time.time(),
            },
        )
        logger.info(
            "Background chat completed chat_id=%s request_id=%s elapsed=%.1fs",
            chat_id,
            request_id,
            time.time() - started_at,
        )

    except ChatCancelled:
        cache_chat_job(
            request_id,
            {
                "status": "stopped",
                "response": "Request stopped.",
                "graph_json": None,
                "stopped": True,
                "request_id": request_id,
                "chat_id": chat_id,
                "started_at": started_at,
                "finished_at": time.time(),
            },
        )
        logger.info(
            "Background chat stopped chat_id=%s request_id=%s",
            chat_id,
            request_id,
        )

    except Exception:
        logger.exception(
            "Background chat failed chat_id=%s request_id=%s",
            chat_id,
            request_id,
        )
        cache_chat_job(
            request_id,
            {
                "status": "failed",
                "response": (
                    "Sorry, something went wrong while processing your message."
                ),
                "graph_json": None,
                "request_id": request_id,
                "chat_id": chat_id,
                "started_at": started_at,
                "finished_at": time.time(),
            },
        )

    finally:
        cache.delete(cancel_key)


def _positive_int(value, default, maximum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    parsed = max(1, parsed)
    if maximum is not None:
        parsed = min(parsed, int(maximum))
    return parsed


def _filtered_expression_rows(payload, query_params):
    rows = payload.get("rows") or []
    filter_keys = payload.get("filter_keys") or []
    active_filters = {
        key: str(query_params.get(key, "") or "").strip()
        for key in filter_keys
    }
    active_filters = {
        key: value
        for key, value in active_filters.items()
        if value
    }
    if not active_filters:
        return rows, {}
    filtered = [
        row
        for row in rows
        if all(
            str(row.get(key, "")) == value
            for key, value in active_filters.items()
        )
    ]
    return filtered, active_filters


def api_expression_table(request, request_id):
    """Return one filtered page of the complete deterministic matrix table."""
    if request.method != "GET":
        return JsonResponse({"response": "Method not allowed."}, status=405)

    payload = get_cached_expression_table(request_id)
    if not isinstance(payload, dict):
        return JsonResponse(
            {
                "response": (
                    "This expression table is no longer available. "
                    "Please run the question again."
                )
            },
            status=404,
        )

    filtered_rows, active_filters = _filtered_expression_rows(
        payload,
        request.GET,
    )
    page_size = _positive_int(
        request.GET.get("page_size"),
        EXPRESSION_TABLE_DEFAULT_PAGE_SIZE,
        maximum=EXPRESSION_TABLE_MAX_PAGE_SIZE,
    )
    total_rows = len(filtered_rows)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = min(
        _positive_int(request.GET.get("page"), 1),
        total_pages,
    )
    start = (page - 1) * page_size
    end = start + page_size

    return JsonResponse({
        "columns": payload.get("columns") or [],
        "filter_keys": payload.get("filter_keys") or [],
        "filters": payload.get("filters") or {},
        "active_filters": active_filters,
        "rows": filtered_rows[start:end],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        },
        "minimum_cells_per_group": payload.get("minimum_cells_per_group"),
    })


def api_expression_table_download(request, request_id):
    """Download every eligible matrix row as CSV without involving the LLM."""
    if request.method != "GET":
        return JsonResponse({"response": "Method not allowed."}, status=405)

    payload = get_cached_expression_table(request_id)
    if not isinstance(payload, dict):
        return JsonResponse(
            {
                "response": (
                    "This expression table is no longer available. "
                    "Please run the question again."
                )
            },
            status=404,
        )

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = (
        f'attachment; filename="vasq_expression_{request_id}.csv"'
    )
    # UTF-8 BOM keeps gene and region labels readable when opened in Excel.
    response.write("\ufeff")
    writer = csv.writer(response)
    columns = payload.get("columns") or []
    column_keys = [column.get("key") for column in columns]
    writer.writerow([column.get("label") for column in columns])
    for row in payload.get("rows") or []:
        writer.writerow([row.get(key, "") for key in column_keys])

    return response


def api_chat_status(request, request_id):
    """Return the current state or final result of a background chat job."""

    if request.method != "GET":
        return JsonResponse(
            {"response": "Method not allowed."},
            status=405,
        )

    request_id = str(request_id)
    job = cache.get(chat_job_key(request_id))
    if not isinstance(job, dict):
        return JsonResponse(
            {
                "status": "not_found",
                "response": (
                    "This request is no longer available. "
                    "The service may have restarted."
                ),
                "request_id": request_id,
            },
            status=404,
        )

    payload = dict(job)
    started_at = payload.get("started_at")
    if payload.get("status") in {"queued", "running", "stopping"}:
        if isinstance(started_at, (int, float)):
            payload["elapsed_seconds"] = round(
                max(0.0, time.time() - started_at),
                1,
            )

    return JsonResponse(payload)


def api_chat(request):
    try:
        if request.method != "POST":
            return JsonResponse(
                {"response": "Method not allowed."},
                status=405,
            )

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
            existing_job = cache.get(chat_job_key(request_id))
            if isinstance(existing_job, dict) and existing_job.get(
                "status"
            ) in {"queued", "running"}:
                stopping_job = dict(existing_job)
                stopping_job["status"] = "stopping"
                cache_chat_job(request_id, stopping_job)
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

        existing_job = cache.get(chat_job_key(request_id))
        if isinstance(existing_job, dict):
            response_payload = dict(existing_job)
            response_payload["status_url"] = chat_status_url(request_id)
            response_status = (
                202
                if response_payload.get("status")
                in {"queued", "running", "stopping"}
                else 200
            )
            return JsonResponse(response_payload, status=response_status)

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

        queued_at = time.time()
        cache_chat_job(
            request_id,
            {
                "status": "queued",
                "request_id": request_id,
                "chat_id": chat_id,
                "queued_at": queued_at,
            },
        )
        _CHAT_EXECUTOR.submit(
            run_chat_job,
            user_input,
            reset_history,
            chat_id,
            request_id,
        )

        return JsonResponse(
            {
                "status": "queued",
                "request_id": request_id,
                "chat_id": chat_id,
                "status_url": chat_status_url(request_id),
            },
            status=202,
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
