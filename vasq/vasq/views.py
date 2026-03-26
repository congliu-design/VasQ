import json
from django.http import JsonResponse
from django.shortcuts import render
from .functions import chat


import logging
logger = logging.getLogger(__name__)

def chat_view(request):
    logger.info("chat_view called: method=%s", request.method)
    try:
        # existing code
        ...
    except Exception:
        logger.exception("chat_view failed")
        raise


# Render landing page
def index(request):
    return render(request, 'vasq/index.html')

# Parse user input
def parse_input(request):
    return json.loads(request.body).get('message', '')

# API endpoint for chat
def api_chat(request):
    try:
        user_input = parse_input(request)
        logger.info("Received message: %s", user_input)

        history = request.session.get('history', [])
        
        content, history, graph_json = chat(user_input, history)

        request.session['history'] = history

        return JsonResponse({
            "response": content,
            "graph_json": graph_json
        })


    except Exception:
        logger.exception("api_chat failed")
        return JsonResponse(
            {"response": "Sorry, something went wrong while processing your message."},
            status=500
        )

