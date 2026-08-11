import json
import uuid
from unittest.mock import patch

from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from . import views


class BackgroundChatTests(TestCase):
    def tearDown(self):
        cache.clear()

    @patch("vasq.views._CHAT_EXECUTOR.submit")
    def test_chat_post_returns_job_immediately(self, submit):
        request_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())

        response = self.client.post(
            reverse("api_chat"),
            data=json.dumps(
                {
                    "message": "What is EGFR?",
                    "request_id": request_id,
                    "chat_id": chat_id,
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertIn("status_url", response.json())
        submit.assert_called_once()

        status_response = self.client.get(response.json()["status_url"])
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "queued")

    @patch("vasq.views.chat")
    def test_background_job_publishes_completed_result(self, chat):
        request_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        chat.return_value = (
            "EGFR answer",
            [{"role": "assistant", "content": "EGFR answer"}],
            {"data": []},
        )

        views.run_chat_job(
            "What is EGFR?",
            False,
            chat_id,
            request_id,
        )

        job = cache.get(views.chat_job_key(request_id))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["response"], "EGFR answer")
        self.assertEqual(job["graph_json"], {"data": []})

    @patch("vasq.views.chat")
    def test_cancelled_job_does_not_start_chat(self, chat):
        request_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        cache.set(
            views.cancellation_key(request_id),
            True,
            timeout=60,
        )

        views.run_chat_job(
            "What is EGFR?",
            False,
            chat_id,
            request_id,
        )

        chat.assert_not_called()
        job = cache.get(views.chat_job_key(request_id))
        self.assertEqual(job["status"], "stopped")
        self.assertTrue(job["stopped"])
