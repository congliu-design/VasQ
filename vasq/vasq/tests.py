import json
import uuid
from unittest.mock import patch

import pandas as pd
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from . import views
from .functions import (
    append_missing_gene_literature_links,
    build_expression_table_payload,
    cap_text_at_line_boundary,
    explicitly_requests_vasq_matrix,
    format_matrix_expression_summary,
    search_gene_literature_evidence,
    select_top_expression_genes,
    wants_web_search,
)


class WebSearchRoutingTests(TestCase):
    def test_missing_gene_specific_links_are_restored_from_search_sections(self):
        evidence = (
            "GENE: APP\n"
            "- AD evidence [pubmed.ncbi.nlm.nih.gov/APP]"
            "(https://pubmed.ncbi.nlm.nih.gov/100/)\n"
            "GENE: APOE\n"
            "- AD evidence [pubmed.ncbi.nlm.nih.gov/APOE]"
            "(https://pubmed.ncbi.nlm.nih.gov/200/)"
        )
        answer = (
            "APP evidence "
            "[pubmed.ncbi.nlm.nih.gov/100]"
            "(https://pubmed.ncbi.nlm.nih.gov/100/)"
        )

        restored = append_missing_gene_literature_links(
            answer,
            evidence,
            ["APP", "APOE"],
        )

        self.assertIn("### Gene-specific literature links", restored)
        self.assertIn("**APOE:**", restored)
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/200/", restored)
        self.assertNotIn("- **APP:**", restored)

    @patch("vasq.functions.run_openai_web_search")
    def test_gene_literature_search_requests_coverage_for_exact_genes(self, search):
        search.return_value = "gene evidence"

        result = search_gene_literature_evidence(
            "Compare Alzheimer disease genes with VasQ expression.",
            ["APP", "APOE", "APP"],
            diseases=["Alzheimer disease"],
        )

        self.assertEqual(result, "gene evidence")
        prompt = search.call_args.args[0]
        self.assertIn("Cover every listed gene", prompt)
        self.assertIn("GENE: SYMBOL", prompt)
        self.assertIn("Exact genes: APP, APOE", prompt)
        self.assertIn("at least one inline source citation", prompt)

    def test_literature_links_override_vasq_matrix_only_route(self):
        question = (
            "Using only the VasQ matrix, compare CLDN5 expression between "
            "Cortex and White Matter, and report literature links."
        )

        self.assertTrue(explicitly_requests_vasq_matrix(question))
        self.assertTrue(wants_web_search(question))

    def test_common_external_evidence_requests_enable_web_search(self):
        questions = [
            "Search the literature for CLDN5 function at the BBB.",
            "Include citations for the scientific claims.",
            "Provide source links for the papers discussed.",
            "Include clickable links to the supporting studies.",
            "Search PubMed for supporting evidence.",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(wants_web_search(question))

    def test_genuine_vasq_only_request_still_skips_web_search(self):
        question = (
            "Using only the VasQ matrix, compare CLDN5 expression between "
            "Cortex and White Matter."
        )

        self.assertTrue(explicitly_requests_vasq_matrix(question))
        self.assertFalse(wants_web_search(question))

    def test_negated_literature_search_does_not_enable_web_search(self):
        questions = [
            "Using only the VasQ matrix; do not search the literature.",
            "Using only the VasQ matrix, without literature evidence.",
            "Use only the VasQ matrix and do not include citations.",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(explicitly_requests_vasq_matrix(question))
                self.assertFalse(wants_web_search(question))


class MatrixExpressionSummaryTests(TestCase):
    def test_full_table_payload_keeps_every_eligible_group(self):
        stats = pd.DataFrame([
            {
                "gene": "APP",
                "brain_region": "Cerebellum",
                "cell_type": "Neuron",
                "mean_expr": 3.1234567,
                "pct_expr": 0.4567,
                "n_cells": 50,
            },
            {
                "gene": "APP",
                "brain_region": "Middle Cerebral Artery",
                "cell_type": "Fib_5",
                "mean_expr": 4.2,
                "pct_expr": 0.8,
                "n_cells": 30,
            },
        ])

        payload = build_expression_table_payload(
            stats,
            gene_order=["APP"],
            group_cols=["brain_region", "cell_type"],
        )

        self.assertEqual(payload["total_rows"], 2)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(payload["rows"][0]["cell_type"], "Fib_5")
        self.assertEqual(payload["rows"][1]["pct_expr"], 45.67)
        self.assertIn("Middle Cerebral Artery", payload["filters"]["brain_region"])

    def test_only_top_three_overall_expression_genes_are_selected_for_plots(self):
        genes = ["APP", "PSEN1", "PSEN2", "APOE", "TREM2"]
        scores = {
            "APP": (1.2, 0.50),
            "PSEN1": (0.8, 0.40),
            "PSEN2": (0.3, 0.20),
            "APOE": (2.4, 0.70),
            "TREM2": (1.5, 0.30),
        }

        selected = select_top_expression_genes(genes, scores, max_genes=3)

        self.assertEqual(selected, ["APOE", "TREM2", "APP"])

    def test_summary_separates_cell_type_ranking_from_group_peak(self):
        comparison_stats = pd.DataFrame([
            {
                "gene": "APP",
                "brain_region": "Middle Cerebral Artery",
                "cell_type": "Fib_5",
                "mean_expr": 4.2,
                "pct_expr": 0.8,
                "n_cells": 30,
            },
            {
                "gene": "APP",
                "brain_region": "Cerebellum",
                "cell_type": "Neuron",
                "mean_expr": 3.1,
                "pct_expr": 0.7,
                "n_cells": 50,
            },
        ])
        cell_type_stats = pd.DataFrame([
            {
                "gene": "APP",
                "cell_type": "Neuron",
                "mean_expr": 2.9,
                "pct_expr": 0.6,
                "n_cells": 500,
            },
            {
                "gene": "APP",
                "cell_type": "Fib_5",
                "mean_expr": 2.2,
                "pct_expr": 0.5,
                "n_cells": 90,
            },
        ])

        summary = format_matrix_expression_summary(
            comparison_stats,
            "APP",
            group_cols=["brain_region", "cell_type"],
            cell_type_stats=cell_type_stats,
            max_rows=1,
            max_cell_type_rows=2,
        )

        self.assertIn("Global maximum among all 2 eligible", summary)
        self.assertIn("Middle Cerebral Artery", summary)
        self.assertIn("Cell type=Fib_5", summary)
        self.assertIn("Overall cell-type ranking", summary)
        self.assertIn("| 1 | Neuron | 2.900 |", summary)
        self.assertIn("Representative detailed comparison groups", summary)

    def test_per_gene_cap_stops_at_a_line_boundary(self):
        capped = cap_text_at_line_boundary(
            "first line\nsecond line is too long",
            24,
            suffix="[Omitted]",
        )

        self.assertLessEqual(len(capped), 24)
        self.assertEqual(capped, "first line\n[Omitted]")


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
            None,
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
    def test_expression_table_is_paged_filtered_and_downloadable(self, chat):
        request_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        expression_table = {
            "columns": [
                {"key": "gene", "label": "Gene"},
                {"key": "brain_region", "label": "Brain region"},
                {"key": "mean_expr", "label": "Mean expression"},
            ],
            "filter_keys": ["gene", "brain_region"],
            "filters": {
                "gene": ["APOE", "APP"],
                "brain_region": ["Cerebellum", "Precuneus"],
            },
            "rows": [
                {"gene": "APP", "brain_region": "Cerebellum", "mean_expr": 2.1},
                {"gene": "APOE", "brain_region": "Precuneus", "mean_expr": 3.2},
            ],
            "total_rows": 2,
            "minimum_cells_per_group": 10,
        }
        chat.return_value = (
            "expression answer",
            [{"role": "assistant", "content": "expression answer"}],
            None,
            expression_table,
        )

        views.run_chat_job(
            "Compare AD genes.",
            False,
            chat_id,
            request_id,
        )

        job = cache.get(views.chat_job_key(request_id))
        self.assertTrue(job["expression_table_url"])
        self.assertIsInstance(
            cache.get(views.expression_table_key(request_id)),
            bytes,
        )
        page = self.client.get(
            job["expression_table_url"],
            {"page_size": 1, "gene": "APOE"},
        )
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.json()["pagination"]["total_rows"], 1)
        self.assertEqual(page.json()["rows"][0]["gene"], "APOE")

        download = self.client.get(job["expression_table_download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn("text/csv", download["Content-Type"])
        self.assertIn("APP,Cerebellum,2.1", download.content.decode("utf-8-sig"))

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
