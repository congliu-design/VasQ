"""Simple gold-set evaluation for vasq.functions.analyze_query_intent.

Run this file from the directory that contains manage.py:

    export OPENAI_API_KEY="your-key"
    python evaluate_query_intent.py

Outputs:
    query_intent_predictions.csv
    query_intent_boolean_metrics.csv
    query_intent_entity_metrics.csv
    query_intent_summary_metrics.csv
    query_intent_category_metrics.csv
    query_intent_evaluation.png

The script directly evaluates all 100 manually annotated TEST_CASES. An
optional ``category`` field can override automatic broad category inference.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from exc

from vasq import functions as vasq_functions


OUTPUT_DIR = Path("query_intent_evaluation")
BOOLEAN_FIELDS = ["is_scientific", "asks_expression", "asks_drugs", "use_vasq"]
ENTITY_FIELDS = ["genes", "diseases"]


# Gold labels: change or extend these cases for your final evaluation.
# Optional category metadata example:
# {
#     "query": "...",
#     "category": "cross_organ_expression",
#     "expected": {...},
# }
TEST_CASES = [
    {
        "query": "Hello",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of CLDN5 in capillary endothelial cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Compare ABCB1 expression between cortex and white matter.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCB1"], "diseases": [],
        },
    },
    {
        "query": "Which brain vascular cell types express TFRC?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Which genes are most highly expressed in pericytes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is APOE associated with Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["APOE"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Which genes are associated with Parkinson's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "Which drugs target LRRK2 in Parkinson's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["LRRK2"], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "Does lecanemab treat Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "What pathways involve PSEN1 and APP?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["PSEN1", "APP"], "diseases": [],
        },
    },
    {
        "query": "How much SLC2A1 is expressed in liver hepatocytes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression in brain endothelial cells and kidney endothelial cells.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "What causes blood-brain barrier breakdown during aging?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of APOE in Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["APOE"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Are there treatments that increase ABCB1 expression at the BBB?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["ABCB1"], "diseases": [],
        },
    },
    {
        "query": "What about its expression in white matter?",
        "history": [
            {"role": "user", "content": "Tell me about CLDN5."},
            {"role": "assistant", "content": "CLDN5 is an endothelial tight-junction gene."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Are there any drugs for it?",
        "history": [
            {"role": "user", "content": "What genes are associated with Alzheimer's disease?"},
            {"role": "assistant", "content": "The discussion concerns Alzheimer's disease."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Can VasQ export a PDF?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    # ------------------------------------------------------------------
    # VasQ manuscript / queue questions supplied by the user
    # ------------------------------------------------------------------
    {
        "query": "What is the expression of glial cells in the hippocampus?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which genes are the highest-ranked markers for capillary cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "How do PLVAP, AQP1, and SLCO2A1 rank as markers for fenestrated capillary cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["PLVAP", "AQP1", "SLCO2A1"], "diseases": [],
        },
    },
    {
        "query": "What diseases, biological pathways, and drugs are directly connected to EGFR?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    {
        "query": "Using only the VasQ matrix, compare CLDN5 expression between the exact region_layer values Cortex and White Matter, this is a single-gene Cell-type comparison.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "What are the genes that are associated with Alzheimers disease? Which brain regions are they expressed in?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Do fenestrated capillary cells have specific transporter genes expressed?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is TFRC expressed across the entire brain or certain cell types and brain regions only?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Which transporters are expressed more in midbrain than in cortical regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What's the expression of CAV1 across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CAV1"], "diseases": [],
        },
    },
    {
        "query": "Which drugs target BBB or crossing of the BBB using transcytosis?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression in the memory-related region of the brain.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Is the expression of SLC7A1 the same across the frontal and parietal lobes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["SLC7A1"], "diseases": [],
        },
    },
    {
        "query": "How did endothelial transporter expression differ between gray matter and white matter?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which vascular cell types and which brain regions are enriched for small-vessel-disease susceptibility genes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["small vessel disease"],
        },
    },
    {
        "query": "What are the transporter/receptor genes with high regional specificity? What are their functions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the expression of INSR in capillary endothelial cells across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["INSR"], "diseases": [],
        },
    },
    {
        "query": "What drugs failed because they cannot cross the human BBB due to no route into receptor-mediated transcytosis?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which vascular cell types and which brain regions are enriched for ischemic stroke susceptibility genes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": ["ischemic stroke"],
        },
    },
    {
        "query": "Compare ABCA7 expression in the memory-related region of the brain.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCA7"], "diseases": [],
        },
    },
    # Additional cases: the complete gold set contains exactly 100 questions.
    # Brain-expression questions
    {
        "query": "How does CLDN5 expression vary across capillary cells in the dorsolateral prefrontal cortex?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Compare MFSD2A, ABCB1, and SLC2A1 expression across vascular cell types.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["MFSD2A", "ABCB1", "SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "What are the top expressed genes in hippocampal astrocytes?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is PLVAP expression higher in gray matter or white matter?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["PLVAP"], "diseases": [],
        },
    },
    {
        "query": "Where is APOE expressed in microglia across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["APOE"], "diseases": [],
        },
    },
    {
        "query": "Does SLC2A1 show high regional specificity in the brain vasculature?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "How does TFRC rank as a marker in arterial versus venous endothelial cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Which brain cell types express INSR?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["INSR"], "diseases": [],
        },
    },
    {
        "query": "Compare ABCA7 and PICALM expression across memory-related brain regions.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCA7", "PICALM"], "diseases": [],
        },
    },
    {
        "query": "Which endothelial genes are most enriched in the choroid plexus?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": [], "diseases": [],
        },
    },
    # Drug-only questions
    {
        "query": "Which drugs are approved for Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Which clinical LRRK2 inhibitors are being tested for Parkinson's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["LRRK2"], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "Are there compounds that modulate TFRC-mediated transcytosis?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "What treatments are available for cerebral small vessel disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["small vessel disease"],
        },
    },
    {
        "query": "Which therapies target amyloid beta?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Which EGFR inhibitors have reached clinical trials?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    {
        "query": "Are any drugs designed to cross the BBB through LRP1-mediated transport?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    # Scientific questions that do not ask for expression or drugs
    {
        "query": "How does CLDN5 contribute to blood-brain barrier integrity?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Which biological pathways involve APOE?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["APOE"], "diseases": [],
        },
    },
    {
        "query": "Is ABCA7 genetically associated with Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["ABCA7"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "What role does MFSD2A play in transcytosis at the BBB?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["MFSD2A"], "diseases": [],
        },
    },
    {
        "query": "Why does the blood-brain barrier become more permeable with aging?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Is the hippocampus involved in memory formation?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What is the functional difference between capillary and arterial endothelial cells?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What are the functions of ABCB1 and ABCG2 at the BBB?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["ABCB1", "ABCG2"], "diseases": [],
        },
    },
    {
        "query": "How are APOE, PSEN1, and APP connected to Alzheimer's disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["APOE", "PSEN1", "APP"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Explain receptor-mediated transcytosis across the BBB.",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What pathological changes characterize cerebral small vessel disease?",
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": ["small vessel disease"],
        },
    },
    # Non-scientific, casual, and application-meta questions
    {
        "query": "Thank you!",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Good morning",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "What can VasQ do?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "How do I download my queue results?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Please clear the chat.",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "How much did this API request cost?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "Can you correct the grammar in this sentence?",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    {
        "query": "This is only a test message.",
        "expected": {
            "is_scientific": False, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": [], "diseases": [],
        },
    },
    # Explicit non-brain or cross-organ expression questions
    {
        "query": "What is APOE expression in the liver?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["APOE"], "diseases": [],
        },
    },
    {
        "query": "Which kidney cell types express SLC2A1?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "How much CAV1 is expressed in lung endothelial cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CAV1"], "diseases": [],
        },
    },
    {
        "query": "Is TFRC highly expressed in cardiac muscle?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression in skin and brain endothelial cells.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "What is INSR expression in skeletal muscle?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["INSR"], "diseases": [],
        },
    },
    {
        "query": "Which colon cell types express EGFR?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    {
        "query": "Where is SLC2A1 expressed in the pancreas?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["SLC2A1"], "diseases": [],
        },
    },
    {
        "query": "Compare CLDN5 expression between brain and kidney endothelial cells.",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    # Follow-up questions with explicit recent context
    {
        "query": "Where is it expressed?",
        "history": [
            {"role": "user", "content": "Tell me about ABCB1."},
            {"role": "assistant", "content": "ABCB1 is an efflux transporter at the BBB."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["ABCB1"], "diseases": [],
        },
    },
    {
        "query": "What about capillary cells?",
        "history": [
            {"role": "user", "content": "Where is TFRC expressed in the brain?"},
            {"role": "assistant", "content": "The current expression query concerns TFRC."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Are there treatments for it?",
        "history": [
            {"role": "user", "content": "Explain cerebral small vessel disease."},
            {"role": "assistant", "content": "The discussion concerns cerebral small vessel disease."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": [], "diseases": ["small vessel disease"],
        },
    },
    {
        "query": "How does its expression vary across regions?",
        "history": [
            {"role": "user", "content": "What is the function of MFSD2A?"},
            {"role": "assistant", "content": "MFSD2A regulates BBB lipid transport and transcytosis."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["MFSD2A"], "diseases": [],
        },
    },
    {
        "query": "Which drugs target that gene?",
        "history": [
            {"role": "user", "content": "Is LRRK2 associated with Parkinson's disease?"},
            {"role": "assistant", "content": "LRRK2 is associated with Parkinson's disease."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": True, "use_vasq": False,
            "genes": ["LRRK2"], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "And in white matter?",
        "history": [
            {"role": "user", "content": "Show CAV1 expression in cortex."},
            {"role": "assistant", "content": "The expression query concerns CAV1 in cortex."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["CAV1"], "diseases": [],
        },
    },
    {
        "query": "Does it have the same pattern in astrocytes?",
        "history": [
            {"role": "user", "content": "Compare APOE expression across microglia."},
            {"role": "assistant", "content": "The expression comparison concerns APOE."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": False, "use_vasq": True,
            "genes": ["APOE"], "diseases": [],
        },
    },
    {
        "query": "What pathways is it involved in?",
        "history": [
            {"role": "user", "content": "Tell me about EGFR."},
            {"role": "assistant", "content": "EGFR is a receptor tyrosine kinase."},
        ],
        "expected": {
            "is_scientific": True, "asks_expression": False,
            "asks_drugs": False, "use_vasq": False,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    # Questions that explicitly require both drug and expression workflows
    {
        "query": "Which drugs target TFRC, and where is TFRC expressed in the brain?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["TFRC"], "diseases": [],
        },
    },
    {
        "query": "Are there treatments targeting ABCA7, and which brain regions express ABCA7?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["ABCA7"], "diseases": [],
        },
    },
    {
        "query": "Which BBB drugs use LRP1, and which vascular cell types express LRP1?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["LRP1"], "diseases": [],
        },
    },
    {
        "query": "Which EGFR inhibitors are available, and how is EGFR expressed across brain regions?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["EGFR"], "diseases": [],
        },
    },
    {
        "query": "Which LRRK2 drugs are in trials for Parkinson's disease, and where is LRRK2 expressed in the brain?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["LRRK2"], "diseases": ["Parkinson's disease"],
        },
    },
    {
        "query": "What treatments are linked to APOE in Alzheimer's disease, and which brain cells express APOE?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["APOE"], "diseases": ["Alzheimer's disease"],
        },
    },
    {
        "query": "Are there compounds that modulate CAV1, and what is its regional expression pattern?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["CAV1"], "diseases": [],
        },
    },
    {
        "query": "Which therapies affect CLDN5, and in which brain regions is CLDN5 expressed?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["CLDN5"], "diseases": [],
        },
    },
    {
        "query": "Which INSR agonists could affect BBB transport, and how is INSR expressed across capillary cells?",
        "expected": {
            "is_scientific": True, "asks_expression": True,
            "asks_drugs": True, "use_vasq": True,
            "genes": ["INSR"], "diseases": [],
        },
    },
]


def normalize_entity(value: str) -> str:
    """Normalize harmless spelling/punctuation variation before scoring."""
    value = str(value).casefold().replace("’", "'")
    value = value.replace("alzheimer disease", "alzheimer's disease")
    value = value.replace("parkinson disease", "parkinson's disease")
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    aliases = {
        "alzheimer disease": "alzheimer disease",
        "alzheimer s disease": "alzheimer disease",
        "alzheimers disease": "alzheimer disease",
        "parkinson disease": "parkinson disease",
        "parkinson s disease": "parkinson disease",
        "parkinsons disease": "parkinson disease",
        "cerebral small vessel disease": "small vessel disease",
    }
    return aliases.get(value, value)


def normalized_set(values) -> set[str]:
    return {normalize_entity(value) for value in (values or []) if str(value).strip()}


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def infer_category(case: dict) -> str:
    """Use an explicit category when supplied; otherwise infer a broad one."""
    if case.get("category"):
        return str(case["category"])
    expected = case["expected"]
    if case.get("history"):
        return "follow_up"
    if not expected["is_scientific"]:
        return "non_scientific_or_meta"
    if expected["asks_expression"] and expected["asks_drugs"]:
        return "expression_and_drug"
    if expected["asks_drugs"]:
        return "drug_only"
    if expected["asks_expression"] and not expected["use_vasq"]:
        return "non_brain_expression"
    if expected["asks_expression"]:
        return "brain_expression"
    return "scientific_non_expression"


def boolean_classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in BOOLEAN_FIELDS:
        expected = predictions[f"expected_{field}"].astype(bool)
        predicted = predictions[f"predicted_{field}"].astype(bool)
        tp = int((expected & predicted).sum())
        tn = int((~expected & ~predicted).sum())
        fp = int((~expected & predicted).sum())
        fn = int((expected & ~predicted).sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append({
            "field": field,
            "n": len(predictions),
            "positive_gold": int(expected.sum()),
            "negative_gold": int((~expected).sum()),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": safe_divide(tp + tn, len(predictions)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": (recall + specificity) / 2,
        })
    return pd.DataFrame(rows)


def entity_metrics(selected_cases: list[dict], rows: list[dict]) -> pd.DataFrame:
    output = []
    for field in ENTITY_FIELDS:
        tp = fp = fn = 0
        for case, row in zip(selected_cases, rows):
            expected_values = normalized_set(case["expected"].get(field, []))
            predicted_values = normalized_set(
                json.loads(row[f"predicted_{field}"])
            )
            tp += len(expected_values & predicted_values)
            fp += len(predicted_values - expected_values)
            fn += len(expected_values - predicted_values)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        output.append({
            "field": field,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_match": sum(row[f"correct_{field}"] for row in rows) / len(rows),
            "precision_micro": precision,
            "recall_micro": recall,
            "f1_micro": safe_divide(
                2 * precision * recall,
                precision + recall,
            ),
        })
    return pd.DataFrame(output)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    if len(TEST_CASES) != 100:
        raise SystemExit(
            f"Expected exactly 100 TEST_CASES, found {len(TEST_CASES)}."
        )

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count helper failures that analyze_query_intent catches internally and
    # routes to fallback_query_intent.
    original_helper = vasq_functions.call_helper_api
    helper_stats = {"calls": 0, "errors": 0}

    def tracked_helper(*args, **kwargs):
        helper_stats["calls"] += 1
        try:
            return original_helper(*args, **kwargs)
        except Exception:
            helper_stats["errors"] += 1
            raise

    vasq_functions.call_helper_api = tracked_helper

    rows = []
    for case_number, case in enumerate(TEST_CASES, start=1):
        calls_before = helper_stats["calls"]
        errors_before = helper_stats["errors"]
        prediction = vasq_functions.analyze_query_intent(
            case["query"],
            history=case.get("history"),
        )
        expected = case["expected"]

        row = {
            "case": case_number,
            "category": infer_category(case),
            "query": case["query"],
            "helper_called": helper_stats["calls"] > calls_before,
            "used_fallback": helper_stats["errors"] > errors_before,
            "resolved_question": prediction.get("resolved_question", ""),
        }

        boolean_matches = []
        for field in BOOLEAN_FIELDS:
            expected_value = bool(expected[field])
            predicted_value = bool(prediction.get(field, False))
            row[f"expected_{field}"] = expected_value
            row[f"predicted_{field}"] = predicted_value
            row[f"correct_{field}"] = expected_value == predicted_value
            boolean_matches.append(row[f"correct_{field}"])

        entity_matches = []
        for field in ENTITY_FIELDS:
            expected_values = normalized_set(expected.get(field, []))
            predicted_values = normalized_set(prediction.get(field, []))
            row[f"expected_{field}"] = json.dumps(expected.get(field, []), ensure_ascii=False)
            row[f"predicted_{field}"] = json.dumps(prediction.get(field, []), ensure_ascii=False)
            row[f"correct_{field}"] = expected_values == predicted_values
            entity_matches.append(row[f"correct_{field}"])

        row["routing_exact"] = all(boolean_matches)
        row["entities_exact"] = all(entity_matches)
        row["full_exact"] = row["routing_exact"] and row["entities_exact"]
        mismatched_fields = [
            field
            for field in BOOLEAN_FIELDS + ENTITY_FIELDS
            if not row[f"correct_{field}"]
        ]
        row["mismatched_fields"] = ", ".join(mismatched_fields)
        rows.append(row)

        print(
            f"[{case_number:03d}/{len(TEST_CASES)}] "
            f"{'PASS' if row['full_exact'] else 'FAIL'} - {case['query']}"
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "query_intent_predictions.csv", index=False)

    boolean_df = boolean_classification_metrics(predictions)
    boolean_df.to_csv(output_dir / "query_intent_boolean_metrics.csv", index=False)

    entity_df = entity_metrics(TEST_CASES, rows)
    entity_df.to_csv(output_dir / "query_intent_entity_metrics.csv", index=False)

    summary_df = pd.DataFrame([
        {"metric": "routing_exact", "value": predictions["routing_exact"].mean()},
        {"metric": "entities_exact", "value": predictions["entities_exact"].mean()},
        {"metric": "full_exact", "value": predictions["full_exact"].mean()},
        {"metric": "fallback_rate", "value": predictions["used_fallback"].mean()},
    ])
    summary_df.to_csv(output_dir / "query_intent_summary_metrics.csv", index=False)

    category_df = (
        predictions.groupby("category", as_index=False)
        .agg(
            n=("full_exact", "size"),
            routing_exact=("routing_exact", "mean"),
            entities_exact=("entities_exact", "mean"),
            full_exact=("full_exact", "mean"),
            fallback_rate=("used_fallback", "mean"),
        )
        .sort_values("category")
    )
    category_df.to_csv(output_dir / "query_intent_category_metrics.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].barh(boolean_df["field"], boolean_df["f1"], color="#8E3B70")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("F1")
    axes[0].set_title("Boolean routing F1")
    axes[0].grid(axis="x", alpha=0.2)
    for position, value in enumerate(boolean_df["f1"]):
        axes[0].text(min(value + 0.02, 1.01), position, f"{value:.2f}", va="center")

    exact_labels = ["Routing", "Entities", "Full query"]
    exact_values = [
        predictions["routing_exact"].mean(),
        predictions["entities_exact"].mean(),
        predictions["full_exact"].mean(),
    ]
    axes[1].bar(exact_labels, exact_values, color=["#557A95", "#73956F", "#8E3B70"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Exact-match rate")
    axes[1].set_title(f"Overall exact match (n={len(predictions)})")
    axes[1].grid(axis="y", alpha=0.2)
    for position, value in enumerate(exact_values):
        axes[1].text(position, min(value + 0.02, 1.01), f"{value:.2f}", ha="center")

    axes[2].barh(category_df["category"], category_df["full_exact"], color="#4C956C")
    axes[2].set_xlim(0, 1.05)
    axes[2].set_xlabel("Full exact-match rate")
    axes[2].set_title("Performance by category")
    axes[2].grid(axis="x", alpha=0.2)
    for position, value in enumerate(category_df["full_exact"]):
        axes[2].text(min(value + 0.02, 1.01), position, f"{value:.2f}", va="center")

    figure.tight_layout()
    figure.savefig(output_dir / "query_intent_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("\nBoolean routing metrics")
    print(boolean_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nEntity metrics")
    print(entity_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nOverall summary")
    print(summary_df.to_string(index=False, formatters={"value": "{:.3f}".format}))
    print("\nPerformance by category")
    print(category_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nFallbacks: {int(predictions['used_fallback'].sum())}/{len(predictions)}")
    print(f"Outputs: {output_dir.resolve()}")

    mismatches = predictions.loc[
        ~predictions["full_exact"],
        ["case", "category", "query", "mismatched_fields"]
        + [f"expected_{field}" for field in BOOLEAN_FIELDS + ENTITY_FIELDS]
        + [f"predicted_{field}" for field in BOOLEAN_FIELDS + ENTITY_FIELDS],
    ]
    if not mismatches.empty:
        print("\nMismatched cases:")
        print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
