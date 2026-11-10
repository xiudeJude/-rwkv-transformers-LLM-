"""Generate cross-language sampling fixtures for the frontend check.

The reference values come from ``app.services.sampling`` (the same pure
pipeline used by the concept lab), not from a hand-written formula. The
frontend Node script replays every case through ``src/utils/sampling.ts`` and
asserts each probability within 1e-6.

Run from the repository root:
    .venv/Scripts/python.exe backend/scripts/generate_sampling_fixtures.py
"""

import json
import os
import sys
from typing import Any, Dict, List

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.sampling import compute_distribution

FIXTURE_PATH = os.path.join(
    REPO_ROOT, "frontend", "tests", "fixtures", "sampling_cases.json"
)

CASES: List[Dict[str, Any]] = [
    {
        "name": "raw_softmax",
        "mode": "raw",
        "logits": [2.4, 1.1, 0.3, -0.6, -1.4, 3.2, 0.0],
        "ids": [101, 102, 103, 104, 105, 106, 107],
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "temperature_0_35",
        "mode": "temperature",
        "logits": [2.8, 2.1, 1.4, 0.2, -0.5, -1.2],
        "ids": [201, 202, 203, 204, 205, 206],
        "temperature": 0.35,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "temperature_tiny_clamped",
        "mode": "temperature",
        "logits": [0.2, 0.19, 0.18, -3.0],
        "ids": [301, 302, 303, 304],
        "temperature": 1e-9,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "top_k_three",
        "mode": "topK",
        "logits": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        "ids": [401, 402, 403, 404, 405, 406],
        "temperature": 1.0,
        "top_k": 3,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "top_k_larger_than_candidates",
        "mode": "topK",
        "logits": [1.5, 1.2, 0.9, 0.1],
        "ids": [501, 502, 503, 504],
        "temperature": 1.0,
        "top_k": 999,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "top_p_0_78",
        "mode": "topP",
        "logits": [3.1, 2.7, 2.2, 1.6, 0.8, 0.1, -0.7],
        "ids": [601, 602, 603, 604, 605, 606, 607],
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.78,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "repetition_negative_logits",
        "mode": "repetition",
        "logits": [-1.5, -0.2, 0.7, -2.4, 1.1],
        "ids": [701, 702, 703, 704, 705],
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.8,
        "seen_ids": [702, 704],
    },
    {
        "name": "combined_all_stages",
        "mode": "combined",
        "logits": [4.2, 3.6, 3.1, 2.4, 1.8, 1.1, 0.4, -0.3, -1.0, -1.8],
        "ids": [801, 802, 803, 804, 805, 806, 807, 808, 809, 810],
        "temperature": 0.65,
        "top_k": 6,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "seen_ids": [803, 808],
    },
    {
        "name": "combined_single_candidate_survives",
        "mode": "combined",
        "logits": [2.0, 1.9, 1.8, 1.7, 1.6],
        "ids": [901, 902, 903, 904, 905],
        "temperature": 1.2,
        "top_k": 2,
        "top_p": 1e-6,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
    {
        "name": "combined_top_k_above_candidate_count",
        "mode": "combined",
        "logits": [0.5, 0.4, 0.3],
        "ids": [1001, 1002, 1003],
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "seen_ids": [1002],
    },
    {
        "name": "empty_candidates",
        "mode": "raw",
        "logits": [],
        "ids": [],
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seen_ids": [],
    },
]


def _backend_kwargs(case: Dict[str, Any]) -> Dict[str, Any]:
    """Map frontend mode switches to no-op defaults for the backend pipeline."""
    mode = case["mode"]
    return {
        "temperature": case["temperature"] if mode in {"temperature", "combined"} else 1.0,
        "top_k": case["top_k"] if mode in {"topK", "combined"} else 0,
        "top_p": case["top_p"] if mode in {"topP", "combined"} else 1.0,
        "repetition_penalty": (
            case["repetition_penalty"] if mode in {"repetition", "combined"} else 1.0
        ),
        "seen_ids": case["seen_ids"] if mode in {"repetition", "combined"} else [],
    }


def build_fixture() -> Dict[str, Any]:
    cases = []
    for case in CASES:
        expected = compute_distribution(
            case["logits"], case["ids"], **_backend_kwargs(case)
        )
        items = [
            {"id": token_id, "token": f"<tok_{token_id}>", "logit": logit}
            for token_id, logit in zip(case["ids"], case["logits"])
        ]
        cases.append({
            "name": case["name"],
            "mode": case["mode"],
            "items": items,
            "params": {
                "temperature": case["temperature"],
                "topK": case["top_k"],
                "topP": case["top_p"],
                "repetitionPenalty": case["repetition_penalty"],
                "seenTokenIds": case["seen_ids"],
            },
            "expected": expected,
        })
    return {
        "generated_by": "backend/scripts/generate_sampling_fixtures.py",
        "source": "backend/app/services/sampling.py",
        "tolerance": 1e-6,
        "cases": cases,
    }


def main() -> None:
    fixture = build_fixture()
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fixture, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(fixture['cases'])} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
