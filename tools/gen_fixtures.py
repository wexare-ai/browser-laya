"""Generate golden fixtures from the reference Python implementation of Laya.

Run:
    uv venv -p 3.12 tools/.venv
    uv pip install -p tools/.venv/bin/python laya
    tools/.venv/bin/python tools/gen_fixtures.py

Writes packages/laya-web/test/fixtures/sequences.json (tokenized sequences + marker positions,
which the TypeScript port must reproduce byte for byte) and answers.json (full model answers,
used by the opt-in end-to-end test).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "packages", "laya-web", "test", "fixtures")

# The cases deliberately cover the awkward corners of build_sequence: dict states, criteria that
# are not strings, falsy-but-real criteria, an option list that overflows head_max_len, and a
# state long enough to be truncated.
CASES = [
    {
        "name": "shoes_department",
        "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "returns": "Exchanges, refunds, wrong or damaged items",
                    "shipping": "Delivery status, delays, lost packages",
                    "billing": "Charges, invoices, payment problems",
                },
            }
        },
    },
    {
        "name": "dict_state_multi_question",
        "state": {
            "from": "user@acme.com",
            "subject": "Duplicate charge on invoice #4411",
            "body": "Hi, we were billed twice for March and I need this refunded today.",
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical"],
            },
            "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
        },
    },
    {
        "name": "choice_list_criteria",
        "state": "The build is failing on CI with a segfault in the linker.",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "What is this about?",
                "criteria": ["build", "runtime", "docs", "security"],
            }
        },
    },
    {
        "name": "falsy_criteria",
        "state": "Nothing much happened today.",
        "questions": {
            "flag": {
                "type": "choice",
                "instructions": "Pick a bucket.",
                "criteria": {"zero": 0, "false_one": False, "empty": "", "null_one": None, "real": "a description"},
            }
        },
    },
    {
        "name": "structured_criteria",
        "state": "Customer asked for a refund after 45 days.",
        "questions": {
            "policy": {
                "type": "choice",
                "instructions": "Which policy applies?",
                "criteria": {
                    "standard": {"window_days": 30, "notes": "normal returns"},
                    "extended": {"window_days": 90, "notes": "holiday returns"},
                },
            }
        },
    },
    {
        "name": "noul_with_criteria",
        "state": "Click here to verify your account or it will be closed within 24 hours.",
        "questions": {
            "is_phishing": {
                "type": "noul",
                "instructions": "Is this a phishing attempt?",
                "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
            }
        },
    },
    {
        "name": "many_options",
        "state": "I want to change the language of the interface to German.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Which intent does the user express?",
                "criteria": {
                    f"intent_{i}": f"description number {i} covering a specific user goal" for i in range(14)
                },
            }
        },
    },
    {
        "name": "long_state_truncated",
        "state": "The quick brown fox jumps over the lazy dog. " * 120,
        "questions": {
            "is_long": {"type": "noul", "instructions": "Is this text repetitive?"},
        },
    },
    {
        "name": "unicode_state",
        "state": {"message": "Mein Konto wurde zweimal belastet — bitte um Rückerstattung. 支払いが二重です。"},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "payments and refunds", "support": "everything else"},
            }
        },
    },
]


def main() -> None:
    import laya
    from laya.agent import Agent
    from laya.common import build_sequence, render_options

    os.makedirs(OUT, exist_ok=True)
    agent = Agent("convaiinnovations/laya", device="cpu")
    cfg = agent.cfg
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)

    sequences, answers = [], []
    for case in CASES:
        per_question = {}
        for qid, qdef in case["questions"].items():
            q = Agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, case["state"], q, max_len, head_max_len)
            per_question[qid] = {
                "rendered_options": render_options(q),
                "ids": ids,
                "markers": markers,
                "qtype": {"choice": 0, "score": 1, "noul": 2}[q["t"]],
            }
        sequences.append({"name": case["name"], "state": case["state"], "questions": case["questions"], "expected": per_question})
        print("sequenced", case["name"], flush=True)

        result = agent.predict(case["state"], case["questions"])
        answers.append({"name": case["name"], "state": case["state"], "questions": case["questions"], "expected": result})
        print("answered ", case["name"], flush=True)

    meta = {
        "laya_version": laya.__version__,
        "model": "convaiinnovations/laya",
        "max_len": max_len,
        "head_max_len": head_max_len,
        "temperature": cfg.get("temperature"),
        "temperature_by_options": cfg.get("temperature_by_options"),
    }
    with open(os.path.join(OUT, "sequences.json"), "w") as f:
        json.dump({"meta": meta, "cases": sequences}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "answers.json"), "w") as f:
        json.dump({"meta": meta, "cases": answers}, f, indent=1, ensure_ascii=False)
    print("wrote fixtures to", os.path.abspath(OUT))


if __name__ == "__main__":
    sys.exit(main())
