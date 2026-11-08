import json
import re
import time
import random
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

TICKETS_FILE = RAW_DIR / "tickets.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

ALLOWED = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3
BUDGET = 100000

ZERO_SHOT_PROMPT = """
CLASSIFY
You are a support ticket classification assistant.
Classify the customer's ticket into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this exact format:
{"category":"one_allowed_category","reason":"short reason"}

Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
CLASSIFY
You are a support ticket classification assistant.
Classify the customer's ticket into exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: Someone made an unauthorized transaction using my account.
Output: {"category":"fraud","reason":"unauthorized transaction"}

Ticket: My account was hacked and money was stolen.
Output: {"category":"fraud","reason":"account hacked"}

Ticket: The app crashes whenever I try to open it.
Output: {"category":"technical","reason":"app crash"}

Ticket: I was charged twice for the same order.
Output: {"category":"billing","reason":"duplicate charge"}

Ticket: The rider has not delivered my order.
Output: {"category":"delivery","reason":"delivery issue"}

Return JSON only in this exact format:
{"category":"one_allowed_category","reason":"short reason"}

Do not include markdown or additional text.
"""


def redact_text(text):
    """Redact PII in the required order and return replacement counts."""
    text = str(text or "")
    counts = {}

    text, counts["email"] = re.subn(
        r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text
    )
    text, counts["phone"] = re.subn(
        r"\b[6-9]\d{9}\b", "[PHONE]", text
    )
    text, counts["order_id"] = re.subn(
        r"ORD-\d{6}", "[ORDER_ID]", text
    )

    return text, counts


def classify_output(raw_output):
    """Validate JSON first, then allow one unambiguous category keyword."""
    if not raw_output:
        return None, "flagged_invalid_output"

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            category = str(parsed.get("category", "")).lower().strip()
            if category in ALLOWED:
                return category, "classified_json"
            return None, "flagged_invalid_output"
    except (json.JSONDecodeError, TypeError):
        pass

    words = set(re.findall(r"[a-z]+", raw_output.lower()))
    matches = words.intersection(ALLOWED)

    if len(matches) == 1:
        return next(iter(matches)), "classified_keyword"

    return None, "flagged_invalid_output"


def call_with_retry(ticket_id, redacted_text, system_prompt):
    """Retry transient errors up to three times after the initial attempt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

    retries = 0
    tokens_used = 0

    while True:
        try:
            response = mock_llm.chat(messages)
            tokens_used += response.usage.total_tokens
            raw_output = response.choices[0].message.content
            return raw_output, tokens_used, retries

        except mock_llm.TransientLLMError:
            if retries >= MAX_RETRIES:
                print(f"ERROR: {ticket_id} failed after {MAX_RETRIES} retries")
                return None, tokens_used, retries

            retries += 1
            wait_time = (
                BASE_DELAY
                * (2 ** (retries - 1))
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retries} wait={wait_time:.3f}s"
            )
            time.sleep(wait_time)


def load_data():
    tickets = pd.read_csv(TICKETS_FILE)
    segments = pd.read_csv(SEGMENTS_FILE)

    required_ticket_columns = {
        "ticket_id", "customer_id", "text", "true_category"
    }
    required_segment_columns = {
        "customer_id", "segment_id", "segment_name", "is_premium"
    }

    if not required_ticket_columns.issubset(tickets.columns):
        raise ValueError(
            f"Missing ticket columns: "
            f"{required_ticket_columns - set(tickets.columns)}"
        )

    if not required_segment_columns.issubset(segments.columns):
        raise ValueError(
            f"Missing segment columns: "
            f"{required_segment_columns - set(segments.columns)}"
        )

    if segments["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs in customer_segments.csv")

    segment_lookup = segments.set_index("customer_id").to_dict("index")
    return tickets, segment_lookup


def run_batch(tickets, system_prompt, run_name, token_budget=None):
    """Run one independent batch. Reset mock counters immediately before it."""
    mock_llm.reset()

    rows = []
    total_tokens = 0
    retry_total = 0
    budget_exhausted = 0

    for _, ticket in tickets.iterrows():
        ticket_id = str(ticket["ticket_id"])
        raw_text = ticket["text"]

        if pd.isna(raw_text) or not str(raw_text).strip():
            rows.append({
                "ticket_id": ticket_id,
                "true_category": ticket["true_category"],
                "category": None,
                "status": "skipped_empty",
                "tokens_used": 0,
                "raw_output": None,
            })
            continue

        raw_text = str(raw_text)

        if mock_llm.est_tokens(raw_text) > 400:
            rows.append({
                "ticket_id": ticket_id,
                "true_category": ticket["true_category"],
                "category": None,
                "status": "flagged_long",
                "tokens_used": 0,
                "raw_output": None,
            })
            continue

        if token_budget is not None and total_tokens >= token_budget:
            budget_exhausted += 1
            rows.append({
                "ticket_id": ticket_id,
                "true_category": ticket["true_category"],
                "category": None,
                "status": "budget_exhausted",
                "tokens_used": 0,
                "raw_output": None,
            })
            continue

        redacted_text, _ = redact_text(raw_text)

        raw_output, tokens_used, retries = call_with_retry(
            ticket_id, redacted_text, system_prompt
        )

        total_tokens += tokens_used
        retry_total += retries

        if raw_output is None:
            category = None
            status = "flagged_invalid_output"
        else:
            category, status = classify_output(raw_output)

        rows.append({
            "ticket_id": ticket_id,
            "true_category": ticket["true_category"],
            "category": category,
            "status": status,
            "tokens_used": tokens_used,
            "raw_output": raw_output,
        })

    result = pd.DataFrame(rows)

    classified = result[
        result["status"].isin(["classified_json", "classified_keyword"])
        & result["true_category"].notna()
    ]

    if len(classified):
        accuracy = (
            classified["category"].astype(str).str.lower()
            == classified["true_category"].astype(str).str.lower()
        ).mean()
    else:
        accuracy = 0.0

    summary = {
        "run_name": run_name,
        "total_tickets": len(result),
        "classified_json": int((result["status"] == "classified_json").sum()),
        "classified_keyword": int((result["status"] == "classified_keyword").sum()),
        "flagged_invalid_output": int(
            (result["status"] == "flagged_invalid_output").sum()
        ),
        "skipped_empty": int((result["status"] == "skipped_empty").sum()),
        "flagged_long": int((result["status"] == "flagged_long").sum()),
        "budget_exhausted": int(
            (result["status"] == "budget_exhausted").sum()
        ),
        "classified_count": len(classified),
        "accuracy": round(float(accuracy), 4),
        "total_tokens": int(total_tokens),
        "tokens_per_classified_ticket": round(
            total_tokens / len(classified), 2
        ) if len(classified) else 0,
        "retry_count": retry_total,
        "budget": token_budget,
    }

    print(f"\n========== {run_name.upper()} SUMMARY ==========")
    print(json.dumps(summary, indent=2))

    return result, summary


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    tickets, segment_lookup = load_data()

    print("Ticket rows:", len(tickets))
    print("Customer segment rows:", len(segment_lookup))

    # Redaction audit across the complete ticket dataset.
    redaction_totals = {"email": 0, "phone": 0, "order_id": 0}
    remaining_pii = {"email": 0, "phone": 0, "order_id": 0}

    for value in tickets["text"]:
        if pd.isna(value):
            continue

        redacted, counts = redact_text(value)

        for key in redaction_totals:
            redaction_totals[key] += counts[key]

        remaining_pii["email"] += len(
            re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted)
        )
        remaining_pii["phone"] += len(
            re.findall(r"\b[6-9]\d{9}\b", redacted)
        )
        remaining_pii["order_id"] += len(
            re.findall(r"ORD-\d{6}", redacted)
        )

    print("\n========== REDACTION AUDIT ==========")
    print("Replacement counts:", redaction_totals)
    print("Remaining PII matches:", remaining_pii)
    print("Total replacements:", sum(redaction_totals.values()))

    if any(remaining_pii.values()):
        raise ValueError("PII remains after redaction. Stopping safely.")

    # 1. Zero-shot run.
    zero_result, zero_summary = run_batch(
        tickets, ZERO_SHOT_PROMPT, "zero_shot"
    )

    # 2. Few-shot run.
    few_result, few_summary = run_batch(
        tickets, FEW_SHOT_PROMPT, "few_shot"
    )

    # Save comparison and zero-shot confusion matrix.
    comparison = pd.DataFrame([zero_summary, few_summary])
    comparison.to_csv(
        OUTPUT_DIR / "triage_comparison.csv", index=False
    )

    zero_classified = zero_result[
        zero_result["status"].isin(
            ["classified_json", "classified_keyword"]
        )
        & zero_result["true_category"].notna()
    ]

    confusion = pd.crosstab(
        zero_classified["true_category"],
        zero_classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    confusion.to_csv(OUTPUT_DIR / "triage_zero_shot_confusion.csv")

    # 3. Token budget demonstration: try 6000, then 5000 if necessary.
    budget_result, budget_summary = run_batch(
        tickets, ZERO_SHOT_PROMPT, "budget_demo_6000",
        token_budget=6000
    )

    if budget_summary["budget_exhausted"] == 0:
        print("\n6000-token budget was not exhausted; retrying demo at 5000.")
        budget_result, budget_summary = run_batch(
            tickets, ZERO_SHOT_PROMPT, "budget_demo_5000",
            token_budget=5000
        )

    budget_result.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv", index=False
    )

    # 4. Final few-shot run with a high token budget.
    final_result, final_summary = run_batch(
        tickets, FEW_SHOT_PROMPT, "final_few_shot",
        token_budget=BUDGET
    )

    final_rows = []

    for _, ticket in tickets.iterrows():
        ticket_id = str(ticket["ticket_id"])
        customer_id = ticket["customer_id"]

        result_row = final_result[
            final_result["ticket_id"] == ticket_id
        ].iloc[0]

        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name")
        is_premium = str(segment.get("is_premium", "")).strip().lower()

        status = result_row["status"]
        category = result_row["category"]

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route = "human_review"
            priority = "NORMAL"
        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_premium in {"true", "1", "1.0", "yes"}:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        final_rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": int(result_row["tokens_used"]),
        })

    final_columns = [
        "ticket_id", "customer_id", "status", "category",
        "segment_name", "priority", "route", "tokens_used"
    ]

    final_df = pd.DataFrame(final_rows, columns=final_columns)
    final_df.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

    metrics = {
        "redaction_counts": redaction_totals,
        "remaining_pii_matches": remaining_pii,
        "zero_shot": zero_summary,
        "few_shot": few_summary,
        "budget_demo": budget_summary,
        "final_few_shot": final_summary,
        "final_route_counts": final_df["route"].value_counts().to_dict(),
        "final_status_counts": final_df["status"].value_counts().to_dict(),
    }

    with open(OUTPUT_DIR / "triage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n========== FINAL ROUTING ==========")
    print("Rows:", len(final_df))
    print("Columns:", final_df.columns.tolist())
    print("\nRoute counts:")
    print(final_df["route"].value_counts())
    print("\nStatus counts:")
    print(final_df["status"].value_counts())
    print("\nCreated:")
    print("outputs/triage_results.csv")
    print("outputs/triage_budget_demo.csv")
    print("outputs/triage_comparison.csv")
    print("outputs/triage_zero_shot_confusion.csv")
    print("outputs/triage_metrics.json")


if __name__ == "__main__":
    main()
