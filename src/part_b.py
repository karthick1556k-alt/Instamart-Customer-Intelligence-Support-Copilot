
"""Part B: Offline support-ticket classification using mock_llm.py."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm  # noqa: E402


INPUT_FILE = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
FALLBACK_CATEGORY = "technical"

SYSTEM_PROMPT = """
CLASSIFY

You are a customer-support ticket classifier.
Classify the ticket into exactly one category:
billing, delivery, technical, or fraud.

Return only valid JSON with these keys:
{"category": "one allowed category", "reason": "short explanation"}

Examples:
- Unauthorized transaction or hacked account: fraud
- Invoice or incorrect charge: billing
- App crash, login, OTP, or error: technical
- Late delivery or rider issue: delivery
"""


def call_classifier(ticket_text, max_attempts=4):
    """Call the mock LLM with retries and validate its JSON response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]

    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response")

            result = json.loads(content)

            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object")

            category = str(result.get("category", "")).strip().lower()
            reason = str(result.get("reason", "")).strip()

            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"Unsupported category: {category!r}")

            return {
                "predicted_category": category,
                "reason": reason,
                "status": "success",
                "attempts": attempt,
                "error": "",
            }

        except mock_llm.TransientLLMError as exc:
            last_error = str(exc)
            if attempt < max_attempts:
                time.sleep(0.05 * attempt)

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            last_error = str(exc)
            # A malformed response may be transient, so retry it too.
            if attempt < max_attempts:
                time.sleep(0.05 * attempt)

    return {
        "predicted_category": FALLBACK_CATEGORY,
        "reason": "Fallback used after classification failures",
        "status": "fallback",
        "attempts": max_attempts,
        "error": last_error,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Ticket file not found: {INPUT_FILE}\n"
            "Run generate_data.py from the project root first."
        )

    tickets = pd.read_csv(INPUT_FILE)
    required = {"ticket_id", "customer_id", "order_id", "text", "true_category"}

    missing_columns = required - set(tickets.columns)
    if missing_columns:
        raise ValueError(f"Missing ticket columns: {sorted(missing_columns)}")

    # Reset deterministic mock behaviour before each independent run.
    mock_llm.reset()

    results = []
    for _, row in tickets.iterrows():
        ticket_text = row["text"]

        if pd.isna(ticket_text) or not str(ticket_text).strip():
            result = {
                "predicted_category": "none",
                "reason": "Ticket text is missing; classification skipped",
                "status": "skipped_missing_text",
                "attempts": 0,
                "error": "",
            }
        else:
            result = call_classifier(str(ticket_text))

        results.append(result)

    result_df = pd.DataFrame(results)
    predictions = pd.concat(
        [tickets.reset_index(drop=True), result_df.reset_index(drop=True)],
        axis=1,
    )

    # Save every ticket, including skipped and fallback cases.
    predictions_path = OUTPUT_DIR / "part_b_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # Evaluate only tickets with usable text and a valid target category.
    valid = (
        predictions["text"].notna()
        & predictions["text"].astype("string").str.strip().ne("").fillna(False)
        & predictions["true_category"].isin(ALLOWED_CATEGORIES)
    )
    evaluation = predictions.loc[valid].copy()

    labels = sorted(ALLOWED_CATEGORIES)
    y_true = evaluation["true_category"]
    y_pred = evaluation["predicted_category"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / "part_b_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    cm_df.to_csv(OUTPUT_DIR / "part_b_confusion_matrix.csv")

    summary = {
        "input_tickets": int(len(tickets)),
        "evaluated_tickets": int(len(evaluation)),
        "missing_text_tickets": int(
            predictions["status"].eq("skipped_missing_text").sum()
        ),
        "fallback_tickets": int(predictions["status"].eq("fallback").sum()),
        "successful_classifications": int(
            predictions["status"].eq("success").sum()
        ),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "categories": labels,
    }

    with open(OUTPUT_DIR / "part_b_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 60)
    print("PART B: SUPPORT-TICKET CLASSIFICATION")
    print("=" * 60)
    print(f"Input tickets:                 {len(tickets)}")
    print(f"Tickets evaluated:             {len(evaluation)}")
    print(f"Missing-text tickets skipped:  {summary['missing_text_tickets']}")
    print(f"Successful classifications:    {summary['successful_classifications']}")
    print(f"Fallback classifications:      {summary['fallback_tickets']}")
    print(f"Accuracy:                      {accuracy:.4f}")
    print(f"Macro precision:               {precision:.4f}")
    print(f"Macro recall:                  {recall:.4f}")
    print(f"Macro F1:                      {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion matrix (actual rows, predicted columns):")
    print(cm_df.to_string())

    print("\nSaved outputs:")
    for name in [
        "part_b_predictions.csv",
        "part_b_classification_report.csv",
        "part_b_confusion_matrix.csv",
        "part_b_metrics.json",
    ]:
        print(f"  outputs/{name}")


if __name__ == "__main__":
    main()
