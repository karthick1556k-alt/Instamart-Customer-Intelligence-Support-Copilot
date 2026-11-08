
"""Part C: Offline policy assistant with retrieval and evaluation."""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm  # noqa: E402


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = """
POLICY ASSISTANT

Answer questions using only the supplied policy context.
Do not invent policy rules or use outside knowledge.
If the context does not answer the question, say:
I could not find that information in the policy documents.
Keep answers concise and grounded in the policy text.
"""

STOP_WORDS = {
    "the", "a", "an", "is", "of", "to", "for", "in", "on", "how",
    "what", "when", "who", "are", "do", "does", "i", "my", "be",
    "can", "after", "from", "it", "or", "and", "at", "by", "which",
    "why", "if", "than", "that", "must", "have", "you", "me",
    "i", "will", "does", "did", "was", "were", "with", "within",
}


def terms(text):
    """Return normalized terms for lightweight lexical retrieval."""
    words = re.findall(r"[a-z]+", str(text).lower())
    return {word[:5] for word in words if word not in STOP_WORDS}


def load_policies():
    """Load policy documents without changing their contents."""
    policies = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            policies[path.name] = content

    if not policies:
        raise FileNotFoundError(f"No policy text files found in {POLICY_DIR}")

    return policies


def retrieve_policy(question, policies):
    """Rank policy documents by lexical relevance to a question."""
    query_terms = terms(question)

    if not query_terms:
        return None, 0.0

    scores = {}

    for filename, content in policies.items():
        document_terms = terms(content)
        overlap = query_terms & document_terms

        # Proportion of query terms supported by this document.
        coverage = len(overlap) / len(query_terms)

        # A small precision component discourages unrelated long documents.
        precision = len(overlap) / max(len(document_terms), 1)
        scores[filename] = coverage + 0.25 * precision

    best_doc = max(scores, key=scores.get)
    best_score = scores[best_doc]

    # Do not answer when no meaningful query terms match.
    if best_score <= 0:
        return None, 0.0

    return best_doc, best_score


def answer_question(question, policies):
    """Retrieve a document and ask the offline mock LLM for an answer."""
    source_doc, retrieval_score = retrieve_policy(question, policies)

    if source_doc is None:
        return {
            "answer": "I could not find that information in the policy documents.",
            "source_doc": "NONE",
            "retrieval_score": 0.0,
            "status": "abstained",
        }

    # Give the mock assistant only the selected source document.
    context = f"[{source_doc}]\n{policies[source_doc]}"
    user_content = f"CONTEXT:\n{context}\nQUESTION:\n{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = mock_llm.chat(messages)
    answer = (response.choices[0].message.content or "").strip()

    fallback = "I could not find that information in the policy documents."

    # If the retrieved document has no terms in common, abstain.
    if not answer:
        answer = fallback
        source_doc = "NONE"
        status = "abstained"
    else:
        status = "answered"

    return {
        "answer": answer,
        "source_doc": source_doc,
        "retrieval_score": round(retrieval_score, 4),
        "status": status,
    }


def fact_supported(answer, expected_fact):
    """Check whether the answer contains the expected fact's key terms."""
    if pd.isna(expected_fact) or not str(expected_fact).strip():
        return None

    expected = terms(expected_fact)
    actual = terms(answer)

    if not expected:
        return None

    # At least one expected normalized term must occur in the answer.
    # This is a simple proxy, not a semantic entailment metric.
    return bool(expected & actual)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE}")

    policies = load_policies()
    evaluation = pd.read_csv(EVAL_FILE)

    required = {"question", "expected_doc", "expected_fact"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")

    mock_llm.reset()

    results = []

    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = (
            "NONE"
            if pd.isna(row["expected_doc"])
            else str(row["expected_doc"]).strip()
        )

        result = answer_question(question, policies)

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": row["expected_fact"],
            **result,
        })

    results_df = pd.DataFrame(results)

    results_df["source_correct"] = (
        results_df["source_doc"] == results_df["expected_doc"]
    )
    results_df["fact_supported"] = results_df.apply(
        lambda row: fact_supported(row["answer"], row["expected_fact"]),
        axis=1,
    )

    results_path = OUTPUT_DIR / "part_c_policy_answers.csv"
    results_df.to_csv(results_path, index=False)

    answerable = results_df["expected_doc"].ne("NONE")
    out_of_scope = ~answerable

    source_accuracy = (
        results_df.loc[answerable, "source_correct"].mean()
        if answerable.any()
        else 0.0
    )
    out_of_scope_abstention_rate = (
        results_df.loc[out_of_scope, "status"].eq("abstained").mean()
        if out_of_scope.any()
        else 0.0
    )
    fact_support_rate = (
        results_df.loc[answerable, "fact_supported"].eq(True).mean()
        #results_df.loc[answerable, "fact_supported"].fillna(False).mean()"""
        if answerable.any()
        else 0.0
    )

    summary = {
        "evaluation_questions": int(len(results_df)),
        "answerable_questions": int(answerable.sum()),
        "out_of_scope_questions": int(out_of_scope.sum()),
        "source_document_accuracy": round(float(source_accuracy), 4),
        "expected_fact_support_rate": round(float(fact_support_rate), 4),
        "out_of_scope_abstention_rate": round(
            float(out_of_scope_abstention_rate), 4
        ),
        "documents_loaded": sorted(policies.keys()),
        "note": (
            "Fact support uses normalized keyword overlap, not semantic "
            "entailment. Review answers manually before deployment."
        ),
    }

    with open(
        OUTPUT_DIR / "part_c_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 65)
    print("PART C: POLICY ASSISTANT EVALUATION")
    print("=" * 65)
    print(f"Policy documents loaded:       {len(policies)}")
    print(f"Evaluation questions:          {len(results_df)}")
    print(f"Answerable questions:          {int(answerable.sum())}")
    print(f"Out-of-scope questions:        {int(out_of_scope.sum())}")
    print(f"Source document accuracy:      {source_accuracy:.2%}")
    print(f"Expected-fact support rate:    {fact_support_rate:.2%}")
    print(f"Out-of-scope abstention rate:  {out_of_scope_abstention_rate:.2%}")

    print("\nQuestion results:")
    print(
        results_df[
            [
                "question",
                "expected_doc",
                "source_doc",
                "source_correct",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    print("  outputs/part_c_policy_answers.csv")
    print("  outputs/part_c_metrics.json")


if __name__ == "__main__":
    main()

