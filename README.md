# Instamart Customer Intelligence Platform

## Overview

An end-to-end customer intelligence and support automation platform built using Python, SQLite, pandas and scikit-learn.

The project combines customer analytics, machine learning, support ticket classification, policy retrieval and an AI-assisted customer support workflow.

### Technology Stack

- Python 3.11+
- SQLite 3.25+
- pandas and NumPy
- scikit-learn
- joblib
- TF-IDF and cosine similarity
- Deterministic offline mock LLM

### Project Dataset

Generated dataset:
- Customers: 1,200
- Products: 40
- Raw orders: 13,084
- Support tickets: 164

## Reproduce

### Environment Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Execution Order

For a fresh project environment:

1. Generate the datasets using `generate_data.py`.
2. Execute Part A data preparation and SQL analysis.
3. Execute Part B customer analytics and lapse prediction.
4. Execute Part C ticket classification and triage.
5. Execute Part D policy retrieval and support agent workflows.

The supplied `generate_data.py` and `mock_llm.py` files must remain unmodified.

The repository includes generated data, a populated SQLite database, model artifacts and execution outputs. Avoid regenerating data or replaying stateful Part D scenarios when validating the existing results.

## Part A

### Data Analysis and Database Preparation

- Generated customer, product, order and ticket datasets.
- Created the SQLite database at `data/instamart.db`.
- Prepared clean customer, product and order datasets.
- Implemented SQL queries in `sql/queries.sql`.
- Generated seven SQL analysis output files (`part_a_q1.csv` through `part_a_q7.csv`).

## Part B

### Customer Intelligence and Lapse Prediction

Performed RFM analysis and K-Means clustering to identify customer segments.

#### RFM Results

- Customers included in RFM analysis: 1,175
- Median recency: 19 days
- Minimum recency: 0 days
- Maximum recency: 180 days
- Mean frequency: 9.46
- Maximum frequency: 33
- Mean monetary value: 2,552.06
- Maximum monetary value: 14,766.55

#### Customer Segmentation

- Number of clusters: 4
- K-Means inertia: 743.1
- Silhouette score: 0.526
- PCA explained variance: 63.4% and 21.8%
- Combined PCA explained variance: 85.2%

| Customer Segment | Customers |
|---|---:|
| Growth Customers | 436 |
| Premium | 138 |
| Frequent Shoppers | 419 |
| At-Risk Customers | 182 |
| Total | 1,175 |

#### Customer Lapse Prediction

- Lapse cohort: 1,156
- Lapsed customers: 443
- Lapse rate: 38.32%
- Training records: 924
- Testing records: 232
- Baseline accuracy: 0.6164
- Logistic Regression test F1: 0.648
- Random Forest cross-validation F1: 0.6866
- Selected Random Forest test F1: 0.6369

Random Forest was selected based on cross-validation F1 performance.

## Part C

### Support Ticket Intelligence and Triage

Processed 164 support tickets using PII redaction, zero-shot classification, few-shot classification and routing.

#### PII Redaction

| PII Type | Redacted |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |

No detected PII remained after redaction.

#### Classification Results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets processed | 164 | 164 |
| Classified tickets | 147 | 147 |
| Accuracy | 0.8571 | 1.0000 |
| Tokens used | 11,740 | 24,220 |
| Tokens per classified ticket | 79.86 | 164.76 |
| Retries | 19 | 19 |

The accuracy figures apply to the 147 classified tickets, not all 164 tickets.

#### Budget Validation

- Token budget: 6,000
- Classified tickets: 76
- Tokens used: 6,031
- Budget-exhausted tickets: 76
- Accuracy on classified tickets: 0.8421
- Retries: 9

#### Final Routing

| Queue | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| Total | 164 |

Priority distribution:
- HIGH: 45
- NORMAL: 119

## Part D

### Policy Retrieval and Customer Support Agent

Implemented a support agent using policy retrieval, order lookup, refund eligibility checks, simulated refund processing, human escalation and checkpoint persistence.

#### Policy Retrieval Evaluation

- Evaluation questions: 14
- Answerable questions: 12
- Expected document retrieved in top 3: 12/12
- Expected fact found: 12/12
- Unanswerable questions correctly handled: 2/2

#### Scenario Execution

Total durable execution records: 10.

| Execution Status | Count |
|---|---:|
| COMPLETED | 6 |
| ERROR | 2 |
| STOPPED_SAFELY | 1 |
| FALLBACK_MAX_ITERATIONS | 1 |
| Total | 10 |

The two initial errors for S1 and S2 were followed by successful executions. S7 reached the maximum iteration limit and returned a fallback response.

#### Execution Limit Tests

Timeout test:
- Result: FALLBACK_TIMEOUT
- Successful LLM calls: 0
- Tokens used: 0
- Tools executed: 0
- Human handoff: Required

Token limit test:
- Result: FALLBACK_TOKEN_LIMIT
- Successful LLM calls: 2
- Reported tokens: 400
- Tools executed: 1

These were isolated tests using temporary output and state files.

#### Simulated Refund Ledger

| Order ID | Refund Amount | Status |
|---|---:|---|
| ORD-100048 | 128.25 | SIMULATED_APPROVED |
| ORD-100067 | 135.00 | SIMULATED_APPROVED |
| Total | 263.25 | Simulation only |

No actual payments or refunds were issued.

#### Checkpoint Validation

- Scenarios represented: 7 (S1–S7)
- Durable execution records: 10
- Refund ledger entries: 2
- In-progress scenarios at validation: 0

Detailed results are available in `outputs/part_d_validation_report.md`.

## Limitations and Next Steps

### Limitations

- The LLM is a deterministic offline mock, not a production hosted model.
- Refund processing is simulated and does not issue actual payments.
- Historical durable Part D records do not contain per-run token and model-call metrics.
- The existing `part_d_metrics.json` is historical and does not reflect the latest 10 execution records.
- S7 ended with a safe maximum-iteration fallback rather than successful completion.
- The dataset is generated and does not represent actual production customer behaviour.
- Part C accuracy is calculated on classified tickets; unclassified tickets are handled separately.

### Next Steps

- Integrate a production LLM with secure API configuration.
- Improve retrieval using semantic embeddings and hybrid search.
- Add production database and API integration.
- Extend customer lapse prediction with additional behavioural features.
- Add automated monitoring and model performance tracking.
- Integrate real support ticket and order management systems with appropriate access controls.
- Expand testing for edge cases, agent recovery and execution limits.