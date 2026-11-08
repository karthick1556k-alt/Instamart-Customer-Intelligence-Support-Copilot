# Instamart Customer Intelligence Platform

An end-to-end customer intelligence and support automation project built using Python, SQLite, pandas and scikit-learn.

## Project Objectives

- Customer data analysis using SQL and pandas.
- RFM-based customer segmentation using K-Means.
- Customer lapse prediction using machine learning.
- Support ticket classification and PII redaction.
- Policy retrieval using TF-IDF and cosine similarity.
- Customer support agent with simulated refund processing.

## Technology Stack

- Python 3.11+
- SQLite 3.25+
- pandas and NumPy
- scikit-learn
- joblib
- TF-IDF and cosine similarity
- Local deterministic mock LLM

## Part A — Data Analysis

- Generated customer, product, order and ticket datasets.
- Created SQLite database and performed SQL-based analysis.
- Generated seven query result CSV files.

## Part B — Customer Intelligence

- Performed RFM analysis and K-Means customer segmentation.
- Identified 1,175 customers across four segments.
- Built a machine learning model to predict customer lapse.
- Selected Random Forest based on cross-validation F1 score.

| Segment | Customers |
|---|---:|
| Growth Customers | 436 |
| Premium | 138 |
| Frequent Shoppers | 419 |
| At-Risk Customers | 182 |
| Total | 1,175 |

## Part C — Support Ticket Intelligence

- Processed 164 support tickets.
- Detected and redacted emails, phone numbers and order IDs.
- Implemented zero-shot and few-shot ticket classification.
- Routed tickets to support, security, premium and human-review queues.
- Achieved 1.00 few-shot accuracy on 147 classified tickets.

## Part D — Policy Retrieval and Support Agent

- Implemented policy document retrieval.
- Used TF-IDF and cosine similarity.
- Implemented order lookup and refund eligibility validation.
- Added simulated refund processing.
- Implemented human escalation and fallback handling.
- Added checkpoint persistence and execution limits.

See `outputs/part_d_validation_report.md` for the detailed validation report.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt