\# Part D — Validation Report



\## 1. Overview



Part D implements a policy retrieval and customer support agent for the Instamart Customer Intelligence Platform.



The workflow uses policy retrieval, order lookup, refund eligibility validation, simulated refund processing, human escalation and checkpoint persistence.



\## 2. Scenario Execution Results



A total of 10 durable execution records were found.



| Status | Count |

|---|---:|

| COMPLETED | 6 |

| ERROR | 2 |

| STOPPED\_SAFELY | 1 |

| FALLBACK\_MAX\_ITERATIONS | 1 |

| Total | 10 |



S1 and S2 initially encountered errors and subsequently completed successfully.



S3, S4, S5 and S6 completed successfully.



S7 reached the maximum iteration limit and returned a fallback response.



\## 3. Policy Retrieval Evaluation



The policy retrieval evaluation used 14 questions.



| Evaluation | Result |

|---|---:|

| Total questions | 14 |

| Answerable questions | 12 |

| Correct expected document in top 3 | 12/12 |

| Expected fact found | 12/12 |

| Unanswerable questions correctly handled | 2/2 |



The evaluation also showed that zero-overlap retrieval performed worse than the TF-IDF retrieval approach.



\## 4. Execution Limit Validation



Two isolated tests were performed using temporary state and output files.



\### Timeout Test



\- Result: FALLBACK\_TIMEOUT

\- Successful LLM calls: 0

\- Tokens used: 0

\- Tools executed: 0

\- Human handoff: Required



\### Token Limit Test



\- Result: FALLBACK\_TOKEN\_LIMIT

\- Successful LLM calls: 2

\- Reported tokens: 400

\- Tools executed: 1



The token limit test stopped further agent execution after the cumulative token limit was exceeded.



\## 5. Refund Ledger Validation



The refund ledger contains two simulated approved refunds.



| Order ID | Refund Amount | Status |

|---|---:|---|

| ORD-100048 | 128.25 | SIMULATED\_APPROVED |

| ORD-100067 | 135.00 | SIMULATED\_APPROVED |



Total simulated refund amount: 263.25.



These are simulation records only. No actual payment or refund was issued.



\## 6. Checkpoint and Execution Log Validation



\- Checkpoint state contains scenarios S1 through S7.

\- The in-progress scenario list is empty.

\- The execution log contains 10 durable records.

\- The latest S7 execution ended with FALLBACK\_MAX\_ITERATIONS.

\- The refund ledger contains two entries.



\## 7. Known Limitations



\- Historical execution records do not contain per-run token and LLM-call metrics.

\- The existing part\_d\_metrics.json file is historical and does not reflect the latest 10 execution records.

\- S7 did not complete successfully and returned a safe fallback response.

\- The LLM implementation uses a deterministic offline mock.

\- Refund processing is simulated and does not issue real payments.



\## 8. Conclusion



Part D demonstrates policy retrieval, tool-based customer support, simulated refund validation, human escalation, checkpoint persistence and execution-limit handling.



The validation confirms successful execution of six scenarios, safe fallback handling for the remaining scenarios, correct policy retrieval evaluation and preservation of the two simulated refund ledger entries.

