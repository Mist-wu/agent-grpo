# Core baseline results

These results cover the same 500 GSM8K and 500 HotpotQA final-test questions. They do not complete the 6,000-question evaluation or any trained condition. All original generations and v1 scores remain saved; both conditions were uniformly rescored under v2 without new generation.

| Condition | Dataset | N | V2 score | 95% CI | V1 score | Final-tag compliance |
|---|---|---:|---:|---|---:|---:|
| base-direct | gsm8k | 500 | 63.8% | [60.0%, 67.8%] | 3.0% | 0.4% |
| base-direct | hotpotqa | 500 | 0.1363 | [0.1158, 0.1590] | 0.1363 | 2.2% |
| base-tools | gsm8k | 500 | 18.2% | [15.0%, 21.6%] | 0.4% | 0.2% |
| base-tools | hotpotqa | 500 | 0.1048 | [0.0935, 0.1173] | 0.1048 | 0.4% |

The current base-tools condition scored below base-direct on both core datasets. HotpotQA direct receives full context while the tool condition retrieves question-local documents, so this is a comparison of these complete prompting/tool setups. It is not evidence of a training effect. Stored traces include repeated searches over empty GSM8K local-document collections; those actual failures are retained.

Paired question bootstrap intervals for tools minus direct:
- gsm8k: -0.4560, 95% CI [-0.5040, -0.4060] (N=500).
- hotpotqa: -0.0315, 95% CI [-0.0532, -0.0108] (N=500).

Pending: cold-SFT and GRPO for seeds 42/43/44, E2 adapter comparison, all deferred GSM-Symbolic evaluations, final cost accounting and before/after training trajectories.
