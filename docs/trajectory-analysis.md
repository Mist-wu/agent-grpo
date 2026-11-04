# Trajectory analysis

PENDING: after formal runs, inspect ten before/after pairs from held-out data. Separate wrong arithmetic, retrieval failures, malformed/extra calls, context truncation and reward exploitation. Report concrete observed reward-hacking examples only; if none are observed, say none observed and state the inspected scope. Do not invent an example to satisfy the initial brief.

## Observed before training

Automated inspection of the first 128 actual core base-tools rows found 35 trajectories reaching the four-tool-call limit. None of those 35 received a correctness reward. For example, `gsm8k:test:588` repeatedly searched the empty question-local document collection for “Jared typing speed increase”; `gsm8k:test:906` similarly repeated “Tim shoe boxes”. Both received four empty search results, then attempted a fifth call and stopped with total reward -0.1. These are observed baseline tool-selection failures, not invented reward-hacking successes.

The stored diagnostic is `results/diagnostics/initial-tool-baseline.json`, with the source-file SHA256 and full observed turns. This is model-assisted inspection of saved traces, not human annotation. No after-training comparison exists yet, so the ten before/after pairs remain PENDING.
