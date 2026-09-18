# pgvector partitioning — evidence

Everything behind the GitHub Discussion on per-collection recall in the
pgvector backend, kept on a separate branch so the code branch stays one
logical unit per change.

**Start with [`results/SUMMARY.md`](results/SUMMARY.md)**: every measurement
in tables, A → B → C side by side, generated from the logs by
`harness/summarize.py` — nothing typed by hand.

| File | What it is |
|---|---|
| `discussion.md` | the proposal, as posted |
| `methodology.md` | what was measured, how, and what the numbers do not cover |
| `connection-leak.md` | the empty-result connection leak, standalone |
| `ivfflat-empty-index.md` | the IVFFlat index built before any rows exist, standalone |
| `pull-request-bodies.md` | three PR bodies, held until a maintainer asks |
| `results/SUMMARY.md` | all measurements as tables |
| `results/*.txt` | the raw campaign logs every figure comes from — 200k modest, 200k generous, 1M, 3M; the multi-KB re-runs; the query-mode, multi-KB-at-default-effort and empty-partition trials. Files ending in `-requetes-aiguilles` or `-sans-prechauffage` are superseded runs, kept for the record. Step labels inside the raw logs are partly in French; the summary and the write-ups are the readable form |
| `harness/` | the campaign scripts that produced them; the benchmark itself is `tests/vector/bench_pgvector_partitioning.py` on the code branch. Host paths are replaced by `$REPO` / `$EVIDENCE_DIR`; the embedding endpoint is read from `OWUI_BENCH_EMBED_URL` / `OWUI_BENCH_EMBED_MODEL` |

The code is on `feat/pgvector-partitioning-v3`; this branch is that branch
plus this directory.
