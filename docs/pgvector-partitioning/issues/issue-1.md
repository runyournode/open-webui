# issue: pgvector per-collection search returns a fraction of the true neighbours

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## Summary

`document_chunk` is one table under one global vector index, and a search is scoped with `WHERE collection_name = …`. pgvector applies that filter after walking the ANN index, so a query returns only the candidates that happen to belong to the collection. The larger the table and the smaller the collection, the fewer survive. No error is raised; results are silently incomplete.

## Steps to Reproduce

Load 1M chunks of 1024 dimensions across 20 knowledge bases of 15k rows (1.5 % of the table each) plus per-file, per-memory and per-web-search collections for the remaining 70 %; build the default index; search one knowledge base through `PgvectorClient.search()` and compare with an exact kNN over the same collection. The harness that does this is `tests/vector/bench_pgvector_partitioning.py` on the branch linked below (`generate`, `index`, `measure --config A --target kb`).

## Expected Behavior

recall@10 close to 1: the ten nearest chunks of the collection are returned.

## Actual Behavior

HNSW `m=16`, 200 queries per point, recall@10 against exact kNN: `ef_search` 10 → 0.143, 40 (default) → **0.269**, 100 → 0.459, 400 → 0.937. Worst decile at the default: 0.00. At 3M rows the default gives 0.284. IVFFlat at its default `probes=1` gives 0.835 (`lists=1000`); at any higher `probes` the planner abandons the index for an exact sort, correct but at 46 ms (1M) and 135 ms (3M) per search.

`hnsw.iterative_scan` / `ivfflat.iterative_scan`, which make pgvector keep walking until enough rows pass the filter, are never set by the codebase. Setting `hnsw.iterative_scan = relaxed_order` takes the default point from 0.269 to 0.914, with the SQL time of a search going from 4.7 ms to 8.7 ms.

## Additional Information

Measurements, method and a structural proposal (per-collection partitioning) are in the Discussion: <DISCUSSION_URL>. Raw logs and summary tables: https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning. Related to #17998 and #20737, which are about the Python BM25 fallback rather than the vector index.
