# issue: pgvector per-collection search returns a fraction of the true neighbours

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## Summary

`document_chunk` is one table under one global vector index, and a search is scoped with `WHERE collection_name = …`. pgvector applies that filter after walking the ANN index, so a query returns only the candidates that happen to belong to the collection. The larger the table and the smaller the collection, the fewer survive. No error is raised; results are silently incomplete.

## Steps to Reproduce

Load 1M chunks of 1024 dimensions across 20 knowledge bases of 15k rows (1.5 % of the table each) plus per-file, per-memory and per-web-search collections for the remaining 70 %; build the default index; search one knowledge base through `PgvectorClient.search()` and compare with an exact kNN over the same collection. The harness that does this is `tests/vector/bench_pgvector_partitioning.py` on the branch linked below (`generate`, `index`, `measure --config A --target kb`).

## Expected Behavior

recall@10 close to 1: the ten nearest chunks of the collection are returned.

## Actual Behavior

HNSW `m=16`, 200 queries per point, recall@10 against an exact kNN over the same collection: 0.143 at `ef_search=10`, **0.269 at the default 40**, 0.459 at 100, 0.937 at 400 (SQL time 3.0 → 4.7 → 7.0 → 9.3 ms). Worst decile at the default: 0.00 — one query in ten gets nothing relevant. At 3M rows the default gives 0.284. IVFFlat, the default method, gives 0.835 at its default `probes=1`; at any higher `probes` the planner drops the index for an exact sort, which is correct but costs 46 ms (1M) and 135 ms (3M) per search.

Why: an HNSW scan collects `ef_search` candidates — the query's nearest neighbours across the *whole* table — and the `collection_name` predicate is applied to those candidates afterwards. A collection holding 1.5 % of the rows owns about 1.5 % of them, so most of the requested ten are simply never examined. Raising `ef_search` widens the candidate list and helps slowly, at the cost shown above; it does not change the mechanism.

pgvector 0.8 added the remedy for exactly this case: iterative scans (`hnsw.iterative_scan`, values `off | relaxed_order | strict_order`; `ivfflat.iterative_scan`, values `off | relaxed_order`; both default `off`, bounded by `hnsw.max_scan_tuples` and `ivfflat.max_probes`), which make the scan keep going after filtering until it has the requested number of rows. Open WebUI never sets them. With `hnsw.iterative_scan = relaxed_order`, the default point goes from 0.269 to 0.914 and the SQL time of a search from 4.7 ms to 8.7 ms.

## Additional Information

Measurements, method and a structural proposal (per-collection partitioning) are in the Discussion: <DISCUSSION_URL>. Raw logs and summary tables: https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning. Related to #17998 and #20737, which are about the Python BM25 fallback rather than the vector index.
