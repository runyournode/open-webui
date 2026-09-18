# issue: the pgvector IVFFlat index is built before there is anything to train on, and never rebuilt

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## What happens

pgvector builds an IVFFlat index by running k-means over the table's rows to place its list centroids, and its documentation says: *"Create the index after the table has some data."* `PgvectorClient.__init__` does the opposite on a fresh install:

```python
Base.metadata.create_all(bind=connection)          # creates an empty document_chunk
index_method, index_options = self._vector_index_configuration()
self._ensure_vector_index(index_method, index_options)   # indexes it, with zero rows
```

`_ensure_vector_index` returns early whenever an index of that name exists, so the index is never rebuilt once data arrives. IVFFlat is the default method (`PGVECTOR_INDEX_METHOD` unset, no halfvec). An install that upgraded into the version that introduced the index is unaffected — its table already had rows; a fresh install is not. The index is valid, queries work, nothing is logged: it is simply untrained.

## Measured

Through `PgvectorClient.search()`, on a 15k-row collection with `lists = 15` (pgvector's `rows/1000`), 40 queries, recall@10 against an exact sort. The only difference between the two rows of each pair is *when* the index was built:

| `probes` | index built | recall@10 | worst decile | median latency |
|---|---|---|---|---|
| **1 (default)** | after loading | **1.000** | 1.00 | 14.1 ms |
| **1** | **empty** | **0.733** | **0.50** | 14.5 ms |
| 3 | after loading | 1.000 | 1.00 | 17.1 ms |
| 3 | empty | 0.968 | 0.90 | 20.6 ms |

At `ivfflat.probes = 1` — pgvector's default, which the codebase never changes — an index trained on nothing returns 73 % of the correct neighbours where a trained one returns 100 %, and the worst tenth of queries returns half. The gap narrows as `probes` rises: the training deficit costs most exactly where one is trying to be fast.

## A related failure: `lists` and build memory

`PGVECTOR_IVFFLAT_LISTS` is one global value, and the build needs roughly 50 × `lists` sample vectors in `maintenance_work_mem`. At 1M rows with `lists = 1000` — pgvector's own `rows/1000` guidance — and `maintenance_work_mem = 256 MB`, `CREATE INDEX` fails with `ERROR: memory required is 403 MB, maintenance_work_mem is 256 MB`. Because it runs inside `__init__`, the backend fails to start.

## Proposed fix

Defer the vector index until the table holds enough rows to train on (a bounded existence check, `SELECT count(*) FROM (SELECT 1 FROM … LIMIT :n)`), then build it once over real data with `lists` sized from the row count actually present; below the threshold a search is an exact scan of a small table, correct by construction. A narrower variant defers IVFFlat only and leaves HNSW eager, which pgvector explicitly supports. Write-up: https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/ivfflat-empty-index.md; measurement: `results/ivfflat-empty-partition.md` in the same directory; reproduction: `harness/ivfflat_empty_partition.py`. Context in the Discussion: <DISCUSSION_URL>.
