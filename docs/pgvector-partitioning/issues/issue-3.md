# issue: pgvector IVFFlat index is created on an empty table and never retrained

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## Summary

`PgvectorClient.__init__` runs `Base.metadata.create_all()` and then `_ensure_vector_index()`. On a fresh install the index is therefore built on an empty `document_chunk`, and `_ensure_vector_index` returns early whenever an index of that name exists, so it is never rebuilt. IVFFlat is the default method (`PGVECTOR_INDEX_METHOD` unset, no halfvec), and IVFFlat trains its list centroids at `CREATE INDEX` time; pgvector's documentation asks for the index to be created after the data is loaded. An install that upgraded into the version that introduced the index is unaffected — the table already had rows.

## Steps to Reproduce

Fresh install with `VECTOR_DB=pgvector`, upload documents, search. Or, to isolate the effect: on a 15k-row collection build the IVFFlat index (`lists = 15`) once before inserting the rows and once after, and compare recall@10 against an exact sort through `PgvectorClient.search()` (`harness/ivfflat_empty_partition.py` on the evidence branch linked below).

## Expected Behavior

The same recall whichever order the index and the rows were created in.

## Actual Behavior

At `ivfflat.probes = 1`, pgvector's default and one the codebase never changes: recall@10 **0.733** for the index built empty against **1.000** for the same index built after loading, worst decile 0.50 against 1.00. At `probes = 3`: 0.968 against 1.000. Nothing is logged; the index is valid, only untrained.

Related: `PGVECTOR_IVFFLAT_LISTS` is a single global value, and the build needs about 50 × `lists` sample vectors in `maintenance_work_mem`. At 1M rows with `lists = 1000` (pgvector's own `rows/1000` guidance) and `maintenance_work_mem = 256 MB`, `CREATE INDEX` fails with `ERROR: memory required is 403 MB`; since it runs inside `__init__`, the backend fails to start.

## Additional Information

A proposed fix (defer the vector index until the table holds enough rows to train on, size `lists` from the row count) is written up at https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/ivfflat-empty-index.md; measurements at https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/results/ivfflat-empty-partition.md. Context: <DISCUSSION_URL>.
