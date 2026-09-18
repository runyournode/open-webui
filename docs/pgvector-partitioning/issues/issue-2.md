# issue: pgvector `search`, `query` and `get` keep their pooled connection when the result is empty

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## Summary

In `backend/open_webui/retrieval/vector/dbs/pgvector.py`, `search`, `query` and `get` each return early when the result set is empty, and that early return skips the `self.session.rollback()` the populated path performs. The session keeps its transaction open and, with it, the connection it checked out of the pool.

## Steps to Reproduce

`query_collection` fans out over `asyncio.to_thread`, and `PgvectorClient.session` is a `scoped_session`, so every thread gets its own Session. Start 12 threads that each run one `search()` against a collection with no rows, then inspect `client.session.get_bind().pool`: 12 connections checked out, 0 free (default pool: size 5, overflow 10). The next search waits out the 30 s checkout timeout and fails with `QueuePool limit of size 5 overflow 10 reached`. Script: `harness/leak_causality.py` on the evidence branch linked below.

## Expected Behavior

A read that finds nothing ends its transaction and returns its connection, as the populated path does.

## Actual Behavior

One connection retained per thread that ever got an empty result. Direction of causation checked at 3 threads against a pool of 15, where no timeout can occur: a non-empty search leaves 0 connections checked out, an empty one leaves 3; a following non-empty burst shows 3 checked out and 3 free at once, which is a leak, not exhaustion. A timeout could not produce these results anyway: it raises, and the `except` branch returns `None`, not an empty `SearchResult`.

Empty results are not rare: a filtered ANN scan whose candidates all belong to other collections returns nothing (see the recall issue linked below), as does an empty knowledge base, a file whose chunks were deleted, or a metadata filter that matches nothing.

## Additional Information

Fix: a `rollback()` before each of the three early returns, nine lines in one file, with a regression test that fails on `dev` and passes with the change — commit 57ab728ec on https://github.com/runyournode/open-webui/tree/feat/pgvector-partitioning-v3 (`fix: release the connection when a pgvector read returns nothing`). Write-up: https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/connection-leak.md. Found while measuring the recall problem described in the Discussion: <DISCUSSION_URL>.
