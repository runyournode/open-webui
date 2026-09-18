# issue: a pgvector read that returns nothing never gives its connection back

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## What happens

In `backend/open_webui/retrieval/vector/dbs/pgvector.py`, `PgvectorClient.search`, `.query` and `.get` return early when the result set is empty, skipping the `self.session.rollback()` the populated path performs:

```python
results = result_proxy.all()
...
if not results:
    return SearchResult(ids=ids, ...)      # no rollback

for row in results:
    ...
self.session.rollback()  # read-only transaction
return SearchResult(...)
```

The session keeps its transaction open and, with it, the connection it checked out of the pool.

## Why one thread never sees it

A single thread reusing a single session runs the next statement on the same open transaction.
`query_collection` fans out over `asyncio.to_thread`, and `PgvectorClient.session` is a `scoped_session` — thread-local — so every thread gets its own Session.
A thread that finishes while still holding a connection never gives it back.

Twelve threads issuing one empty search each, against the default pool (size 5, overflow 10): **12 connections checked out, 0 free**.
The next burst waits out the 30 s checkout timeout and fails with `QueuePool limit of size 5 overflow 10 reached`.
Vector search is down and nothing in the application log says why.

## Direction of causation

Checked that it is not the other way round (exhausted pool → timeouts → empty results): 3 threads against a pool of 15, where no timeout can occur, the only variable being whether the searched collection has rows.

| | rows returned | checked out | free |
|---|---|---|---|
| 3 threads, non-empty | 10, 10, 10 | 0 | 3 |
| 3 threads, **empty** | 0, 0, 0 | **3** | **0** |
| 3 threads, non-empty again | 10, 10, 10 | 3 | 3 |

Only the empty case retains its connections. The third row — three checked out *and* three free — is a leak, not exhaustion.
A timeout could not produce these rows: it raises, and the `except` branch returns `None`, not an empty `SearchResult`.

## Why an empty result is not rare

It is what a filtered ANN scan returns when none of its candidates belong to the collection searched (<RECALL_ISSUE_URL>), which on a large shared table is routine.
It is also what an empty knowledge base, a file whose chunks were deleted, or a metadata filter that matches nothing return.

## Fix

A `rollback()` before each of the three early returns: nine lines in one file, commit 57ab728ec on https://github.com/runyournode/open-webui/tree/feat/pgvector-partitioning-v3 (`fix: release the connection when a pgvector read returns nothing`).
The regression test asserts `pool.checkedout() == 0` after a search, a query and a get against a collection that does not exist; it fails on `dev` and passes with the change.
Reproduction script: `harness/leak_causality.py` on https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning.
Context: <DISCUSSION_URL>.
