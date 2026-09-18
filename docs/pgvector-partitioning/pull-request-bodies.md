# Pull request bodies, held until a maintainer asks

The contributing guidelines are explicit: **do not open a code pull request
unless a maintainer has explicitly requested it.** Nothing here is open, and
nothing will be opened without that request. The proposal stage is the
Discussion draft in `owui-partitionnement-discussion.md`.

The checklist also asks that a PR be **one logical unit with no unrelated
commits**. The branch is not one unit — it is three — so this document holds
three PR bodies rather than one. They are independent: the first can be taken
alone, the second can be taken alone, and the third depends on neither.

| | Title | Commits | Depends on |
|---|---|---|---|
| 1 | `fix: release the connection when a pgvector read returns nothing` | 1 | — |
| 2 | `perf: let pgvector keep scanning until enough rows survive the filter` | 1 | — |
| 3 | `feat: partition document_chunk by collection in the pgvector backend` | 3 | — |

Branch against `dev`, five signed commits, in that order.

---

# PR 1 — `fix: release the connection when a pgvector read returns nothing`

## Maintainer Request

*(to be linked — this PR is not opened until one exists)*

## Summary

`PgvectorClient.search`, `.query` and `.get` each return early when the result
set is empty, and that early return skips the `self.session.rollback()` their
populated paths take. The session keeps its transaction open, and with it the
connection it checked out of the pool.

One thread reusing one session never notices. `query_collection` fans out over
`asyncio.to_thread`, and `self.session` is a `scoped_session` — thread-local by
construction — so every thread gets its own Session. A thread that finishes
while still holding a connection never gives it back.

Twelve threads issuing one empty search each, against the default pool (size 5,
overflow 10), leave **12 connections checked out and 0 free**. The next search
waits out the 30-second checkout timeout and fails with `QueuePool limit of size
5 overflow 10 reached`. The backend has stopped answering vector searches and
nothing in the application log says why.

An empty result is not an exotic case. It is what a filtered index scan returns
when none of the candidates it walked belong to the collection being searched,
which happens routinely on a large shared `document_chunk` — so the installs
most likely to hit it are the ones where it hurts most. It is also reachable
from an empty knowledge base, a file whose chunks were deleted, or a metadata
filter that matches nothing.

Nine lines in one file.

## Testing

- `pytest tests/` against PostgreSQL 18.6 / pgvector 0.8.6, in both the
  unpartitioned and partitioned layouts: 127 passed, 13 skipped.
- Added `test_empty_result_returns_the_connection_to_the_pool`, which asserts
  `pool.checkedout() == 0` after a search, a query and a get against a
  collection that does not exist. It **fails on `dev`** (2 failed, both layouts)
  and passes with the change.
- Reproduced the exhaustion directly: 12 threads × 1 empty search, before the
  fix 12 checked out / 0 free; after, 0 checked out / 5 free, and a second burst
  succeeds.
- Checked the direction of causation, since "an exhausted pool causes timeouts
  which produce the empty results" is the obvious alternative. At 3 threads
  against a pool of 15, where a timeout is impossible, a **non-empty** search
  leaves 0 connections checked out and an **empty** one leaves 3 — same threads,
  same concurrency, only the result differs. A subsequent non-empty burst then
  shows 3 checked out *and* 3 free simultaneously, which is a leak signature
  rather than exhaustion. A timeout would also not produce a `SearchResult` at
  all: it raises, and the `except` branch returns `None`.

## Changelog Entry

### Fixed

- pgvector: a search, query or get that returned no rows left its database
  connection checked out, which could exhaust the connection pool on installs
  where per-collection searches often come back empty.

## Additional Context

Found while benchmarking multi-knowledge-base retrieval: ten concurrent
per-collection searches failed at 1M and 3M rows with pool timeouts, and only in
the configuration where searches come back empty. The recorded latency was
30 051 ms, which is the checkout timeout rather than anything the database did.

---

# PR 2 — `perf: let pgvector keep scanning until enough rows survive the filter`

## Maintainer Request

*(to be linked)*

## Summary

pgvector applies `WHERE collection_name = …` **after** walking the ANN index, so
a per-collection search only returns the candidates that happen to survive the
filter. As `document_chunk` grows, fewer do — silently.

Measured on PostgreSQL 18.6 / pgvector 0.8.6, 1M rows of 1024 dimensions, each
knowledge base ~1.5 % of the table, recall@10 against exact kNN over 200 queries
through `PgvectorClient.search()`:

| HNSW `ef_search` | as shipped | with iterative scan |
|---|---|---|
| 10 | 0.143 | 0.892 |
| 40 (the default) | **0.269** | **0.914** |
| 100 | 0.459 | 0.933 |
| 400 | 0.937 | 0.947 |

`hnsw.iterative_scan` / `ivfflat.iterative_scan` let pgvector keep walking until
enough rows survive. This adds a `PGVECTOR_ITERATIVE_SCAN` setting, **off by
default**, and applies the method-appropriate GUC per search when it is on.

**It is a trade, not a free win**: at the default `ef_search` the SQL time of a
search goes from 4.67 ms to 8.72 ms. At a recall of 0.27 the trade looks worth
making, but it is the operator's call — which is why nothing changes unless
the setting is turned on, and why its comment carries the measured figures.

One implementation detail worth flagging: pgvector registers its GUCs when its
library is first loaded into a backend, which only happens on first use of the
vector type. Setting one before then raises `unrecognized configuration
parameter`, which would abort the caller's transaction — so the type is touched
once per connection first, and the setting's presence is confirmed in
`pg_settings` and cached per process.

## Testing

- `pytest tests/` in both layouts: 127 passed, 13 skipped.
- Recall and latency swept at `ef_search` 10 / 40 / 100 / 400 and `probes`
  1 / 10 / √lists / lists÷4, at 200k, 1M and 3M rows, under two build
  parameterisations per method.
- PostgreSQL server log checked for `ERROR`/`FATAL` after a full suite run on a
  freshly recreated container: clean.

## Changelog Entry

### Added

- `PGVECTOR_ITERATIVE_SCAN` (off by default), which lets pgvector keep scanning
  the vector index until enough rows pass the collection filter. Recovers most
  of the recall lost to post-filtering, at some latency cost.

---

# PR 3 — `feat: partition document_chunk by collection in the pgvector backend`

## Maintainer Request

*(to be linked)*

## Summary

Even with iterative scans, one table and one index serve every collection. Two
costs follow that no setting touches.

**The index carries rows that never use it.** Open WebUI creates a collection
per uploaded file, per user memory and per web search; on a realistic table
those are ~70 % of rows, and a query over one of them filters a handful of rows
and takes the btree anyway. At 3M rows the HNSW index is 10.9 GB where the rows
that actually use it need 6.1 GB; the IVFFlat index is 22.9 GB, larger than the
16.2 GB heap.

**Vacuum walks all of it.** With the IVFFlat index in place, reclaiming one
deleted knowledge base took 59.5 s against 0.8 s with per-collection partitions;
after a raw `DELETE`, a vacuum that does clean the indexes went from 13.3 s to
0.2 s. Deleting a knowledge base becomes a `DROP` of its partition, leaving no
dead tuples, and autovacuum works per partition rather than on the parent.

This adds `PARTITION BY LIST (part_key)` behind `PGVECTOR_PARTITIONING`,
**default off**: a knowledge base gets its own partition with its own vector and
GIN indexes; everything else is hashed into a fixed number of buckets carrying
only a btree on `collection_name`. `VectorDBBase` is unchanged — the partition
key is derived from the collection name at each call site. A migration script
converts an existing table and can roll back.

1M rows, HNSW `m=16`, knowledge-base collections, default `ef_search`:

| | as shipped | + iterative scan | partitioned |
|---|---|---|---|
| recall@10 | 0.269 | 0.914 | **0.977** |
| worst decile | 0.00 | 0.80 | **1.00** |
| SQL time, median | 4.67 ms | 8.72 ms | **3.06 ms** |
| vector index | 4186 MB | 4186 MB | **2103 MB** |

Partitioning is the only one of the three that improves recall and latency at
once.

For the ~70 % of rows in bucketed collections — the ones that *lose* their
vector index — recall stays **1.000 across all 208 measurements**, and their
median SQL time goes from 2.36 ms to 2.14 ms.

## Testing

- `pytest tests/` in both layouts, both pgcrypto modes: 127 passed, 13 skipped.
  The suite covers insert, upsert, get, search, query, delete,
  delete_collection, has_collection, hybrid search still declining as upstream
  does, partition routing, concurrent partition creation, concurrent worker
  startup, and the migration including rollback, resumption, an encrypted
  table, and refusal to swap when the copy is incomplete. Without a database
  the suite skips rather than errors.
- Four measurement campaigns — 200k (modest and generous memory profiles), 1M
  and 3M (generous) — each covering both index methods, two build
  parameterisations per method, four search efforts per parameterisation, and
  both collection families reported separately. 632 recorded measurements.
- Migration timed end to end: 31 s for 194k rows, 207 s for 970k, 676 s for
  2.91M.
- Method and limitations written up in full; see Additional Context.

## Changelog Entry

### Added

- `PGVECTOR_PARTITIONING` (default off), which partitions `document_chunk` by
  collection so each knowledge base gets its own vector index, and a migration
  script to convert an existing table.

## Additional Context

Things that argue *against* this change, stated here rather than left for
review to find:

- **Selecting several knowledge bases does not get faster.** `query_collection`
  issues one search per collection and merges afterwards; each search inherits
  the per-collection recall (at the default `ef_search`, 0.16 unpartitioned
  against 0.99 partitioned for one knowledge base), but wall time is set by the
  client-side fan-out and is the same in both layouts.

- **IVFFlat regresses at pgvector's default `probes` with the single global
  `lists`** (0.73 against 0.84 at 1M), and only wins once `lists` is sized per
  partition (0.90) — a setting that does not exist yet.
- **A manual `VACUUM` of the whole table touches 53 relations instead of one**
  — 0.7 s against 0.1 s at 3M when there is nothing to clean. Autovacuum is
  unaffected, since it works per partition.

- **Migration needs downtime.** `ATTACH PARTITION` cannot avoid it: PostgreSQL
  refuses to attach a table whose partition key holds more than one value.
- **Three new settings**, which is more than the guidelines would like.

The routing is the part we are least sure about. It is an allow-list — matching
explicitly on what earns a partition — rather than a deny-list, because a
deny-list would give a partition to any collection shape added later and put DDL
in the ingestion path. That reasoning may not survive contact with how you
expect collection naming to evolve.
