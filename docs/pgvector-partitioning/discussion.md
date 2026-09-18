# pgvector: per-collection search returns a fraction of its results, and most of the index is unused

*Draft for a GitHub Discussion (Ideas). Not a pull request — following the
contributing guidelines, this is the proposal stage. Everything below has an
implementation behind it, tested end to end on a branch against `dev` (links
at the end); we are glad to open it as pull requests if a maintainer asks, or
to rework it first on your feedback.*

Related, but a different bottleneck: #17998 and #20737 are about the Python BM25
fallback fetching whole collections. This is about the vector index itself, so
the two are complementary rather than overlapping.

---

## What we ran into

`document_chunk` holds every collection's chunks in one table under one global
vector index, and a search is scoped with `WHERE collection_name = …`. pgvector
applies that filter after walking the ANN index, so only the candidates that
happen to survive it come back. As the table grows, fewer do. Nothing errors —
the answers just quietly get worse, which is what makes it easy to miss.

On PostgreSQL 18.6 / pgvector 0.8.6, 1M rows of 1024 dimensions, each knowledge
base about 1.5% of the table, recall@10 against exact kNN over 200 queries,
measured through `PgvectorClient.search()`:

| HNSW `ef_search` | recall@10 | worst decile |
|---|---|---|
| 10 | 0.143 | 0.00 |
| 40 (the default) | 0.269 | 0.00 |
| 100 | 0.459 | 0.20 |
| 400 | 0.937 | 0.80 |

At the default setting roughly three of every four correct neighbours are
missing, and the worst tenth of queries return nothing relevant. At 3M rows the
default gives 0.284.

### A test with real embeddings — and what it does and does not show

Synthetic vectors can make recall@10 feel abstract, so we buried 199 passages
embedded by a real model in one otherwise-synthetic knowledge base, along with
near-duplicates built from those passages at six noise levels, and asked 48
questions whose answers are among them. Two ranks are recorded per question:
where an exact sort puts the right passage — a property of the embedding model
— and where the index puts it. Only the gap belongs to the index.

At 1M rows, HNSW `m=16`, `ef_search=10`: the model ranks the right passage first
for **44 of 48** questions and within its first ten for 45. The index puts it
first for **17** and within the first ten for 17. At `ef_search=400`, 37 and 39.
(Counted within the first ten on purpose: pgvector returns at most `ef_search`
rows unless iterative scans are on, so at `ef_search=10` a "within 50" count
would compare the ten rows the index can return against the fifty an iterative
scan returns.)

That loss is **not** the post-filter loss above, and two things in the numbers
say so. Every real passage and every near-duplicate built from one live in the
same knowledge base, so near a real question the collection filter has nothing
to discard. Accordingly, iterative scans — which exist to recover what the
filter discards — change nothing within the first ten in any row of this test:
identical to as-shipped everywhere. And the partitioned layout, which has no
filter to apply at all, is no better at `m=16`: 15 within ten against 17 at
`ef_search=10`, 36 against 39 at 400. This is the index's own approximation
inside a dense region of near-duplicates, and it depends on the graph rather
than on the layout: at `m=32` the partitioned layout finds 45 of 45 within ten
at `ef_search=10` where the unpartitioned one finds 41, and at 400 all three
find 45. A per-collection graph over 15k rows of which 14.8k are near-duplicates
of 199 centres is a degenerate case for HNSW at `m=16` — neighbour lists fill
with clones and the walk settles in the wrong cluster — which is why that
layout does slightly worse than the global graph, where the same rows are
interleaved with a million others. At 3M the pattern repeats (partitioned 27
and 36 within ten at `ef_search=10` against 23 and 28; 38 against 42 at
`m=16`, 400), and at 200k, where the region is sparser, nothing misses
anything under HNSW. So this is a second, independent point rather than an
illustration of the first: an approximate index at low effort loses real
answers, and the graph parameter matters at least as much as the layout.

## Three other things, independent of any proposal

**A read that returns nothing never gives its connection back.** `search`,
`query` and `get` each return early on an empty result set without the rollback
their populated paths take, so the session keeps its transaction — and its
pooled connection. `query_collection` fans out over threads and `scoped_session`
is thread-local, so each thread leaks one. Twelve empty searches leave twelve
connections checked out and none in the pool; the next search waits out the
30-second checkout timeout and fails. This compounds with the recall problem
above, since a filtered index scan that finds no surviving candidates is exactly
how an empty result arises. Nine lines to fix, with a regression test; written
up separately, and it stands entirely on its own.

**IVFFlat indexes are created before any rows exist.** `__init__` calls
`create_all()` and then `_ensure_vector_index()`, so on a fresh install the index
is built on an empty table. IVFFlat trains its list centroids at `CREATE INDEX`
time, and pgvector's documentation asks for the index to be created after the
data. It is never rebuilt afterwards. Measured on a 15k-row index at
`ivfflat.probes = 1`, which is pgvector's default and one the codebase never
changes: **recall 0.733 for an index built empty against 1.000 for the same
index built after loading**, with the worst decile at 0.50. The gap narrows as
`probes` rises, so the training deficit costs most exactly where you were trying
to be fast.

**An IVFFlat index can fail to build.** At 1M rows with `lists = 1000` —
pgvector's own `rows/1000` guidance — and `maintenance_work_mem = 256 MB`:
`ERROR: memory required is 403 MB`. pgvector samples roughly 50 × `lists`
vectors to train, so the demand follows `lists` rather than the table. Since the
index is created inside the client's `__init__`, it surfaces as a failed startup.

## The cheap fix first

Before anything structural: **`hnsw.iterative_scan` is a GUC the codebase never
sets**, and it lets pgvector keep walking the index until enough rows survive the
filter. At the default `ef_search` it takes recall@10 from 0.269 to 0.914.

It is a trade rather than a free win — the SQL time of a search goes from 4.7 ms
to 8.7 ms — so the branch adds it as a setting, `PGVECTOR_ITERATIVE_SCAN`,
**off by default**: nothing changes for an install that does not turn it on. At
a recall of 0.27 the trade looks worth making, and it is a few dozen lines,
completely independent of everything below, so it may be the only part of this
worth acting on.

## Where we got to on the structural side

Even with iterative scans, the shape stays: one table, one index, every
collection sharing it. Two costs follow, and no GUC touches either.

The index carries rows that never use it. Open WebUI creates a collection per
uploaded file, per user memory and per web search; on a realistic table those
are around 70% of rows, and a query over one of them filters a handful of rows
and takes the btree anyway. At 3M rows the HNSW index is 10.9 GB where the rows that actually use it need 6.1 GB; the IVFFlat index is 22.9 GB, larger than the
16.2 GB heap it indexes. And vacuum walks all of it — with the IVFFlat index in
place, reclaiming one deleted knowledge base took 59.5 s against 0.8 s when
each collection had its own partition. Deleting a knowledge base becomes a
`DROP` of its partition, which leaves no dead tuples at all; deleting a file
leaves them in a bucket that carries only a btree, so autovacuum — which works
per partition, never on the parent — never crosses a vector index for it.

### What we considered and set aside

We looked at a number of approaches before settling on partitioning, and we may
well have dismissed something too quickly:

| Approach | Why we set it aside |
|---|---|
| Switch to Qdrant or Milvus multi-tenancy | pgvector is the only backend implementing `hybrid_search()`; `VectorDBBase.hybrid_search()` returns `None` and no other backend overrides it, so the fallback rebuilds BM25 in memory per query |
| External knowledge feature (`retrieval/external.py`) | Retrieval only, pure vector distance, no BM25 or RRF — loses hybrid search |
| SQL view | Purely logical, no index of its own, no effect on the plan |
| Materialised view per knowledge base | Copies the vectors, and `REFRESH` rebuilds everything; no incremental refresh in PostgreSQL |
| Multicolumn HNSW `(collection_name, vector)` | Not supported by pgvector, and meaningless anyway: HNSW is a proximity graph with no ordering, so no prefix semantics |
| Partial index per knowledge base on one table | Works and is reversible, but vacuum stays table-wide and walks every partial index, so each one added makes vacuum dearer; and excluding `file-*` needs a generated column plus query-side changes |
| One table per knowledge base | Requires parameterising the table name throughout `pgvector.py`, dynamic `UNION ALL` for multi-collection queries, and invites schema drift between tables |
| Table inheritance (`INHERITS`) | Table inheritance can emulate partitioning, but requires manual trigger-based row routing and relies on constraint exclusion for pruning; declarative partitioning (PostgreSQL 10+) provides these mechanisms natively. |
| A `DEFAULT` partition | Creating each new partition makes PostgreSQL scan the whole default under `ACCESS EXCLUSIVE` to prove no row belongs in it |
| One partition per `file-{id}` | `CREATE TABLE … PARTITION OF` takes `ACCESS EXCLUSIVE` on the parent, and a collection is created on every upload |
| `PARTITION BY HASH (collection_name)` alone | Prunes well, but mixes knowledge bases and per-file collections in the same partitions, so they cannot have different indexes |
| Partitioning by expression | PostgreSQL prunes on an expression only when the query filters on that exact expression, so `WHERE collection_name = …` would not prune |

### What we ended up trying

`PARTITION BY LIST (part_key)` behind a flag, default off: a knowledge base gets
its own partition with its own vector and GIN indexes, everything else is hashed
into a fixed number of buckets carrying only a btree on `collection_name`.

The part we are least sure about is the routing. We made it an allow-list —
matching explicitly on what earns a partition — rather than bucketing known
prefixes, because a deny-list would give a partition to any collection shape
added later, including the sha256-named ones from `/process/text` and
`/process/web`, and put DDL in the ingestion path. That reasoning may not hold up
against how you expect collection naming to evolve.

`VectorDBBase` is unchanged; the partition key is derived from the collection
name at each call site.

### How it measures out

1M rows, HNSW `m=16`, knowledge-base collections, at the default `ef_search`:

| | as shipped | + iterative scan | partitioned |
|---|---|---|---|
| recall@10 | 0.269 | 0.914 | **0.977** |
| worst decile | 0.00 | 0.80 | **1.00** |
| SQL time, median | 4.67 ms | 8.72 ms | **3.06 ms** |
| vector index | 4186 MB | 4186 MB | **2103 MB** |

Partitioning is the only one of the three that improves recall and latency at
the same time. Across the tuning range the ordering holds: at `ef_search` 10 the
three are 0.143 / 0.892 / 0.966, at 400 they are 0.937 / 0.947 / 0.998.

Selecting several knowledge bases changes nothing structurally: `query_collection`
issues one search per collection in both layouts and merges afterwards, so each
of the N searches inherits the per-collection recall above. Measured through
`query_collection` itself at 200k rows, default `ef_search`, with N = 1 / 3 / 10:
unpartitioned 0.16 / 0.36 / 0.93, with iterative scans 0.96 / 1.00 / 1.00,
partitioned 0.99 / 1.00 / 1.00. The unpartitioned figure climbs with N because
the merged answer spans a growing share of the table and the filter discards
less. Wall time is the same in every layout — about 16, 40 and 130 ms — and is
set by the client-side fan-out rather than by the index. At a high search
effort the partitioned layout is slightly slower in this fan-out (at 1M and 3M,
`ef_search=400`, N=10: 168 ms against 135 ms and 228 against 136): at 400
candidates the per-search execution costs the same in both layouts — the walk
no longer stops early, which is where partitioning saves its work — and the
partitioned parent adds about a millisecond of planning per query (1.0–1.6 ms
against 0.6–0.7 ms on the partition alone), which the fan-out multiplies.

One caveat that works against the baseline rather than for it: the query
vectors are rows of the collection searched, so each query's own row is one of
its ten exact neighbours and a guaranteed hit. Re-measured at 200k rows with
that row excluded, the unpartitioned figure drops from 0.294 to 0.206 while the
partitioned one stays at 0.994 — the tables above understate the gap.

For the ~70% of rows in bucketed collections — the ones that *lose* their vector
index — recall stays 1.000 across all 208 measurements, since they were already
served by a bitmap scan and an exact sort. Their median SQL time goes from
2.36 ms to 2.14 ms: marginally better, consistently.

### Where it is less clear-cut


- **IVFFlat: a regression with the single global `lists`, a win only with a
  per-partition one.** At pgvector's default `probes = 1`, the partitioned
  index is *worse* than the global one when both take the one
  `PGVECTOR_IVFFLAT_LISTS` value — 0.73 against 0.84 at 1M, 0.59 against 0.71
  at 3M — because a `lists` suited to the whole table is far too high for a
  15k-row partition. Sized from each partition's own row count it beats every
  unpartitioned point at that setting: 0.90 at 1M and 0.86 at 3M, at 4–6 ms
  against 8–18 ms. Above the default, the unpartitioned planner abandons the
  index for an exact sort (recall 1.0 at 46 ms / 135 ms), and with per-partition
  `lists` the partitioned one does the same (43 ms / 129 ms). So IVFFlat needs a
  `lists` chosen per partition, and that is a setting which does not exist yet
  — see the separate note on IVFFlat.
- **The partition's own index has to be good enough.** Partitioning removes the
  filtering loss and then exposes the index's intrinsic approximation: at 1M and
  the default `ef_search`, `m=16` on a 15k-row partition reaches 0.977 where
  `m=32` reaches 1.000. On one shared index `m` was masked by the filter; per
  partition it is visible.

- **Migration needs downtime** — the copy is a snapshot: 31 s for 194k rows,
  207 s for 970k, 676 s for 2.91M. `ATTACH PARTITION` does not avoid it: PostgreSQL refuses to attach a
  table whose partition key holds more than one value.
- **The planner may decline the partition's index** and sort exactly instead,
  which shows up as recall 1.000 at 130 ms at 3M.
- **Three new settings** added. The flag is hard to avoid; the bucket count and the routing pattern
  could probably be constants.

## What would help us most

1. Whether the diagnosis is useful to you at all — particularly the connection
   leak, which is independent of everything else and is nine lines.
2. Whether `hnsw.iterative_scan` is worth setting on its own. It is small,
   isolated, and recovers most of the recall (but increases latency).
3. If partitioning is a direction you would consider, whether the allow-list
   routing is the right shape — and whether you would rather implement it
   yourselves.

The implementation is on a fork, as five commits against `dev` — the connection
leak fix on its own, then the iterative scan change, then partitioning, then
the migration script, then a test suite and benchmark harness — with the
measurements alongside:

- code: https://github.com/runyournode/open-webui/tree/feat/pgvector-partitioning-v3
- diff against `dev`: https://github.com/open-webui/open-webui/compare/dev...runyournode:open-webui:feat/pgvector-partitioning-v3
- measurements and write-ups (methodology, the two standalone notes, raw logs): https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning
- every measurement in tables: https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/results/SUMMARY.md

No PR is open, and none will be unless a maintainer asks for one.
