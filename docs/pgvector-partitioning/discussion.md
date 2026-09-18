# pgvector: a per-collection search on the shared vector index loses most of its neighbours — measurements and a partitioning proposal

Measurements of a recall problem in the pgvector backend, one setting that recovers most of it, and a tested structural change. Three parts stand on their own and are filed as issues: the recall loss itself (#ISSUE_1), a connection leak on empty results (#ISSUE_2), and the IVFFlat index built before any rows exist (#ISSUE_3). The implementation is on a fork as five commits against `dev`, tested end to end; we can open it as pull requests if asked, or rework it first on your feedback. Links at the end.

## The problem

`document_chunk` is one table under one global vector index, and a search is scoped with `WHERE collection_name = …`. pgvector applies that filter after the ANN walk, so only the candidates that happen to belong to the collection come back. The smaller the collection relative to the table, the fewer survive, and nothing signals it.

PostgreSQL 18.6, pgvector 0.8.6, 1M rows of 1024 dimensions, 20 knowledge bases of 15k rows (1.5 % each), the remaining 70 % in per-file, per-memory and per-web-search collections; recall@10 against an exact kNN over 200 queries, through `PgvectorClient.search()`, HNSW `m=16`:

| `ef_search` | recall@10 | worst decile |
|---|---|---|
| 10 | 0.143 | 0.00 |
| 40 (default) | **0.269** | 0.00 |
| 100 | 0.459 | 0.20 |
| 400 | 0.937 | 0.80 |

At 3M rows the default gives 0.284. IVFFlat, the default method, gives 0.835 at its default `probes=1`; above that the planner abandons the index for an exact sort — correct, at 46 ms (1M) and 135 ms (3M) per search.

## A setting that recovers most of it

`hnsw.iterative_scan` and `ivfflat.iterative_scan` make pgvector keep walking until enough rows pass the filter. The codebase never sets them. At the default `ef_search`, `relaxed_order` takes recall@10 from 0.269 to 0.914 and the SQL time of a search from 4.7 ms to 8.7 ms. The branch exposes it as `PGVECTOR_ITERATIVE_SCAN`, off by default. It is independent of everything below.

## Partitioning by collection

What iterative scans do not change: one index carries every collection. The per-file, per-memory and per-web-search collections — about 70 % of rows — are looked up by name and sorted over a handful of rows; they never use the vector index but sit in it. At 3M rows the HNSW index is 10.9 GB where the knowledge bases alone need 6.1 GB, the IVFFlat index is 22.9 GB against a 16.2 GB heap, and a vacuum that has to clean the index after a knowledge base is deleted takes 59.5 s (IVFFlat in place).

The change: `PARTITION BY LIST (part_key)` behind `PGVECTOR_PARTITIONING`, off by default. A knowledge base gets its own partition with its own vector and GIN indexes; everything else is hashed into a fixed number of buckets carrying only a btree on `collection_name`. Routing is an allow-list (bare UUID or `knowledge-bases`) rather than a deny-list of known prefixes, so a collection shape added later is bucketed instead of getting a partition and putting DDL in the ingestion path. `VectorDBBase` is unchanged; the key is derived from the collection name at each call. A script migrates an existing table (31 s for 194k rows, 207 s for 970k, 676 s for 2.91M) and rolls back.

1M rows, HNSW `m=16`, knowledge bases, default `ef_search`:

| | as shipped | iterative scan | partitioned |
|---|---|---|---|
| recall@10 | 0.269 | 0.914 | **0.977** |
| worst decile | 0.00 | 0.80 | **1.00** |
| SQL time, median | 4.67 ms | 8.72 ms | **3.06 ms** |
| vector index | 4186 MB | 4186 MB | **2103 MB** |

The ordering holds across the range: 0.143 / 0.892 / 0.966 at `ef_search` 10, 0.937 / 0.947 / 0.998 at 400. The bucketed 70 % keep recall 1.000 in all 208 measurements (they were already served by the btree and an exact sort) and go from 2.36 ms to 2.14 ms. Deleting a knowledge base becomes a `DROP` of its partition; the vacuum after it goes from 59.5 s to 0.8 s.

Two things a reader should know about these figures. The query vectors are stored rows, so every configuration gets one guaranteed hit in ten; measured with that row excluded (200k rows) the as-shipped figure is 0.206 instead of 0.294 and the partitioned one 0.994 instead of 0.996 — the tables understate the gap. And selecting several knowledge bases changes nothing structurally: `query_collection` issues one search per collection in both layouts and merges afterwards. Measured through it at 200k rows and the default effort, merged recall for N = 1 / 3 / 10 is 0.16 / 0.36 / 0.93 as shipped and 0.99 / 1.00 / 1.00 partitioned; wall time is the same in both (about 16, 40 and 130 ms), set by the client-side fan-out.

## Alternatives considered

| Approach | Why not |
|---|---|
| Qdrant or Milvus multi-tenancy | pgvector is the only backend with native `hybrid_search()`; the others fall back to rebuilding BM25 in Python per query |
| `retrieval/external.py` | Vector distance only, no BM25 or RRF |
| SQL view or materialised view per knowledge base | A view has no index of its own; a materialised view copies the vectors and `REFRESH` rebuilds everything |
| Multicolumn HNSW `(collection_name, vector)` | Not supported by pgvector, and HNSW has no prefix semantics |
| Partial index per knowledge base | Reversible, but vacuum stays table-wide and walks every partial index; excluding `file-*` needs a generated column |
| One table per knowledge base | Table name parameterised throughout `pgvector.py`, `UNION ALL` for multi-collection queries, schema drift |
| `INHERITS` | Trigger-based routing and constraint exclusion; declarative partitioning does both natively |
| A `DEFAULT` partition | Every new partition scans the whole default under `ACCESS EXCLUSIVE` |
| One partition per file | `CREATE TABLE … PARTITION OF` takes `ACCESS EXCLUSIVE` on the parent at every upload |
| `PARTITION BY HASH (collection_name)` | Prunes, but mixes knowledge bases and per-file rows in the same partitions, so they cannot carry different indexes |
| Partitioning by expression | Prunes only when the query filters on the same expression; `WHERE collection_name = …` would not |

## Where it is weaker

- **IVFFlat regresses with the single global `lists`.** At `probes=1` the partitioned index gives 0.73 against 0.84 (1M) and 0.59 against 0.71 (3M), because a `lists` fit for the table is far too large for a 15k-row partition. Sized per partition it gives 0.90 and 0.86 at 4–6 ms — but that setting does not exist yet.
- **The partition's graph has to be good enough.** With no filter to blame, the index's own approximation shows: `m=16` reaches 0.977 at the default effort where `m=32` reaches 1.000. On a partition made of near-duplicates, `m=16` can do slightly worse than the global graph.
- **The planner sometimes declines a partition's index** and sorts exactly: recall 1.000 at 128–130 ms on a 45k-row partition at 3M, with correct statistics in place.
- **Selecting many knowledge bases is not faster**: at `ef_search=400` the per-search cost is the same in both layouts and the partitioned parent adds about 1 ms of planning per query.
- **Migration needs downtime**: the copy is a snapshot, and `ATTACH PARTITION` cannot take a table whose key holds more than one value.
- **Three new settings**: the flag, the bucket count and the routing pattern. The last two could be constants.

## Questions

1. Is the diagnosis useful, and are the three issues worth taking on their own? The connection leak is nine lines.
2. Is `PGVECTOR_ITERATIVE_SCAN` worth having, and should it default on?
3. If partitioning is a direction you would consider: is the allow-list routing the right shape, and would you rather implement it yourselves than take the branch?

## Links

- Code, five commits against `dev`: https://github.com/runyournode/open-webui/tree/feat/pgvector-partitioning-v3 — diff: https://github.com/open-webui/open-webui/compare/dev...runyournode:open-webui:feat/pgvector-partitioning-v3
- Every measurement as tables: https://github.com/runyournode/open-webui/blob/evidence/pgvector-partitioning/docs/pgvector-partitioning/results/SUMMARY.md
- Method and limitations, the two standalone notes, raw logs and the harness: https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning
