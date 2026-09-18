# What was measured, and how

Companion to the `document_chunk` partitioning patch. This document exists so a
reviewer can tell what the numbers describe without reading the harness — and,
just as importantly, what they do **not** describe.

Harness: `tests/vector/bench_pgvector_partitioning.py`.
Every figure in the discussion was produced by it, and every result it emits
carries the configuration that produced it.

---

## 1. Everything is measured through Open WebUI's own client

Recall and latency come from `PgvectorClient.search(collection_name, vectors,
limit)` — the same entry point `query_collection` calls — so the measurements
exercise `adjust_vector_length`, the LATERAL join over a `VALUES` list, the
distance normalisation, `_collection_scope()` and `_apply_iterative_scan()`.

This matters, and it is a change from an earlier round. That round issued
hand-written SQL that imitated `search()`, and the imitation was a
simplification: a single-vector `ORDER BY vector <=> …` where the real statement
builds a lateral join. The planner can legitimately choose differently between
the two, and it did — see §9.

The exact baseline is still raw SQL, because its whole purpose is to avoid the
index the client would use.

**Two latencies are reported, not one.** `search()` binds the query vector as a
thousand parameters, and on a small collection that Python-side cost is most of
what a caller waits for — it would flatten exactly the differences this exists
to show. So each result carries the wall time of the full client call *and* the
cursor execution alone, captured from a SQLAlchemy `before/after_cursor_execute`
pair. The comparisons in the discussion use the SQL time; the client time is
there so nobody mistakes one for the other.

## 2. What is under test

`document_chunk` in two layouts, with identical data in both:

- **Unpartitioned** — one table, one global vector index, one GIN index, one
  btree on `collection_name`. This is Open WebUI as shipped.
- **Partitioned** — `PARTITION BY LIST (part_key)`. Each knowledge base gets a
  partition with its own vector and GIN indexes. Every other collection is
  hashed into one of 32 buckets carrying only a btree on `collection_name` —
  **no vector index, no full-text index**.

The second layout is only reachable through the migration script, so every
partitioned measurement is taken on a table that was migrated, not on one built
partitioned from scratch. That is the path an existing install would take.

## 3. Environment

| | |
|---|---|
| Runner image | `ghcr.io/open-webui/open-webui:dev` (the project's nightly pre-release build) |
| Code under test | the feature branch, mounted over the image, so only dependencies come from it |
| Database | `pgvector/pgvector:0.8.6-pg18-trixie` — PostgreSQL 18.6, pgvector 0.8.6 |
| Host | WSL2, 31 GB RAM, ~1 TB disk |

Since `:dev` is rebuilt nightly, each result records the image digest and the
installed versions, and a mismatch against the branch's pins fails the run. The
five packages this backend depends on — sqlalchemy 2.0.50, psycopg2-binary
2.9.12, psycopg 3.3.4, pgvector 0.4.2, alembic 1.18.4 — are recorded with every
measurement.

**PostgreSQL memory is a variable, not a constant.** Two profiles:

| Profile | `shared_buffers` | `maintenance_work_mem` | `work_mem` |
|---|---|---|---|
| generous | 2 GB | 24 GB | 256 MB |
| modest | 512 MB | 256 MB | 16 MB |

`maintenance_work_mem` is the setting that matters most here: an HNSW graph needs
roughly `rows × (dim×4 + m×2×4)` bytes, and a build that exceeds it spills to
disk and collapses in throughput. At 1024 dimensions that is ~4 GB per million
rows. The modest profile is chosen so the *global* index cannot fit while each
*partition's* index can — that contrast is the point of running it.

Measured at 200k rows, where the m=16 graph is ~0.85 GB against 256 MB: the
global HNSW build took **84 s under the modest profile against 29 s under the
generous one** (m=16), and 217 s against 98 s (m=32). pgvector reports the
spill as a NOTICE, but a parallel build does not surface it to the client, so
the build time is the evidence here rather than the notice.

Index builds additionally pass a per-campaign `maintenance_work_mem` sized to the
data (1–16 GB): pgvector reserves the whole amount as shared memory for the
duration, so giving a 1M-row build what a 3M-row build needs simply takes memory
out of circulation. `/dev/shm` is sized to match, or `CREATE INDEX` fails
outright.

`max_locks_per_transaction` is deliberately left at PostgreSQL's default of 64,
so lock pressure is measured as an untuned install would experience it.

## 4. The dataset

Generated, not real. It reproduces the collection shapes Open WebUI actually
creates, in the proportions a populated install tends to have:

| Shape | Chunks per collection | Share of rows | Layout after migration |
|---|---|---|---|
| knowledge base (bare UUID) | rows×0.3/20 collections | 30 % | its own partition + indexes |
| `file-{id}` | 20 | ~56 % | bucket, btree only |
| `user-memory-{user}` | 5 | ~7 % | bucket, btree only |
| `web-search-{user}-{hash}` | 10 | ~7 % | bucket, btree only |

So **~70 % of rows are bucketed** and lose their vector index under
partitioning. Each knowledge base is ~1.5 % of the table.

**Vectors are synthetic but clustered.** A pool of 500 random centroids and 5000
random perturbations; each row is one centroid plus one perturbation. Uniform
random vectors in 1024 dimensions are all roughly equidistant, which makes
recall@k meaningless — every candidate looks equally good and any index scores
well.

**Cluster choice is hashed, independent of collection.** Deriving both the
cluster and the collection from `mod()` on the same counter would make them share
factors, handing each collection a disjoint set of clusters and leaving it
trivially separable in vector space — which flatters the unpartitioned baseline,
whose whole difficulty is that a collection is scattered through a shared graph.

**Generation verifies itself.** After loading, the harness checks actual row and
collection counts per shape against a plan computed before any insert, checks
that every id is distinct, and aborts on any mismatch. A dataset that does not
match its plan cannot reach a measurement.

## 5. Real embeddings, and a haystack that is actually hard

199 passages are embedded through a local model (`text-embedding-harrier-oss-v1-0.6b`,
1024 dimensions) and written over rows of one knowledge base — *replacing* rows
rather than adding them, so the composition still matches the verified plan. 48
of them have a question written to retrieve them; the questions are embedded at
the same time and stored, so a measurement campaign never has to reach the
embedding endpoint and every campaign queries with byte-identical vectors.

Left there, the test would be trivial. The synthetic vectors sit almost
orthogonal to real embeddings — around 0.94 cosine distance from a question,
where the right passage is at 0.35 — so the index would only have to separate
two distributions. So a share of that collection's remaining synthetic rows is
rewritten as *a real passage plus scaled noise*, at six noise levels, producing
neighbours spread from just beside a passage to well away from it: 2 801 such
rows at 200k, 14 801 at 1M, and 20 000 at 3M, where a cap keeps the rewrite
bounded.

**Counted within the first ten.** pgvector's HNSW scan returns at most
`ef_search` rows unless iterative scans are on. At `ef_search=10` the
unpartitioned and partitioned layouts therefore return ten rows while
configuration B returns fifty, and a "reachable within 50 but not returned"
count compares them unequally — B looks better for a mechanical reason. The
comparable figure is the number of questions whose passage is within the first
ten returned (`index_top10`), and it is the one the write-ups quote. The harness
now records when a measurement's requested depth exceeds `ef_search` without
iterative scans.

**What this test measures, and what it cannot.** Every real passage and every
near-duplicate built from one live in the same knowledge base, so near a real
question the collection filter has nothing to discard. The needle test therefore
does not measure the post-filter loss; it measures the index's own accuracy
inside a dense region of near-duplicates. The evidence is in the results: at 1M,
HNSW `m=16`, `ef_search=10`, the partitioned layout misses 33 answers and the
unpartitioned one 31. The filter's share is what remains once the graph is
adequate — at `m=32`, 7 missed as shipped against 0 partitioned at 1M. At 200k,
with 2 801 near-duplicates instead of 14 801, nothing misses anything in any
configuration.

Some of those near-duplicates necessarily land nearer a question than its own
passage does. That is why **two ranks are reported per question**: where an exact sort puts
the right passage, which is a property of the embedding model, and where the
index puts it. Only the difference is attributable to the index, and the figure
that carries is *reachable but missed* — the passage was in the exact top-50 and
the index did not return it.

## 6. What is measured, and on which collections

Recall and latency are measured **per collection**, by running a vector
similarity search scoped to one collection — the same shape the application
issues. Results are reported for two disjoint targets, **separately and never
pooled**:

- **`--target kb`** — knowledge-base collections. Under partitioning these have
  a partition and a vector index of their own. This is the case partitioning is
  designed to improve.
- **`--target buckets`** — `file-*`, `user-memory-*` and `web-search-*`,
  sampled across all three shapes. Under partitioning these have **no vector
  index at all**. This is the case that must not regress, and it is the majority
  of the table.

Measuring only the first would describe the best case for partitioning. Pooling
the two would be worse still: they have different index structures, so a large
gain on 30 % of rows could mask a loss on the other 70 %.

A knowledge-base measurement is 10 collections × 20 query vectors = 200
queries. A bucket measurement samples three collections of each bucketed shape
and cannot take 20 queries from a collection that holds 5 or 10 rows, so it is
9 collections and **105 queries** (3 × 20 file chunks, 3 × 5 memory chunks,
3 × 10 web-search chunks). Query vectors are drawn from the collection being
searched, so they sit inside its cluster structure rather than in empty space —
with a consequence stated in §13.

## 7. The multi-knowledge-base case

Measured through `query_collection` itself, not through an imitation of it,
because the interesting part is exactly what that function does: **it does not
aggregate**. It fans out over the cartesian product of queries × collections
with `asyncio.gather` + `asyncio.to_thread`, so N collections mean N independent
per-collection searches, each asking for the full `k` rather than `k/N`, merged
afterwards by content hash and re-sorted. N = 1, 3, 10.

Ground truth for the merged answer is the exact top-k of the union of the N
collections, which is what a user selecting N knowledge bases is asking for.
The query vectors are rows of the first selected collection, with that row
excluded from both the exact answer and the merged one.

**The campaign multi-KB runs drew their queries from the wrong collection**,
and their recall column is not informative. They queried from the first
knowledge base, which is the one holding the real embeddings; those form a
tight cluster nearly orthogonal to the synthetic bulk, so any index finds them,
and every layout reported 0.92–1.0 at every effort — which the per-collection
measurement on the same data flatly contradicts. Their latency figures stand,
since the cost of the fan-out does not depend on which collection is queried.
The harness now skips the needle collection; the figures quoted for recall come
from a re-run at 200k rows (`d6-multikb-default-effort-200000`), at
`ef_search` 40 and 400, 20 queries per point.

**Why the partitioned fan-out is slower at high effort.** The 1M and 3M
multi-KB re-runs were taken at `ef_search=400`. At that effort the per-search
execution costs the same in both layouts — 15–18 ms at 3M with `m=32`, measured
with `EXPLAIN ANALYZE` — because the walk no longer stops early, which is where
partitioning saves its work; and planning a query against the 53-partition
parent costs 1.0–1.6 ms against 0.6–0.7 ms for the same query on the partition
alone. The fan-out multiplies both by N. At the default effort (the 200k run)
wall times are the same in every layout.

**Why the unpartitioned figure climbs with N.** Open WebUI sends one query per
knowledge base and PostgreSQL applies the collection filter after the HNSW walk
in each of them; the individual searches are the same whatever N is. What
changes is the *question*: merged recall is measured against the exact top-k of
the union of the N collections. Each filtered search returns the near rows that
belong to its collection and misses the far ones, because the walk only visits
the ~40–70 nearest rows overall. A single collection is 1.5 % of the table, so
its ten true neighbours sit at global ranks of roughly 40, 110, 200 … 660 —
mostly far, mostly missed. A ten-collection union is 15 % of the table, so its
ten true neighbours sit at global ranks of roughly 5, 12, 19 … 67 — all near,
each returned by whichever of the ten searches owns it, and the merge in Python
puts them back together. No search improved; the far rows simply stopped being
part of the answer. Selecting many knowledge bases pushes the query towards a
global search, which HNSW does well — and the case that suffers is the common
one, a single knowledge base.

## 8. Ground truth for recall

recall@10 = |approximate top-10 ∩ exact top-10| / 10, averaged over all queries,
with the worst decile reported alongside the mean.

The exact baseline materialises the collection's rows in a CTE first, then sorts:

```sql
WITH candidates AS MATERIALIZED (
  SELECT id, vector FROM document_chunk WHERE collection_name = :c AND part_key = :p
)
SELECT id FROM candidates ORDER BY vector <=> :q LIMIT 10
```

**`SET enable_indexscan = off` is not sufficient on PostgreSQL 18**, and this
matters to anyone reproducing these numbers. A disabled node is discouraged, not
forbidden: the planner still chooses the vector index when nothing else can
satisfy the `ORDER BY`, and marks it `Disabled: true` in the plan. Measured that
way, "exact" recall comes from the very index under test at whatever search
effort is in force — an approximation compared against itself, which shows up as
recall appearing to *fall* as `ef_search` rises. Materialising first puts the
sort on a result with no index on it, so it has to be exact; the plan is
`Seq Scan → Sort`.

## 9. Which access path was chosen

Recall and latency alone cannot tell an accurate index from one the planner
declined to use: both can show recall 1.0. So every measurement carries the
plan of the statement the client actually emitted, captured from the cursor
event and explained on the raw driver cursor at the same search effort.

This is what corrected an earlier claim. Measured with hand-written SQL, the
IVFFlat index appeared never to be used; measured through the client it *is*
used at low `probes`, and the planner switches to an exact sort as `probes`
rises. The conclusion changed shape — the index is bypassed at every setting
that reaches good recall, rather than always — and only the plan capture
revealed it.

## 10. Configurations

Three, measured on identical data, **for each index method**:

| | |
|---|---|
| **A** | Unpartitioned, as shipped. `PGVECTOR_ITERATIVE_SCAN` off. |
| **B** | Unpartitioned, with iterative index scans enabled (`PGVECTOR_ITERATIVE_SCAN=relaxed_order`; off by default in the branch). |
| **C** | Partitioned, iterative scans off, so A→C isolates partitioning. |

**B exists to keep the comparison honest.** Most of the post-filter recall loss
can also be reduced by a GUC the codebase never sets, so reporting only A against
C would credit partitioning with a win a one-line configuration change also
delivers. The GUC is method-specific — `hnsw.iterative_scan` or
`ivfflat.iterative_scan` — and setting only one of them would make B identical
to A under the other method. It is applied by the client, not by the harness,
because the client is what decides it.

**Both index methods are measured.** `ivfflat` is Open WebUI's default
(`vector_index_configuration()` returns it unless `PGVECTOR_INDEX_METHOD` is set
or `PGVECTOR_USE_HALFVEC` is on); `hnsw` is what quality-sensitive installs
choose.

**Build parameters and search effort are both swept**, because a conclusion that
holds at one point in the tuning range is not a conclusion:

| Method | Builds | Search effort |
|---|---|---|
| HNSW | `(m=16, ef_construction=64)`, `(m=32, ef_construction=128)` | `ef_search` 10 / 40 / 100 / 400 |
| IVFFlat | `lists = rows/1000`, `lists = sqrt(rows)` | `probes` 1 / 10 / √lists / lists÷4 |

Partitioned IVFFlat is additionally measured with `lists` sized from each
partition's own row count, which one global setting cannot express.

### Cache state

Every configuration is measured **after a warm-up pass** over the same queries.
Without it the configuration measured first pays for every page the others then
find resident, and latency ends up ranking the execution order rather than the
configurations — which matters as soon as the working set outgrows RAM. This
applies to the multi-knowledge-base measurement too; see §13.

## 11. Metrics

| Metric | Definition |
|---|---|
| recall@10 | against the exact baseline above; mean and worst decile |
| needle rank | position of a known real passage, exact and approximate, over 48 questions |
| SQL time | cursor execution alone, median and p95 |
| client time | wall time of the full `search()` call, median and p95 |
| index size | `pg_relation_size` summed per index kind, across all partitions |
| index build time | wall time for `CREATE INDEX`, at a stated `maintenance_work_mem` |
| `delete_collection` | wall time of the real API call, plus the vacuum that follows |
| vacuum | also measured after a raw `DELETE`, the pessimistic case |

The churn and vacuum steps run once per campaign, after the last index build,
so **they were measured with the IVFFlat index in place** (`lists = 1732` at
3M), not the HNSW one. The 59.5 s figure quoted in the discussion is an
IVFFlat vacuum.

| migration | wall time end to end, including building every partition index |

Two vacuum figures are reported on purpose. A raw `DELETE` followed by `VACUUM`
is not what the application does — it calls `delete_collection()`, which is a
`DELETE` in one layout and a `DROP` of the collection's partition in the other.
Reporting only the favourable one would be a choice, so both are given.

Both are manual `VACUUM` statements on the table, which is not how vacuum
happens in production: autovacuum works per relation, so under partitioning it
vacuums each partition on its own schedule and never recurses through the
parent. The 0.7 s against 0.1 s "plain vacuum" figure at 3M is therefore the
cost of a manual whole-table vacuum that has nothing to clean, not something an
install pays in normal operation; the figures that matter are the ones where
index cleanup actually runs (13.3 s → 0.2 s after a raw delete, 59.5 s → 0.8 s
after `delete_collection()`).

## 12. Operational discipline

- The stack is torn down and the database recreated before each campaign, so no
  process from a previous one can still be writing.
- **Before every campaign** the functional suite is run and the PostgreSQL log
  must contain no unexpected `ERROR` or `FATAL`. A campaign that would start on
  a dirty log does not start. Two errors are allowlisted because two tests raise
  them deliberately (a write to a collection whose partition does not exist yet,
  and a constraint violation that must not be mistaken for a missing partition).
- Campaigns run **strictly one at a time**. Nothing else touches the database
  while one runs.
- Generation is verified before any measurement runs.
- **A step that is still progressing is never killed.** Progress is tracked
  through `pg_stat_progress_create_index` and the migration cursor; a run is
  stopped only if it is genuinely stuck, and that is reported rather than the
  run being quietly dropped. An abandoned index build yields no number at all,
  which is worse than a slow one.

## 13. Limitations — what these numbers do not cover

Stated plainly.

- **Query vectors are stored rows, and the published figures include the
  self-match.** Each query is a row of the collection searched, so its own row
  is at distance zero and is one of the ten exact neighbours — a hit a filtered
  index scan finds too. Measured at 200k rows (HNSW m=16, `ef_search=40`) with
  that row excluded from both the exact answer and the index's, the
  unpartitioned figure goes from 0.294 to **0.206**, iterative scans from
  0.935 to 0.915, and the partitioned layout from 0.996 to 0.994. The
  self-match was worth about one hit in ten to the layout that misses most and
  nothing to the one that does not, so the published tables understate the gap
  rather than overstate it. A query offset by a small random amount changes
  nothing (0.294): at 0.04 cosine distance the row is still found. The harness
  now excludes the query's own row by default; the campaigns were run before
  that and are reported as measured.
- **Synthetic vectors, except where stated.** The bulk of the data is clustered
  and decorrelated, but it is not embeddings. Real embeddings have structure
  these do not, and absolute recall figures would differ. The *comparison*
  between layouts is what carries over. §5 is the exception, and it is
  deliberately the case where absolute numbers mean something.
- **Scale ceiling: 3M rows.** 5M was attempted in an earlier round, at
  `maintenance_work_mem = 4 GB` on a 23 GB host, and abandoned: the global
  HNSW graph (~20 GB) was 54 % built after 67 minutes and still in the
  disk-spill phase. It was not retried at the 24 GB used here, where a 5M graph
  would fit only barely. So there is no A/B/C comparison above 3M, and the raw
  log of that attempt did not survive a host restart; the figures come from the
  notes taken at the time.
- **Single host, single run.** No repetitions, so no dispersion is reported.
  Medians and p95 come from the queries within one run, not across runs.
- **The multi-knowledge-base figures come from a separately generated dataset.**
  The migration is done in place, so once a campaign has migrated, its
  unpartitioned table no longer exists and configuration A cannot be measured
  again. The multi-KB numbers at 1M and 3M were therefore taken on a fresh
  dataset — same generator, same parameters, different random vectors — with A,
  B and C all measured on it so the comparison stays internal to one dataset.
- **Multi-KB points are 10 queries each in the campaigns and 20 in the 200k
  re-run**, against 200 for the per-collection measurements. Treat them as
  indicative and the per-collection ones as the result. The campaign multi-KB
  recall figures are superseded for the reason given in §7.
- **The first multi-KB series is superseded and not reported.** It had no
  warm-up pass, and since the partitioned layout is necessarily measured last,
  it paid for every page the earlier layouts left resident — 193 ms against
  20 ms for the same measurement once warmed. The logs are kept, marked
  `-sans-prechauffage`.
- **An upstream connection leak invalidated four multi-KB points in the first
  full series**, since fixed and re-measured. `search`, `query` and `get`
  returned early on an empty result set without rolling back, leaking one pooled
  connection per thread; the affected measurements recorded the 30-second
  checkout timeout instead of a latency. Only the concurrent path was affected —
  the per-collection measurements reuse one session and were never at risk.
- **`hybrid_search` is not benchmarked.** The functional tests assert that the
  decision between the native path and Open WebUI's Python `BM25Retriever`
  fallback is unchanged by partitioning, but no timings were taken.
- **No concurrency outside the multi-KB measurement.** Every other measurement
  is single-client. Lock pressure is counted analytically (relation locks per
  query) rather than measured under load.
- **pgcrypto is tested but not benchmarked.** Both layouts are covered for
  insert, upsert, get, search, query, delete, delete_collection and
  has_collection, and for hybrid search still declining as upstream does. No
  timings were taken under it: encryption dominates, and the comparison between
  layouts would measure `pgp_sym_decrypt` rather than the index.
- **The IVFFlat empty-index issue is out of scope for this patch** and is
  documented separately. The matrix here is unaffected by it: A and B build the
  global index after loading, and C's indexes come from the migration, which
  also builds after loading, so all three are trained.
