# IVFFlat indexes are built before there is anything to train on

**Status: pre-existing upstream behaviour, out of scope for the partitioning
PR, proposed as a separate change.**

This is written up separately so the partitioning PR can mention it without
carrying it. Nothing here is introduced by that patch — the behaviour is in
`dev` today — but the patch does add an inconsistency that makes it worth
fixing, and it interacts with a second issue that only appears once the table is
partitioned.

---

## 1. The problem

pgvector builds an IVFFlat index by running k-means over the table's rows to
place list centroids. Its [documentation](https://github.com/pgvector/pgvector#ivfflat) is explicit:

> Create the index **after** the table has some data.

and, contrasting with the other method:

> HNSW … can be created without any data in the table since there isn't a
> training step like IVFFlat.

An IVFFlat index built on an empty table therefore has no training data. It does
not error — it is created, queries keep working, and nothing in the logs says
anything is wrong.

## 2. Upstream builds it on an empty table

`PgvectorClient.__init__` (`backend/open_webui/retrieval/vector/dbs/pgvector.py`)
does, in order:

```python
Base.metadata.create_all(bind=connection)          # creates document_chunk

index_method, index_options = self._vector_index_configuration()
self._ensure_vector_index(index_method, index_options)
self._ensure_text_search_index()
```

On a **fresh install** `create_all` creates an empty `document_chunk` and
`_ensure_vector_index` immediately indexes it — with zero rows. The index is
never rebuilt afterwards: `_ensure_vector_index` returns early whenever an index
of that name already exists.

And this is the **default path**. `vector_index_configuration()` returns
`ivfflat` unless `PGVECTOR_INDEX_METHOD` is set explicitly or
`PGVECTOR_USE_HALFVEC` is on:

```python
if PGVECTOR_INDEX_METHOD:
    index_method = PGVECTOR_INDEX_METHOD
elif USE_HALFVEC:
    index_method = 'hnsw'
else:
    index_method = 'ivfflat'
```

An install that upgraded into the version which introduced the index is fine —
the table already had rows when the index was first created. A fresh install is
not.

Worth stating plainly: this was invisible because the backend has no test that
measures retrieval quality. The functional suite runs `ivfflat` throughout (it
never sets `PGVECTOR_INDEX_METHOD`, and its 8-dimensional vectors do not trigger
the halfvec branch), and every one of those tests passes against an index trained
on nothing, because they assert that results come back, not that they are the
right ones.

## 3. What the partitioning patch changes

It does not introduce the problem; it reproduces it per partition, and adds an
inconsistency:

| Path | When the index is built | Trained? |
|---|---|---|
| Migration (`build_partition_indexes`) | after the partition is loaded | yes |
| Runtime (`_create_dedicated_partition`) | at partition creation, empty | no |

So a migrated install ends up with well-trained indexes for the knowledge bases
it already had, and untrained ones for every knowledge base created afterwards.
Same install, same configuration, two behaviours — which is worse than being
uniformly wrong, because it is harder to notice.

## 4. Measured cost

Measured on exactly that case, through `PgvectorClient.search()`: a knowledge
base partition of 15 000 rows, `lists = 15` (pgvector's `rows/1000` for that
partition), 40 queries, warm cache, recall@10 against a materialised exact sort.
The only difference between the two cases is *when* the index was built.

| `probes` | index built | recall@10 | worst decile | median latency | index used? |
|---|---|---|---|---|---|
| **1** (pgvector's default) | after loading | **1.000** | 1.00 | 14.1 ms | yes |
| **1** | **empty** | **0.733** | **0.50** | 14.5 ms | yes |
| 3 | after loading | 1.000 | 1.00 | 17.1 ms | yes |
| 3 | empty | 0.968 | 0.90 | 20.6 ms | yes |
| 2 | after loading | 1.000 | 1.00 | 58.8 ms | **no** |
| 2 | empty | 1.000 | 1.00 | 56.3 ms | **no** |

**At the default setting an index trained on nothing returns 73 % of the correct
neighbours where a trained one returns 100 %**, and the worst tenth of queries
returns half. Silently, with no error and nothing in the logs. `ivfflat.probes`
defaults to 1 and the codebase never changes it.

Two things temper that, and both belong in the report:

**The gap closes as you probe more.** At `probes = 3` the untrained index is at
0.968 against 1.000 — scanning a fifth of the partition recovers most of what
bad centroids lost. The training deficit costs most precisely where you were
trying to be fast.

**At `probes = 2` the planner abandons the index** and sorts exactly instead, at
57 ms rather than 14. Both cases then return 1.000 and training is irrelevant,
because there is no index in the plan. Why the planner declines the index at 2
probes and takes it at 3 is not something we can explain — a higher `probes`
should make the index scan dearer, not cheaper — so that row is reported as
observed rather than interpreted.

### Why this supersedes an earlier measurement

An earlier round reported 0.575 against 0.888 for the same comparison. That
measurement queried an unpartitioned table **without a collection filter**, which
is not a shape Open WebUI ever issues — and on the filtered shape the client
actually issues, an unpartitioned IVFFlat index is mostly not reached at all, so
the comparison is not reproducible there.

The case where it *is* reproducible is the dedicated partition, which is what
the table above measures. That also makes it the case that matters: on an
unpartitioned table an untrained index is mostly *dormant* — it costs disk and
vacuum time rather than accuracy, because the queries Open WebUI issues per
collection largely do not reach it. It becomes a correctness problem exactly when
a query does reach it, which is what partitioning changes.

## 5. A second issue that only partitioning exposes

`PGVECTOR_IVFFLAT_LISTS` is a single global value (default 100). pgvector sizes
`lists` from the row count:

> a good place to start is `rows / 1000` for up to 1M rows and `sqrt(rows)` for
> over 1M rows

On one shared table there is one row count, so one value can be right. Once the
table is partitioned there are many: a 1M-row table wants ~1000 lists, while each
15k-row knowledge-base partition wants ~15. Applying the table-level value to a
partition over-partitions it by nearly two orders of magnitude.

A single global setting cannot express both. But sizing `lists` per partition is
not a clean win either, and the benchmark says so: with `lists` sized from each
partition's rows, the partitions become small enough that the planner prefers an
exact sort again — recall 1.000, but at 43 ms (1M) and 129 ms (3M) rather than
3 ms. IVFFlat on a partitioned table wants a `lists` value chosen per partition
*and* large enough to stay worth scanning, and that is a tuning question this
work has not answered.

### `lists` also costs build memory

Sizing `lists` from a table-level setting rather than the partition's row count
is not only a recall question. pgvector samples roughly 50 × `lists` vectors to
train, so an oversized `lists` inflates the memory the build demands — and the
demand follows `lists` rather than the table:

> `ERROR: memory required is 403 MB, maintenance_work_mem is 256 MB`

at 1M rows with `lists = 1000`, which is pgvector's own guidance for that size.
Since the index is created inside the client's `__init__`, it surfaces as a
failed startup. A partition that inherits a table-level value can therefore fail
to index at all on a modestly configured server.

## 6. Proposed fix (for a separate PR)

**Defer the vector index until the partition has something to train on.**

```
CREATE TABLE … PARTITION OF …        -- no vector index yet
   … rows are inserted …
when the partition reaches N rows:
   CREATE INDEX … USING ivfflat (vector …) WITH (lists = <sized from N>)
```

- Below the threshold there is no vector index, so a search over that partition
  is an exact scan of a small table: recall 1.0, and fast because the partition
  is small. Correct by construction rather than approximately correct.
- Above it, the index is built over real data — trained, and with `lists` sized
  from the count that actually exists.
- The runtime and migration paths converge on one behaviour.
- It helps HNSW too: one bulk build over the loaded rows is cheaper than
  maintaining the graph across every insert, even though HNSW does not need the
  training data.

Implementation notes:

- The row check should be bounded, not a `count(*)` over a growing partition:
  `SELECT count(*) FROM (SELECT 1 FROM <partition> LIMIT :n) s` stops at the
  threshold. It only needs to run for partitions not yet known to be indexed,
  and the existing per-process partition cache already tracks that.
- New setting, e.g. `PGVECTOR_PARTITION_INDEX_MIN_ROWS`, defaulting to something
  in the low thousands.
- `lists` sized per partition, clamped to a range wide enough that the planner
  still finds the index worth using — see §5, which is the open question.

**Why this is not in the partitioning PR.** It changes index-creation semantics
for every pgvector install, not just partitioned ones, and it is the kind of
change that deserves its own review and its own before/after numbers. Bundling it
would mean the partitioning PR also had to defend a change to how upstream has
always built indexes. The partitioning PR keeps eager creation — matching
upstream exactly — and points here.

**A narrower fix**, if the above is judged too broad: leave HNSW eager (pgvector
explicitly supports it) and defer only IVFFlat. That targets the real problem at
the cost of two code paths to maintain and test.
