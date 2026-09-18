# IVFFlat: a partition index built while empty, measured through PgvectorClient

Database `multikb_1000000` (partitioned, migrated), one knowledge-base
partition of 15 000 rows, `lists = 15` (pgvector's `rows/1000` for that
partition), 40 queries, a warm-up pass then the measured pass. Recall@10
against a materialised exact sort.

The only difference between the two cases is *when* the index was built:
"loaded" builds it after the rows are inserted (what the migration does),
"empty" builds it before (what `_create_dedicated_partition` does at runtime).

| probes | index built | recall@10 | worst decile | median latency | index used |
|---|---|---|---|---|---|
| **1** (pgvector's default) | loaded | **1.000** | 1.00 | 14.1 ms | yes |
| **1** | **empty** | **0.7325** | **0.50** | 14.5 ms | yes |
| 3 | loaded | 1.000 | 1.00 | 17.1 ms | yes |
| 3 | empty | 0.9675 | 0.90 | 20.6 ms | yes |
| 2 | loaded | 1.000 | 1.00 | 58.8 ms | **no** |
| 2 | empty | 1.000 | 1.00 | 56.3 ms | **no** |

Three things to take from it.

**At the default setting, an index trained on nothing returns 73 % of the
correct neighbours where a trained one returns 100 %**, and the worst decile
drops to 0.5. This is the case that matters: `ivfflat.probes` defaults to 1 and
the codebase never changes it.

**The gap closes as you probe more.** At `probes = 3` only 0.9675 against 1.000
remains: scanning a fifth of the partition is enough to make up for bad
centroids. The training deficit costs most exactly where one is trying to be
fast.

**At `probes = 2` the planner abandons the index** and sorts exactly, at 57 ms
instead of 14. Both cases then return 1.000 and training no longer matters —
there is no index in the plan. Why it declines the index at 2 probes and takes
it at 3 is not something we can explain; that row is reported as observed.

## Why this measurement replaces the earlier one

The earlier figure (0.575 against 0.888) queried an unpartitioned table
**without a collection filter**, a shape Open WebUI never issues. And on the
filtered path the client really issues, an unpartitioned IVFFlat index is mostly
not reached at all, so the comparison is not reproducible there.

The case where it is — and the one the partitioning patch creates — is the
dedicated partition: the migration builds its index after loading, the runtime
path builds it empty. Same install, same configuration, two behaviours.
