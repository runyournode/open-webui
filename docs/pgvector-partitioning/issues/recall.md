# issue: on the shared pgvector index, a knowledge-base search loses most of its neighbours — and the loss grows with the table

**Installation Method**: Docker, `ghcr.io/open-webui/open-webui:dev` (base commit e669f8aef). **Open WebUI Version**: dev. **Operating System**: Linux (WSL2). **Browser**: n/a. **Ollama Version**: n/a. **Database**: PostgreSQL 18.6, pgvector 0.8.6.

## What happens

The pgvector backend keeps every collection's chunks in one table, `document_chunk`, under one vector index, and scopes a search with `WHERE collection_name = …`. pgvector applies that predicate *after* the ANN walk: the index returns its `ef_search` nearest candidates from the whole table, and only those that happen to belong to the collection are kept. A knowledge base that holds 1.5 % of the rows owns about 1.5 % of the candidates, so most of the ten neighbours a query asks for are never examined. Nothing is logged and no error is raised; the answer is quietly incomplete.

This is a scaling problem, not a tuning one: the loss is governed by the collection's share of the table, so every knowledge base gets worse as the install grows, and the small ones worst.

## Measured

1M rows of 1024 dimensions, 20 knowledge bases of 15k rows each, the remaining 70 % in per-file, per-memory and per-web-search collections[^1]; recall@10 against an exact kNN over the same collection, 200 queries per point, through `PgvectorClient.search()`; HNSW `m=16`:

| `ef_search` | recall@10 | worst decile | SQL ms |
|---|---|---|---|
| 10 | 0.143 | 0.00 | 3.0 |
| **40 (default)** | **0.269** | **0.00** | 4.7 |
| 100 | 0.459 | 0.20 | 7.0 |
| 400 | 0.937 | 0.80 | 9.3 |

At the default setting three of every four correct neighbours are missing, and one query in ten returns nothing relevant. At 3M rows the default gives 0.284. IVFFlat, the default index method, gives 0.835 at its default `probes = 1`; at any higher `probes` the planner drops the index for an exact sort — correct, at 46 ms (1M) and 135 ms (3M) per search.

Raising `ef_search` widens the candidate list and helps slowly at the cost shown; it does not change the mechanism.

## What pgvector offers, and Open WebUI does not use

pgvector 0.8 added iterative scans for exactly this case: after filtering, the scan keeps going until it has the requested number of rows. `hnsw.iterative_scan` takes `off | relaxed_order | strict_order`, `ivfflat.iterative_scan` takes `off | relaxed_order`, both default to `off`, and the walk is bounded by `hnsw.max_scan_tuples` (20000) and `ivfflat.max_probes` (32768). The codebase never sets them. With `hnsw.iterative_scan = relaxed_order` the default point goes from 0.269 to 0.914, and the SQL time of a search from 4.7 ms to 8.7 ms.

## What that leaves: one table that does not scale

Iterative scans repair the recall, not the shape that causes it. About 70 % of the rows[^1] belong to collections that are looked up by exact name and sorted over a handful of rows — they never use the vector index, yet they are in it. Every knowledge-base search walks a graph made mostly of other collections' rows; the index grows with the whole install rather than with the knowledge bases; and vacuum walks all of it. At 3M rows the HNSW index is 10.9 GB where the knowledge bases alone need 6.1 GB, the IVFFlat index is 22.9 GB against a 16.2 GB heap, and a vacuum that has to clean the index after a knowledge base is deleted takes 59.5 s.

Giving each knowledge base its own partition and index, on the same 1M dataset at the default `ef_search`: recall@10 **0.977** at **3.06 ms** — against 0.269 at 4.67 ms as shipped and 0.914 at 8.72 ms with iterative scans — with the vector index at 2103 MB instead of 4186, the vacuum at 0.8 s instead of 59.5 s, and the bucketed 70 % unchanged at recall 1.000. Design, alternatives considered and the parts that are weaker are in the Discussion: <DISCUSSION_URL>.

## Reproduce

`tests/vector/bench_pgvector_partitioning.py` on https://github.com/runyournode/open-webui/tree/feat/pgvector-partitioning-v3: `generate --rows 1000000`, `index --method hnsw`, `measure --config A --target kb --ef-search 40`. Raw logs and every measurement as tables: https://github.com/runyournode/open-webui/tree/evidence/pgvector-partitioning/docs/pgvector-partitioning. Related to #17998 and #20737, which are about the Python BM25 fallback rather than the vector index.

[^1]: An assumption, and a conservative one. When a file is added to a knowledge base with stored content, `process_file` writes its chunks to the knowledge-base collection *and* to the file's own `file-{id}` collection (`collection_names.append(file_collection_name)`), so knowledge-base content alone puts at least half of the rows in bucketed collections; files attached to chats without a knowledge base, web searches (`web-search-*`) and memories (`user-memory-*`) only add to that side.
