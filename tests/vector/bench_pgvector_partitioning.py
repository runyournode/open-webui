"""Measure what partitioning `document_chunk` actually buys.

Reports recall@k against exact kNN, query latency, index size, index build time
and vacuum duration for three configurations:

    A  unpartitioned, as shipped today
    B  unpartitioned, with iterative index scans enabled
    C  partitioned

B matters. Most of the post-filter recall loss can also be reduced by a GUC the
codebase never sets, so comparing only A against C would credit partitioning
with a win that a configuration change also delivers. What B cannot deliver is
the smaller index, the cheaper vacuum, and keeping the bucketed rows out of the
graph altogether -- and that is the case this measures.

**Both collection families are measured, separately and never pooled.**
Knowledge bases get a partition and an index of their own under C; everything
else -- per-file, per-memory, per-web-search collections -- is hashed into
buckets that carry only a btree. Those bucketed rows are the majority of a real
table and they are the ones that *lose* their vector index, so measuring only
knowledge bases would report the best case and call it the result.

Every recall and latency figure comes from `PgvectorClient.search()` -- the
same entry point `query_collection` calls -- rather than from SQL written here
to resemble it. An earlier revision did the latter, and its simplification (a
single-vector ORDER BY where the real thing builds a LATERAL join over a VALUES
list) left the planner free to choose differently from the code being reported
on. The exact baseline is still raw SQL, because its whole purpose is to avoid
the index.

Usage (inside the container, against the throwaway database):

    python -m tests.vector.bench_pgvector_partitioning generate --rows 1000000
    python -m tests.vector.bench_pgvector_partitioning embed      # real vectors
    python -m tests.vector.bench_pgvector_partitioning index --method hnsw
    python -m tests.vector.bench_pgvector_partitioning measure --config A --target kb
    python -m tests.vector.bench_pgvector_partitioning measure --config A --target buckets
    python -m tests.vector.bench_pgvector_partitioning needle --config A
    python -m tests.vector.bench_pgvector_partitioning multikb --config A
    python -m tests.vector.bench_pgvector_partitioning churn
    python -m tests.vector.bench_pgvector_partitioning vacuum

`partindex` rebuilds the per-partition indexes with different build parameters,
so the tuning range can be swept without migrating again for every point.

Vectors are synthetic but clustered (a pool of centroids plus a pool of
perturbations). Uniform random vectors in 1024 dimensions are all roughly
equidistant, which makes recall@k meaningless -- every candidate looks equally
good and any index scores well.

Roughly two hundred of them are not synthetic. `embed` replaces the contents of
that many rows in one knowledge base with passages embedded through a real
model, and stores a question for each alongside its embedding. `needle` then
asks those questions and reports where the right passage landed, which is the
case a clustered synthetic distribution cannot speak for: a genuine answer
buried among unrelated neighbours.

Cluster choice is hashed rather than taken from the row number, so it stays
independent of which collection a row belongs to. Deriving both from `mod()` on
the same counter makes them share factors, which hands each collection a
disjoint set of clusters and leaves it trivially separable in vector space --
flattering the unpartitioned baseline, whose whole problem is that a collection
is scattered through a shared graph.

Every result carries the configuration that produced it -- index method and its
parameters, the PostgreSQL memory settings, the verified dataset composition and
the dependency versions. A measurement that does not state its index method
cannot be compared with another.
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

from sqlalchemy import create_engine, text

TABLE = 'document_chunk'
DEFAULT_K = 10

# The collection shapes Open WebUI actually creates. Only knowledge bases earn a
# partition of their own; the rest are bucketed, and they are the majority.
KB_SUFFIX = '-0000-4000-8000-000000000000'
BUCKETED_PREFIXES = ('file-', 'user-memory-', 'web-search-')
CHUNKS_PER_FILE = 20
CHUNKS_PER_MEMORY = 5
CHUNKS_PER_SEARCH = 10


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------


def _plan(rows: int, knowledge_bases: int, kb_share: float, memory_share: float, search_share: float) -> Dict:
    """Exact row and collection counts, decided before a single row is written.

    Every shape's row count is an exact multiple of its chunks-per-collection
    except `file-*`, which absorbs the remainder so the total lands precisely on
    `rows`. `verify_dataset` then checks the database against this plan, so a
    generation that silently produced something else cannot be published as if
    it had not.
    """
    if not 0 < kb_share < 1:
        raise ValueError(f'kb_share must be between 0 and 1, got {kb_share}')
    if memory_share + search_share >= 1:
        raise ValueError('memory_share + search_share must leave room for file collections')

    per_kb = max(1, int(rows * kb_share) // knowledge_bases)
    kb_rows = knowledge_bases * per_kb
    bucket_rows = rows - kb_rows
    if bucket_rows <= 0:
        raise ValueError(f'kb_share {kb_share} leaves no rows for bucketed collections')

    memory_rows = (int(bucket_rows * memory_share) // CHUNKS_PER_MEMORY) * CHUNKS_PER_MEMORY
    search_rows = (int(bucket_rows * search_share) // CHUNKS_PER_SEARCH) * CHUNKS_PER_SEARCH
    file_rows = bucket_rows - memory_rows - search_rows

    return {
        'rows': rows,
        'kb': {'rows': kb_rows, 'collections': knowledge_bases, 'per_collection': per_kb},
        'file': {'rows': file_rows, 'collections': -(-file_rows // CHUNKS_PER_FILE), 'chunks': CHUNKS_PER_FILE},
        'memory': {'rows': memory_rows, 'collections': memory_rows // CHUNKS_PER_MEMORY, 'chunks': CHUNKS_PER_MEMORY},
        'search': {'rows': search_rows, 'collections': search_rows // CHUNKS_PER_SEARCH, 'chunks': CHUNKS_PER_SEARCH},
    }


def _load_shape(engine, name_sql: str, total: int, chunks: int, label: str, centroids: int, noises: int) -> None:
    """Insert one collection shape in committed batches.

    `name_sql` builds the collection name from `grp`, the 0-based collection
    index. Using `div(g - 1 + :offset, :chunks)::bigint` keeps those indices contiguous
    from zero, so the number of collections is exactly ceil(rows / chunks) --
    an earlier version used `div(g + :offset, ...)` and produced one extra
    collection holding a single row.
    """
    batch, done, started = 500_000, 0, time.monotonic()
    while done < total:
        size = min(batch, total - done)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO {TABLE} (id, vector, collection_name, text, vmetadata) '
                    'SELECT gen_random_uuid()::text, '
                    '       c.v + n.v, '
                    f'      {name_sql}, '
                    f"      '{label} chunk ' || g || ' ' || md5(g::text), "
                    "       jsonb_build_object('grp', div(g - 1 + :offset, :chunks)::bigint::text, 'hash', md5(g::text)) "
                    'FROM generate_series(1, :n) g '
                    'JOIN _centroid c ON c.cid = mod(abs(hashint4(g + :offset)), :centroids) '
                    'JOIN _noise n ON n.nid = mod(abs(hashint4(g + :offset + 7919)), :noises)'
                ),
                {'offset': done, 'chunks': chunks, 'n': size, 'centroids': centroids, 'noises': noises},
            )
        done += size
        if total > batch:
            print(f'    {label}: {done}/{total} rows ({time.monotonic() - started:.0f}s)', flush=True)


def generate(
    engine,
    rows: int,
    dim: int,
    knowledge_bases: int,
    kb_share: float,
    centroids: int,
    noises: int,
    memory_share: float = 0.1,
    search_share: float = 0.1,
) -> Dict:
    """Build a table shaped like a real Open WebUI install.

    All four collection shapes are produced, not just knowledge bases and files:
    the bucketed shapes have to exist before anything can measure what happens
    to them when they lose their vector index.
    """
    plan = _plan(rows, knowledge_bases, kb_share, memory_share, search_share)

    print(f'Generating {rows} rows of {dim} dimensions')
    for shape in ('kb', 'file', 'memory', 'search'):
        part = plan[shape]
        print(f'  {shape:7} {part["rows"]:>9} rows in {part["collections"]:>7} collections')
    print(f'  each knowledge base is {100 * plan["kb"]["per_collection"] / rows:.3f}% of the table')

    started = time.monotonic()
    with engine.begin() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        conn.execute(text(f'DROP TABLE IF EXISTS {TABLE} CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS _centroid, _noise CASCADE'))
        conn.execute(
            text(
                'CREATE OR REPLACE FUNCTION rand_vec(dim int) RETURNS vector AS $$ '
                '  SELECT (SELECT array_agg((random() - 0.5)::real) FROM generate_series(1, dim))::vector; '
                '$$ LANGUAGE sql VOLATILE'
            )
        )
        conn.execute(
            text(
                f'CREATE TABLE _centroid AS SELECT g AS cid, rand_vec({dim}) AS v FROM generate_series(0, {centroids - 1}) g'
            )
        )
        conn.execute(
            text(
                f'CREATE TABLE _noise AS SELECT g AS nid, rand_vec({dim}) AS v FROM generate_series(0, {noises - 1}) g'
            )
        )
        conn.execute(text('CREATE UNIQUE INDEX ON _centroid (cid)'))
        conn.execute(text('CREATE UNIQUE INDEX ON _noise (nid)'))
        conn.execute(
            text(
                f'CREATE TABLE {TABLE} ('
                '  id text NOT NULL PRIMARY KEY,'
                f' vector vector({dim}),'
                '  collection_name text NOT NULL,'
                '  text text,'
                '  vmetadata jsonb)'
            )
        )

    # Knowledge bases: a dedicated partition each once migrated.
    _load_shape(
        engine,
        f"lpad(to_hex(div(g - 1 + :offset, :chunks)::bigint), 8, '0') || '{KB_SUFFIX}'",
        plan['kb']['rows'],
        plan['kb']['per_collection'],
        'kb',
        centroids,
        noises,
    )
    # Everything below is bucketed: no vector index of its own after migration.
    _load_shape(
        engine,
        "'file-' || md5(div(g - 1 + :offset, :chunks)::bigint::text)",
        plan['file']['rows'],
        CHUNKS_PER_FILE,
        'file',
        centroids,
        noises,
    )
    _load_shape(
        engine,
        "'user-memory-' || md5(div(g - 1 + :offset, :chunks)::bigint::text)",
        plan['memory']['rows'],
        CHUNKS_PER_MEMORY,
        'memory',
        centroids,
        noises,
    )
    _load_shape(
        engine,
        "'web-search-u1-' || md5(div(g - 1 + :offset, :chunks)::bigint::text)",
        plan['search']['rows'],
        CHUNKS_PER_SEARCH,
        'search',
        centroids,
        noises,
    )

    with engine.begin() as conn:
        conn.execute(text(f'ANALYZE {TABLE}'))
    print(f'Generated in {time.monotonic() - started:.0f}s')
    return plan


def verify_dataset(engine, plan: Dict) -> Dict:
    """Check the database against the plan, and refuse to continue if it differs.

    A benchmark run whose dataset was not what it claimed is indistinguishable
    from a correct one unless something checks. One earlier run held 1.27M rows
    where 1M was requested -- a stray container from a killed run was still
    inserting -- and nothing in the output revealed it.
    """
    with engine.connect() as conn:
        actual = {
            'rows': conn.execute(text(f'SELECT count(*) FROM {TABLE}')).scalar(),
            'distinct_ids': conn.execute(text(f'SELECT count(DISTINCT id) FROM {TABLE}')).scalar(),
        }
        for shape, pattern in (
            ('kb', f'%{KB_SUFFIX}'),
            ('file', 'file-%'),
            ('memory', 'user-memory-%'),
            ('search', 'web-search-%'),
        ):
            row = conn.execute(
                text(
                    f'SELECT count(*) AS rows, count(DISTINCT collection_name) AS collections '
                    f'FROM {TABLE} WHERE collection_name LIKE :pattern'
                ),
                {'pattern': pattern},
            ).one()
            actual[shape] = {'rows': row.rows, 'collections': row.collections}

    problems = []
    if actual['rows'] != plan['rows']:
        problems.append(f'total rows: expected {plan["rows"]}, found {actual["rows"]}')
    if actual['distinct_ids'] != actual['rows']:
        problems.append(f'duplicate ids: {actual["rows"]} rows but {actual["distinct_ids"]} distinct')
    for shape in ('kb', 'file', 'memory', 'search'):
        for field in ('rows', 'collections'):
            expected, found = plan[shape][field], actual[shape][field]
            if expected != found:
                problems.append(f'{shape} {field}: expected {expected}, found {found}')

    if problems:
        raise SystemExit('Dataset does not match the plan:\n  ' + '\n  '.join(problems))

    print('Dataset verified against the plan:')
    for shape in ('kb', 'file', 'memory', 'search'):
        print(f'  {shape:7} {actual[shape]["rows"]:>9} rows in {actual[shape]["collections"]:>7} collections')
    print(f'  total   {actual["rows"]:>9} rows, all ids distinct')
    return actual


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def _collect_notices(conn) -> List[str]:
    """PostgreSQL NOTICEs raised on this connection.

    pgvector reports "hnsw graph no longer fits into maintenance_work_mem" as a
    NOTICE, and the server's default `log_min_messages = warning` means it never
    reaches the server log -- it goes to the client, where psycopg2 keeps it in
    `connection.notices` and nothing reads it. Without this, a build that spilled
    to disk is indistinguishable from one that did not, except by inferring it
    from the progress curve.
    """
    try:
        raw = conn.connection.dbapi_connection
        notices = [n.strip() for n in getattr(raw, 'notices', [])]
        raw.notices.clear()
        return notices
    except Exception:  # noqa: BLE001 - diagnostics must never fail a run
        return []


def build_flat_indexes(engine, method: str, m: int, ef_construction: int, lists: int, work_mem: str) -> None:
    options = f'WITH (m = {m}, ef_construction = {ef_construction})' if method == 'hnsw' else f'WITH (lists = {lists})'
    print(f'Building {method} index {options} on the unpartitioned table')
    started = time.monotonic()
    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL maintenance_work_mem = '{work_mem}'"))
        conn.execute(text('SET LOCAL max_parallel_maintenance_workers = 4'))
        conn.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS idx_document_chunk_vector ON {TABLE} USING {method} (vector vector_cosine_ops) {options}'
            )
        )
        notices = _collect_notices(conn)
    print(f'  vector index built in {time.monotonic() - started:.0f}s', flush=True)
    for notice in notices:
        print(f'  NOTICE: {notice}', flush=True)
    if any('maintenance_work_mem' in n for n in notices):
        print(
            '  ^ the graph exceeded maintenance_work_mem and the build spilled to disk; '
            'this build time is not comparable with one that fit in memory',
            flush=True,
        )

    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL maintenance_work_mem = '{work_mem}'"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_document_chunk_text_search ON {TABLE} USING GIN (to_tsvector('simple', coalesce(text, '')))"
            )
        )
        conn.execute(
            text(f'CREATE INDEX IF NOT EXISTS idx_document_chunk_collection_name ON {TABLE} (collection_name)')
        )
    print(f'  all indexes built in {time.monotonic() - started:.0f}s')

    # A failed build must not be followed by a measurement. Without this the
    # harness happily measures a sequential scan and reports it as index
    # behaviour -- near-perfect recall at a latency nothing can explain.
    with engine.connect() as conn:
        definition = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = :t AND indexname = 'idx_document_chunk_vector'"),
            {'t': TABLE},
        ).scalar()
    if not definition or f'USING {method}' not in definition:
        raise SystemExit(f'Vector index was not created on {TABLE}; refusing to continue. Got: {definition!r}')


def build_partition_indexes(engine, method: str, m: int, ef_construction: int, lists: str, work_mem: str) -> Dict:
    """Rebuild every knowledge-base partition's vector index with given parameters.

    The migration builds these once, from the environment. Sweeping the build
    parameters afterwards would otherwise mean migrating again per point, which
    at three million rows costs more than the measurement it serves.

    `lists = auto` sizes each partition from its own row count -- pgvector's
    `rows/1000` guidance applied where the rows actually are. The default is to
    apply one global value to every partition, which is what the patch does
    today and therefore what has to be measured.
    """
    with engine.connect() as conn:
        partitions = conn.execute(
            text(
                'SELECT c.relname, c.reltuples::bigint FROM pg_class c '
                'JOIN pg_inherits i ON i.inhrelid = c.oid '
                'JOIN pg_class p ON p.oid = i.inhparent '
                "WHERE p.relname = :t AND c.relname LIKE '%\\_p\\_%' "
                'ORDER BY c.relname'
            ),
            {'t': TABLE},
        ).all()
    if not partitions:
        raise SystemExit('No dedicated partitions found -- was the migration run?')

    sized, started = [], time.monotonic()
    for name, reltuples in partitions:
        if method == 'hnsw':
            options = f'WITH (m = {m}, ef_construction = {ef_construction})'
        else:
            n = max(1, int(reltuples))
            value = max(1, round(n / 1000)) if lists == 'auto' else int(lists)
            options = f'WITH (lists = {value})'
            sized.append(value)
        with engine.begin() as conn:
            conn.execute(text(f"SET LOCAL maintenance_work_mem = '{work_mem}'"))
            conn.execute(text('SET LOCAL max_parallel_maintenance_workers = 4'))
            conn.execute(text(f'DROP INDEX IF EXISTS {name}_vector_idx'))
            conn.execute(
                text(f'CREATE INDEX {name}_vector_idx ON {name} USING {method} (vector vector_cosine_ops) {options}')
            )
            notices = _collect_notices(conn)
        for notice in notices:
            print(f'  NOTICE ({name}): {notice}', flush=True)

    built = time.monotonic() - started
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM pg_indexes WHERE indexname LIKE 'document\\_chunk\\_p\\_%vector_idx'")
        ).scalar()
    if count != len(partitions):
        raise SystemExit(f'Expected {len(partitions)} partition vector indexes, found {count}')
    return {
        'partitions_indexed': len(partitions),
        'index_method': method,
        'build_seconds': round(built, 1),
        'lists': 'auto' if lists == 'auto' else (int(lists) if method == 'ivfflat' else None),
        'lists_used': (min(sized), max(sized)) if sized else None,
        'm': m if method == 'hnsw' else None,
        'ef_construction': ef_construction if method == 'hnsw' else None,
    }


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def knowledge_bases(conn, limit: int) -> List[str]:
    """Collections that get a partition and a vector index of their own."""
    return (
        conn.execute(
            text(
                f'SELECT collection_name FROM {TABLE} '
                'WHERE collection_name LIKE :pattern '
                'GROUP BY 1 ORDER BY 1 LIMIT :limit'
            ),
            {'pattern': f'%{KB_SUFFIX}', 'limit': limit},
        )
        .scalars()
        .all()
    )


def bucket_collections(conn, limit: int) -> List[str]:
    """Collections that land in a hash bucket with only a btree on the name.

    These are the majority of a real table and the ones that lose their vector
    index under partitioning, so they are exactly where a regression would hide.
    Sampled across all three bucketed shapes rather than from `file-*` alone.
    """
    collections: List[str] = []
    per_shape = max(1, limit // len(BUCKETED_PREFIXES))
    for prefix in BUCKETED_PREFIXES:
        collections.extend(
            conn.execute(
                text(
                    f'SELECT collection_name FROM {TABLE} '
                    'WHERE collection_name LIKE :pattern '
                    'GROUP BY 1 ORDER BY 1 LIMIT :limit'
                ),
                {'pattern': f'{prefix}%', 'limit': per_shape},
            )
            .scalars()
            .all()
        )
    return collections


def describe_configuration(engine, index_method: str) -> Dict:
    """The settings a result has to carry to be comparable with another."""
    from tests.environment import describe

    wanted = (
        'shared_buffers',
        'maintenance_work_mem',
        'work_mem',
        'max_parallel_maintenance_workers',
        'max_locks_per_transaction',
        'effective_cache_size',
    )
    with engine.connect() as conn:
        settings = {
            row.name: f'{row.setting}{row.unit or ""}'
            for row in conn.execute(
                text('SELECT name, setting, unit FROM pg_settings WHERE name = ANY(:names)'),
                {'names': list(wanted)},
            ).all()
        }
        server = conn.execute(text('SHOW server_version')).scalar()
        pgvector_version = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()

    return {
        'index_method': index_method,
        'postgres': server,
        'pgvector': pgvector_version,
        'settings': settings,
        'environment': describe(),
    }


def query_vectors(conn, collection: str, count: int) -> List[tuple]:
    """Query vectors drawn from the collection itself, so they sit inside its
    cluster structure rather than in empty space.

    Returned twice over: as the text form the exact baseline casts, and as the
    float list `PgvectorClient.search()` expects.
    """
    rows = (
        conn.execute(
            text(f'SELECT vector::text FROM {TABLE} WHERE collection_name = :c ORDER BY id LIMIT :n'),
            {'c': collection, 'n': count},
        )
        .scalars()
        .all()
    )
    return [(row, [float(x) for x in row.strip('[]').split(',')]) for row in rows]


QUERY_MODES = ('self', 'perturbed', 'exclude-self')


def _perturb(vector: List[float], seed: str, relative: float = 0.3) -> List[float]:
    """A vector near `vector` but not equal to it.

    `q = v + relative * |v| * u`, with `u` a random unit vector seeded from the
    row id, so the same row always yields the same query. At 0.3 the cosine
    distance to the original is about 0.04: close enough that the original
    should be the nearest neighbour, far enough that nothing is at distance 0.
    """
    import math
    import random

    rng = random.Random(seed)
    noise = [rng.gauss(0.0, 1.0) for _ in vector]
    scale = relative * math.sqrt(sum(x * x for x in vector)) / math.sqrt(sum(n * n for n in noise))
    return [x + scale * n for x, n in zip(vector, noise)]


def measurement_queries(conn, collection: str, count: int, mode: str) -> List[tuple]:
    """Query vectors for one collection: `(text form, float list, excluded id)`.

    Drawn from the collection itself, so they sit inside its cluster structure
    rather than in empty space. That has a cost the modes address:

    - `self`: the stored vector as it is. Its own row is then at distance 0 and
      is one guaranteed hit in ten -- and a filtered index scan finds it too, so
      the unpartitioned layout is the one this flatters most.
    - `perturbed`: the stored vector plus a small random offset (`_perturb`),
      which is what a real query looks like: near a stored row, identical to none.
    - `exclude-self`: the stored vector, with its own row removed from both the
      exact answer and the index's, so recall is over the ten *other* nearest.

    `exclude-self` is the default. Measured at 200k rows (HNSW m=16,
    ef_search=40), it takes the unpartitioned figure from 0.294 to 0.206 and
    leaves the partitioned one at 0.994: the self-match was worth about one hit
    in ten to the layout that misses most, and nothing to the one that does not.
    `perturbed` changes nothing (0.294), because a query 0.04 away from its row
    still finds it. The published campaigns were run with `self`.
    """
    if mode not in QUERY_MODES:
        raise ValueError(f'query mode must be one of {QUERY_MODES}, got {mode!r}')
    rows = conn.execute(
        text(f'SELECT id, vector::text FROM {TABLE} WHERE collection_name = :c ORDER BY id LIMIT :n'),
        {'c': collection, 'n': count},
    ).all()
    out = []
    for row_id, qtext in rows:
        floats = [float(x) for x in qtext.strip('[]').split(',')]
        if mode == 'perturbed':
            floats = _perturb(floats, row_id)
            qtext = '[' + ','.join(repr(x) for x in floats) + ']'
        out.append((qtext, floats, row_id if mode == 'exclude-self' else None))
    return out


def exact_neighbours(
    conn, collection: str, qvec: str, k: int, part_key: Optional[str], exclude_id: Optional[str] = None
) -> List[str]:
    """Ground truth: a genuine exact kNN.

    Deliberately raw SQL rather than the client -- its whole purpose is to avoid
    the index the client would use. `SET enable_indexscan = off` is not enough on
    PostgreSQL 18: a disabled node is discouraged, not forbidden, so the planner
    still picks the vector index when nothing else can satisfy the ORDER BY, and
    recall ends up measured against the very index under test. Materialising the
    candidates first puts the sort on a result with no index on it.
    """
    clause = 'AND part_key = :p' if part_key else ''
    params = {'c': collection, 'q': qvec, 'k': k}
    if part_key:
        params['p'] = part_key
    if exclude_id is not None:
        clause += ' AND id <> :x'
        params['x'] = exclude_id
    return (
        conn.execute(
            text(
                f'WITH candidates AS MATERIALIZED ('
                f'  SELECT id, vector FROM {TABLE} WHERE collection_name = :c {clause}'
                f') '
                f'SELECT id FROM candidates ORDER BY vector <=> CAST(:q AS vector) LIMIT :k'
            ),
            params,
        )
        .scalars()
        .all()
    )


def make_client(db_url: str, config: str):
    """The backend's own client, so measurements describe Open WebUI's code.

    Earlier revisions of this harness issued hand-written SQL that imitated
    `search()`. It was a simplification -- a single-vector `ORDER BY`, where the
    real thing builds a LATERAL join over a VALUES list, normalises distances
    and adds the partition predicate through `_collection_scope()` -- so the
    planner could legitimately choose differently from the code being reported
    on.

    Iterative scanning is set here rather than by the harness, because it is the
    client that decides whether to apply it: configuration B is exactly
    `PGVECTOR_ITERATIVE_SCAN` turned on, so it is measured by turning it on.
    """
    from open_webui.retrieval.vector.dbs import pgvector as module

    if module.PGVECTOR_DB_URL.split('@')[-1] != db_url.split('@')[-1]:
        raise SystemExit(
            f'The client is configured for {module.PGVECTOR_DB_URL!r} but the benchmark targets {db_url!r}. '
            'Export PGVECTOR_DB_URL to the same database before running.'
        )
    # A and C leave it off so their difference is partitioning alone; B is the
    # one-line configuration change, measured on its own.
    module.PGVECTOR_ITERATIVE_SCAN = 'relaxed_order' if config == 'B' else 'off'
    return module.PgvectorClient()


def client_search(client, collection: str, vector: List[float], k: int) -> tuple:
    """One search through the real client, timed as a caller would experience it."""
    started = time.perf_counter()
    result = client.search(collection_name=collection, vectors=[vector], limit=k)
    elapsed = (time.perf_counter() - started) * 1000.0
    if result is None:
        # `search()` swallows exceptions and returns None; counting that as a
        # recall of zero would report a broken run as a bad result.
        raise SystemExit(f'search() returned None for {collection!r} -- see the backend log for the exception.')
    ids = result.ids[0] if result.ids else []
    return ids, elapsed


def captured_statement(client, collection: str, vector: List[float], k: int) -> Optional[tuple]:
    """The SQL the client actually emitted for a search, with its parameters.

    Taken from a SQLAlchemy cursor event rather than rebuilt by hand, so the plan
    reported below is the plan for the statement the backend really ran.
    """
    from sqlalchemy import event

    captured = {}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT') and 'document_chunk' in statement:
            captured['statement'] = statement
            captured['parameters'] = parameters

    engine = client.session.get_bind()
    event.listen(engine, 'before_cursor_execute', before_cursor_execute)
    try:
        client_search(client, collection, vector, k)
    finally:
        event.remove(engine, 'before_cursor_execute', before_cursor_execute)
    if 'statement' not in captured:
        return None
    return captured['statement'], captured['parameters']


def attach_sql_timer(client) -> List[float]:
    """Time the cursor execution alone, alongside the wall time of the call.

    `search()` builds a statement carrying the query vector as a thousand bound
    parameters, and marshalling them is not free: on a small collection that
    Python-side cost is most of what a caller waits for, which would flatten
    exactly the differences this benchmark exists to show. Reporting both keeps
    the client's own overhead visible instead of blaming the index for it.
    """
    from sqlalchemy import event

    timings: List[float] = []

    def before(conn, cursor, statement, parameters, context, executemany):
        context._bench_started = time.perf_counter()

    def after(conn, cursor, statement, parameters, context, executemany):
        started = getattr(context, '_bench_started', None)
        if started is not None and statement.lstrip().upper().startswith('SELECT') and 'document_chunk' in statement:
            timings.append((time.perf_counter() - started) * 1000.0)

    engine = client.session.get_bind()
    event.listen(engine, 'before_cursor_execute', before)
    event.listen(engine, 'after_cursor_execute', after)
    return timings


def plan_summary(client, collection: str, vector: List[float], k: int, index_method: str, effort: int) -> Dict:
    """Which access path the planner chose for the statement the client issued.

    Recall and latency alone cannot tell an accurate index from one the planner
    declined to use: both can show recall 1.0.

    EXPLAIN runs on the raw driver cursor because the captured statement still
    carries the driver's own placeholders; handing it back to SQLAlchemy as text
    would have it parse `%(param_1)s` as SQL.
    """
    captured = captured_statement(client, collection, vector, k)
    if captured is None:
        return {'error': 'no statement captured'}
    statement, parameters = captured

    setting = 'ivfflat.probes' if index_method == 'ivfflat' else 'hnsw.ef_search'
    client.session.rollback()
    try:
        raw = client.session.connection().connection
        cursor = raw.cursor()
        try:
            cursor.execute("SELECT '[1]'::vector")
            cursor.execute(f'SET LOCAL {setting} = {max(1, effort)}')
            cursor.execute('EXPLAIN (FORMAT JSON) ' + statement, parameters)
            rows = cursor.fetchone()[0]
        finally:
            cursor.close()
    except Exception as e:  # noqa: BLE001 - a plan is diagnostic, never fatal
        client.session.rollback()
        return {'error': str(e)[:160]}
    finally:
        client.session.rollback()

    if isinstance(rows, str):
        rows = json.loads(rows)

    nodes, relations, indexes = [], set(), set()

    def walk(node):
        nodes.append(node['Node Type'])
        if node.get('Relation Name'):
            relations.add(node['Relation Name'])
        if node.get('Index Name'):
            indexes.add(node['Index Name'])
        for child in node.get('Plans', []):
            walk(child)

    walk(rows[0]['Plan'])
    return {
        'nodes': nodes,
        'relations_scanned': len(relations),
        'uses_vector_index': any('vector' in name for name in indexes),
        'indexes': sorted(indexes)[:3],
    }


def _tune_scan(session, index_method: str, effort: int) -> None:
    """Search effort for the method in use, applied to the client's transaction.

    pgvector registers its GUCs when its library first loads into the backend,
    which only happens on first use of the vector type -- so the type is touched
    before the setting, exactly as the client does.
    """
    session.execute(text("SELECT '[1]'::vector"))
    if index_method == 'ivfflat':
        session.execute(text(f'SET LOCAL ivfflat.probes = {max(1, effort)}'))
    else:
        session.execute(text(f'SET LOCAL hnsw.ef_search = {max(1, effort)}'))


def assert_expected_indexes(engine, config: str, target: str, index_method: str) -> None:
    """Check the layout is the one the configuration claims before measuring.

    Buckets legitimately have no vector index -- that is the design -- so this
    only asserts where an index is actually expected: on the table itself for
    the unpartitioned configurations, and on the knowledge-base partitions for
    the partitioned one.
    """
    with engine.connect() as conn:
        if config in ('A', 'B'):
            definition = conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes WHERE tablename = :t AND indexname = 'idx_document_chunk_vector'"
                ),
                {'t': TABLE},
            ).scalar()
            if not definition or f'USING {index_method}' not in definition:
                raise SystemExit(
                    f'Configuration {config} expects a {index_method} index on {TABLE}, found {definition!r}. '
                    'Run the `index` command first.'
                )
        elif config == 'C' and target == 'kb':
            indexed = conn.execute(
                text("SELECT count(*) FROM pg_indexes WHERE indexname LIKE 'document\\_chunk\\_p\\_%vector_idx'")
            ).scalar()
            if not indexed:
                raise SystemExit(
                    'Configuration C expects per-partition vector indexes; none found. Was the migration run?'
                )


def measure(
    engine,
    config: str,
    k: int,
    per_collection: int,
    limit: int,
    effort: int,
    partitioned: bool,
    target: str,
    index_method: str,
    db_url: str,
    query_mode: str = 'exclude-self',
) -> Dict:
    """Recall and latency through `PgvectorClient.search()`.

    The client is what `query_collection` calls, so this exercises
    `adjust_vector_length`, the LATERAL fan-out over query vectors, distance
    normalisation, `_collection_scope()` and `_apply_iterative_scan()` -- the
    whole path, rather than an imitation of the SQL it ends up emitting.
    """
    part_key_for = None
    if partitioned:
        from open_webui.retrieval.vector.dbs.pgvector import part_key_for as _pk

        part_key_for = _pk

    assert_expected_indexes(engine, config, target, index_method)
    client = make_client(db_url, config)
    sql_timings = attach_sql_timer(client)

    recalls, latencies = [], []
    with engine.connect() as conn:
        select = knowledge_bases if target == 'kb' else bucket_collections
        kbs = select(conn, limit)
        print(
            f'Measuring config {config} ({index_method}, effort {effort}) over '
            f'{len(kbs)} {target} collections, {per_collection} queries each'
        )
        # (text form for the exact baseline, float list for the client, and
        # the row to leave out when the mode asks for it)
        queries = {c: measurement_queries(conn, c, per_collection, query_mode) for c in kbs}

    if index_method == 'hnsw' and config != 'B' and effort < k + (1 if query_mode == 'exclude-self' else 0):
        raise SystemExit(
            f'ef_search={effort} is below the {k + 1} rows this measurement asks for: pgvector returns at most '
            'ef_search rows without iterative scans, so recall would be capped by the setting rather than measured.'
        )

    def search_once(collection: str, vector: List[float], excluded: Optional[str]) -> tuple:
        # SET LOCAL only holds inside a transaction we control, and the client
        # rolls back when it is done, so the effort is applied per search.
        client.session.rollback()
        _tune_scan(client.session, index_method, effort)
        if excluded is None:
            return client_search(client, collection, vector, k)
        # One more, so that dropping the query's own row still leaves k.
        got, elapsed = client_search(client, collection, vector, k + 1)
        return [i for i in got if i != excluded][:k], elapsed

    # Warm the cache first: otherwise the configuration measured first pays for
    # every page the others then find resident, and latency ends up ranking the
    # running order rather than the configurations.
    for collection, triples in queries.items():
        for _, vector, excluded in triples:
            search_once(collection, vector, excluded)

    sql_timings.clear()  # drop the warm-up pass
    for collection, triples in queries.items():
        pk = part_key_for(collection) if part_key_for else None
        for qtext, vector, excluded in triples:
            with engine.connect() as conn:
                conn.rollback()
                with conn.begin():
                    truth = exact_neighbours(conn, collection, qtext, k, pk, exclude_id=excluded)

            got, elapsed = search_once(collection, vector, excluded)
            if truth:
                recalls.append(len(set(got) & set(truth)) / len(truth))
            latencies.append(elapsed)

    measured_sql = sorted(sql_timings)
    first = kbs[0]
    plan = plan_summary(client, first, queries[first][0][1], k, index_method, effort)
    client.session.rollback()

    latencies.sort()
    if not latencies:
        raise SystemExit(f'No {target} collections found to measure -- was the dataset generated?')
    return {
        'config': config,
        'target': target,
        'collections': len(kbs),
        'queries': len(latencies),
        'warmed_up': True,
        'search_effort': effort,
        'through_client': True,
        'iterative_scan': config == 'B',
        'query_mode': query_mode,
        'recall_at_k': round(statistics.fmean(recalls), 4) if recalls else None,
        'recall_p10': round(sorted(recalls)[len(recalls) // 10], 4) if recalls else None,
        'latency_ms_median': round(statistics.median(latencies), 2),
        'latency_ms_p95': round(latencies[int(len(latencies) * 0.95)], 2),
        # Cursor execution alone: the same search without the cost of binding a
        # thousand parameters, which is what changes when the index changes.
        'sql_ms_median': round(statistics.median(measured_sql), 2) if measured_sql else None,
        'sql_ms_p95': round(measured_sql[int(len(measured_sql) * 0.95)], 2) if measured_sql else None,
        'plan': plan,
    }


# --------------------------------------------------------------------------
# Real embeddings: needles in a synthetic haystack
# --------------------------------------------------------------------------


def _embed_texts(url: str, model: str, texts: List[str], batch: int = 16) -> List[List[float]]:
    """Embed through an OpenAI-compatible endpoint, in small batches.

    Small batches and no concurrency on purpose: the endpoint is one shared GPU,
    and a benchmark that saturates it is measuring the wrong machine.
    """
    import urllib.request

    out: List[List[float]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        payload = json.dumps({'model': model, 'input': chunk}).encode()
        request = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
        if 'data' not in body:
            raise SystemExit(f'Embedding endpoint returned no data: {str(body)[:200]}')
        # The API does not promise input order, and index is what identifies a row.
        ordered = sorted(body['data'], key=lambda d: d['index'])
        out.extend(d['embedding'] for d in ordered)
        print(f'    embedded {len(out)}/{len(texts)}', flush=True)
    if len(out) != len(texts):
        raise SystemExit(f'Asked for {len(texts)} embeddings, received {len(out)}')
    return out


def implant_needles(engine, url: str, model: str, dim: int, hard_distractors: int) -> Dict:
    """Overwrite rows of one knowledge base with real embeddings and real text.

    Rows are *replaced* rather than added, so the dataset still matches the plan
    `verify_dataset` checks: the composition is unchanged, only the contents of
    a few hundred rows in a single collection.

    Question vectors are embedded here and stored, so a measurement campaign
    never has to reach the embedding endpoint -- and every campaign queries with
    byte-identical vectors, which a fresh embedding call could not guarantee.
    """
    try:
        from tests.vector.needle_corpus import DISTRACTORS, PAIRS
    except ImportError:  # run as a script rather than as a module
        from needle_corpus import DISTRACTORS, PAIRS

    passages = [passage for passage, _ in PAIRS] + list(DISTRACTORS)
    questions = [question for _, question in PAIRS]

    with engine.connect() as conn:
        collection = knowledge_bases(conn, 1)
        if not collection:
            raise SystemExit('No knowledge-base collection found -- generate the dataset first.')
        collection = collection[0]
        ids = (
            conn.execute(
                text(f'SELECT id FROM {TABLE} WHERE collection_name = :c ORDER BY id LIMIT :n'),
                {'c': collection, 'n': len(passages)},
            )
            .scalars()
            .all()
        )
    if len(ids) < len(passages):
        raise SystemExit(f'Knowledge base {collection} holds {len(ids)} rows, need {len(passages)}')

    print(f'Embedding {len(passages)} passages and {len(questions)} questions through {model}')
    passage_vectors = _embed_texts(url, model, passages)
    question_vectors = _embed_texts(url, model, questions)
    if len(passage_vectors[0]) != dim:
        raise SystemExit(f'Endpoint returns {len(passage_vectors[0])}-dimensional vectors, the table holds {dim}')

    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS _needle'))
        conn.execute(
            text(
                'CREATE TABLE _needle ('
                '  id text PRIMARY KEY,'
                '  collection_name text NOT NULL,'
                '  passage text NOT NULL,'
                '  question text,'
                f' qvector vector({dim}))'
            )
        )
        for position, (row_id, passage, vector) in enumerate(zip(ids, passages, passage_vectors)):
            question = questions[position] if position < len(questions) else None
            qvector = question_vectors[position] if position < len(questions) else None
            conn.execute(
                text(
                    f'UPDATE {TABLE} SET vector = CAST(:v AS vector), text = :t, '
                    "vmetadata = vmetadata || jsonb_build_object('needle', true) WHERE id = :id"
                ),
                {'v': str(vector), 't': passage, 'id': row_id},
            )
            conn.execute(
                text(
                    'INSERT INTO _needle (id, collection_name, passage, question, qvector) '
                    'VALUES (:id, :c, :p, :q, CAST(:qv AS vector))'
                ),
                {
                    'id': row_id,
                    'c': collection,
                    'p': passage,
                    'q': question,
                    'qv': str(qvector) if qvector is not None else None,
                },
            )
        conn.execute(text(f'ANALYZE {TABLE}'))

    hard = _add_hard_distractors(engine, collection, dim, hard_distractors)

    return {
        'collection': collection,
        'passages': len(passages),
        'questions': len(questions),
        'model': model,
        'dim': len(passage_vectors[0]),
        'hard_distractors': hard,
    }


def _add_hard_distractors(engine, collection: str, dim: int, count: int) -> Dict:
    """Give the needles real competition.

    Left alone, the synthetic vectors sit almost orthogonal to the embeddings --
    around 0.94 cosine distance from a question, where the right passage is at
    0.35. The index would then only have to separate two distributions, which is
    not what burying a document in a knowledge base is like.

    So a share of the collection's synthetic rows is rewritten as a real passage
    plus scaled noise, at several noise levels, producing neighbours spread from
    just beside a passage to well away from it. Some of them necessarily land
    nearer a question than its own passage does; that is why the needle report
    measures the exact rank as well as the approximate one, and attributes only
    the difference to the index.
    """
    if count <= 0:
        return {'rows': 0}
    alphas = (0.03, 0.05, 0.08, 0.12, 0.2, 0.3)
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS _needle_vec, _scale'))
        conn.execute(
            text(
                'CREATE TABLE _needle_vec AS '
                f'SELECT (row_number() OVER (ORDER BY n.id) - 1) AS rn, d.vector FROM _needle n '
                f'JOIN {TABLE} d ON d.id = n.id'
            )
        )
        needles = conn.execute(text('SELECT count(*) FROM _needle_vec')).scalar()
        conn.execute(
            text(
                'CREATE TABLE _scale AS SELECT * FROM (VALUES '
                + ', '.join(f'({i}, array_fill({a}::real, ARRAY[{dim}])::vector)' for i, a in enumerate(alphas))
                + ') AS t(k, v)'
            )
        )
        noises = conn.execute(text('SELECT count(*) FROM _noise')).scalar()
        rewritten = conn.execute(
            text(
                f'WITH target AS ('
                f'  SELECT id, (row_number() OVER (ORDER BY id) - 1)::int AS rn FROM {TABLE} '
                f'  WHERE collection_name = :c AND id NOT IN (SELECT id FROM _needle) LIMIT :n'
                f') '
                f'UPDATE {TABLE} d SET vector = nv.vector + (nz.v * sc.v) '
                f'FROM target t '
                f'JOIN _needle_vec nv ON nv.rn = mod(abs(hashint4(t.rn)), :needles) '
                f'JOIN _noise nz ON nz.nid = mod(abs(hashint4(t.rn + 7919)), :noises) '
                f'JOIN _scale sc ON sc.k = mod(t.rn, :scales) '
                f'WHERE d.id = t.id'
            ),
            {'c': collection, 'n': count, 'needles': needles, 'noises': noises, 'scales': len(alphas)},
        ).rowcount
        conn.execute(text(f'ANALYZE {TABLE}'))
    return {'rows': rewritten, 'noise_levels': list(alphas)}


def needle_report(
    engine,
    db_url: str,
    config: str,
    index_method: str,
    effort: int,
    partitioned: bool,
    depth: int,
) -> Dict:
    """Can a real answer be found once it is buried in synthetic noise?

    recall@10 over synthetic vectors says how well an index reproduces an exact
    sort. It does not say whether a genuine answer survives the walk, and rank
    is a sharper signal than recall because it says *how far* wrong the miss is.

    Two ranks are reported for every question. The exact one is a property of
    the embedding model: if the model does not put the right passage first, no
    index can. The approximate one is the property under test. Only the gap
    between them is attributable to the index.
    """
    part_key_for = None
    if partitioned:
        from open_webui.retrieval.vector.dbs.pgvector import part_key_for as _pk

        part_key_for = _pk

    client = make_client(db_url, config)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                'SELECT id, collection_name, question, qvector::text FROM _needle WHERE question IS NOT NULL ORDER BY id'
            )
        ).all()
    if not rows:
        raise SystemExit('No needles implanted -- run the `embed` command first.')

    exact_ranks, index_ranks = [], []
    missed_by_index, missed_by_model = [], []
    for row in rows:
        vector = [float(x) for x in row[3].strip('[]').split(',')]
        pk = part_key_for(row.collection_name) if part_key_for else None

        with engine.connect() as conn, conn.begin():
            truth = exact_neighbours(conn, row.collection_name, row[3], depth, pk)

        client.session.rollback()
        _tune_scan(client.session, index_method, effort)
        got, _ = client_search(client, row.collection_name, vector, depth)

        exact_rank = truth.index(row.id) + 1 if row.id in truth else None
        index_rank = list(got).index(row.id) + 1 if row.id in got else None
        exact_ranks.append(exact_rank)
        index_ranks.append(index_rank)
        if exact_rank is not None and index_rank is None:
            missed_by_index.append(row.question)
        if exact_rank is None:
            missed_by_model.append(row.question)
    client.session.rollback()

    def top(ranks, n):
        return sum(1 for r in ranks if r is not None and r <= n)

    found = [r for r in index_ranks if r is not None]
    # pgvector's HNSW scan returns at most ef_search rows unless iterative scans
    # are on, so below that depth "not returned within depth" means "not within
    # ef_search" -- and configuration B, which returns the full depth, is not
    # comparable on that column. `index_top10` is the comparable one.
    capped = index_method == 'hnsw' and config != 'B' and effort < depth
    if capped:
        print(f'note: ef_search={effort} < depth={depth}; the scan returns at most {effort} rows', file=sys.stderr)
    return {
        'config': config,
        'index_method': index_method,
        'search_effort': effort,
        'questions': len(rows),
        'depth': depth,
        'rows_returned_capped_at': effort if capped else None,
        'exact_top1': top(exact_ranks, 1),
        'exact_top10': top(exact_ranks, 10),
        'index_top1': top(index_ranks, 1),
        'index_top10': top(index_ranks, 10),
        'index_median_rank': statistics.median(found) if found else None,
        # The number that belongs to the index rather than to the model: the
        # passage was reachable by an exact sort and the index did not return it.
        'reachable_but_missed': len(missed_by_index),
        'unreachable_for_the_model': len(missed_by_model),
        'examples_missed_by_index': missed_by_index[:3],
    }


# --------------------------------------------------------------------------
# Several knowledge bases at once
# --------------------------------------------------------------------------


def _pin_effort(client, index_method: str, effort: int) -> str:
    """Fix the search effort for every connection the pool hands out.

    `query_collection` fans out over threads, so the effort cannot be set on one
    session the way a single measurement does it. Setting it on connect, and
    committing, makes it hold for whichever thread picks the connection up.

    Returns what the server reports afterwards, so a result carries the effort
    that was actually in force rather than the one that was asked for.
    """
    from sqlalchemy import event

    setting = 'ivfflat.probes' if index_method == 'ivfflat' else 'hnsw.ef_search'
    engine = client.session.get_bind()

    @event.listens_for(engine, 'connect')
    def set_effort(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        # Touch the type first: pgvector registers its settings on library load.
        cursor.execute("SELECT '[1]'::vector")
        cursor.execute(f'SET {setting} = {max(1, effort)}')
        cursor.close()
        dbapi_connection.commit()

    engine.dispose()  # so already-pooled connections also get the setting
    with engine.connect() as conn:
        observed = conn.execute(text('SELECT current_setting(:s)'), {'s': setting}).scalar()
    return f'{setting}={observed}'


def multi_kb_report(
    engine,
    db_url: str,
    config: str,
    index_method: str,
    effort: int,
    partitioned: bool,
    k: int,
    rounds: int,
    counts: List[int],
) -> Dict:
    """What happens when a user selects several knowledge bases at once.

    Measured through `query_collection`, not through an imitation of it, because
    the interesting part is exactly what that function does: it does **not**
    aggregate. It fans out over the cartesian product of queries and
    collections, so N collections mean N independent per-collection searches,
    each asking for the full `k` rather than `k/N`, merged afterwards by content
    hash and re-sorted.

    Partitioning changes nothing structural here -- each of those N searches was
    already scoped to one collection. What it changes is what each one costs.
    """
    import asyncio

    from open_webui.retrieval.utils import query_collection
    from open_webui.retrieval.vector.factory import get_vector_db_client

    # `make_client` is called for its checks and for the iterative-scan setting,
    # but the searches below go through the process-wide client the application
    # builds at import: that is the one `query_doc` reaches for, and pinning the
    # search effort on any other engine would leave these queries running at the
    # server default.
    make_client(db_url, config)
    app_client = get_vector_db_client()
    effort_observed = _pin_effort(app_client, index_method, effort)

    part_key_for = None
    if partitioned:
        from open_webui.retrieval.vector.dbs.pgvector import part_key_for as _pk

        part_key_for = _pk

    with engine.connect() as conn:
        # Never the collection holding the real embeddings: those form a tight
        # cluster nearly orthogonal to the synthetic bulk, and any index finds
        # them -- an earlier run queried from it and reported recall 1.0 for
        # every layout at every effort, which the per-collection measurement
        # on the same data flatly contradicts.
        reserved = None
        if conn.execute(text("SELECT to_regclass('_needle') IS NOT NULL")).scalar():
            reserved = conn.execute(text('SELECT collection_name FROM _needle LIMIT 1')).scalar()
        available = [c for c in knowledge_bases(conn, max(counts) + 1) if c != reserved][: max(counts)]
        if len(available) < max(counts):
            raise SystemExit(f'Need {max(counts)} knowledge bases, found {len(available)}')
        # (own text, text form, float list): the query's own row is dropped from
        # both sides, as the per-collection measurement does by default.
        queries = {
            c: [
                (row.text, row.qtext, [float(x) for x in row.qtext.strip('[]').split(',')])
                for row in conn.execute(
                    text(
                        f'SELECT text, vector::text AS qtext FROM {TABLE} '
                        'WHERE collection_name = :c ORDER BY id LIMIT :n'
                    ),
                    {'c': c, 'n': rounds},
                ).all()
            ]
            for c in available
        }

    async def fan_out(collections: List[str], vector: List[float]) -> Dict:
        async def embedding_function(texts, prefix=None):
            return [vector]

        return await query_collection(
            None,  # no request: the hybrid-search branch needs an app, and this measures the vector path
            collection_names=collections,
            queries=['needle'],
            embedding_function=embedding_function,
            k=k + 1,  # one spare, so dropping the query's own row still leaves k
        )

    # Warm the cache first, exactly as the single-collection measurement does.
    # Without it the layout measured last pays for every page the earlier ones
    # left resident -- and the partitioned layout is always measured last,
    # because the migration is done in place. At three million rows that turned
    # a 25 ms median into 193 ms and made latency rank the running order
    # instead of the layouts.
    for n in counts:
        for round_index in range(rounds):
            asyncio.run(fan_out(available[:n], queries[available[0]][round_index][2]))

    results = []
    for n in counts:
        collections = available[:n]
        latencies, recalls = [], []
        for round_index in range(rounds):
            # One query vector per round, drawn from the first collection, so the
            # same question is put to every layout and every N.
            own_text, vector_text, vector = queries[collections[0]][round_index]

            # Ground truth for the merged answer: the exact top-k of the union,
            # which is what a user selecting N knowledge bases is asking for.
            truth = []
            with engine.connect() as conn, conn.begin():
                for collection in collections:
                    pk = part_key_for(collection) if part_key_for else None
                    truth.extend(
                        conn.execute(
                            text(
                                f'WITH candidates AS MATERIALIZED ('
                                f'  SELECT text, vector FROM {TABLE} WHERE collection_name = :c'
                                f'{" AND part_key = :p" if pk else ""}'
                                f') '
                                f'SELECT text, vector <=> CAST(:q AS vector) AS d FROM candidates '
                                f'WHERE text <> :own ORDER BY d LIMIT :k'
                            ),
                            {'c': collection, 'q': vector_text, 'k': k, 'own': own_text, **({'p': pk} if pk else {})},
                        ).all()
                    )
            truth.sort(key=lambda r: r.d)
            expected = [r.text for r in truth[:k]]

            started = time.perf_counter()
            merged = asyncio.run(fan_out(collections, vector))
            latencies.append((time.perf_counter() - started) * 1000.0)

            documents = [d for d in (merged['documents'][0] if merged['documents'] else []) if d != own_text][:k]
            if expected:
                recalls.append(len(set(documents) & set(expected)) / len(expected))

        latencies.sort()
        results.append(
            {
                'collections': n,
                'searches_issued': n,  # query_collection fans out one search per collection
                'rows_requested': n * k,  # k per collection, not k/n
                'rounds': rounds,
                'warmed_up': True,
                'recall_at_k': round(statistics.fmean(recalls), 4) if recalls else None,
                'latency_ms_median': round(statistics.median(latencies), 2),
                'latency_ms_p95': round(latencies[int(len(latencies) * 0.95)], 2),
            }
        )

    return {
        'config': config,
        'index_method': index_method,
        'search_effort': effort,
        'effort_in_force': effort_observed,
        'k': k,
        'note': 'query_collection issues one search per collection and merges afterwards; it does not aggregate',
        'by_collection_count': results,
    }


def index_report(engine) -> Dict:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                'SELECT c.relname AS table, i.relname AS index, pg_relation_size(i.oid) AS bytes '
                'FROM pg_class c '
                'JOIN pg_index x ON x.indrelid = c.oid '
                'JOIN pg_class i ON i.oid = x.indexrelid '
                "WHERE c.relname LIKE 'document_chunk%' AND c.relkind IN ('r','p') "
                'ORDER BY bytes DESC'
            )
        ).all()
        total_heap = conn.execute(
            text("SELECT sum(pg_table_size(oid)) FROM pg_class WHERE relname LIKE 'document_chunk%' AND relkind = 'r'")
        ).scalar()

    total_heap = float(total_heap or 0)
    vector_bytes = sum(r.bytes for r in rows if 'vector' in r.index)
    text_bytes = sum(r.bytes for r in rows if 'text' in r.index or 'tsvector' in r.index)
    other_bytes = sum(r.bytes for r in rows) - vector_bytes - text_bytes
    return {
        'vector_index_mb': round(vector_bytes / 1024**2, 1),
        'text_index_mb': round(text_bytes / 1024**2, 1),
        'other_index_mb': round(other_bytes / 1024**2, 1),
        'total_index_mb': round(sum(r.bytes for r in rows) / 1024**2, 1),
        'heap_mb': round(total_heap / 1024**2, 1),
        'index_count': len(rows),
    }


def _timed_vacuum(engine, force_index_cleanup: bool) -> float:
    """Time a VACUUM, optionally forcing it to actually visit the indexes.

    PostgreSQL skips index vacuuming when the pages holding dead tuples are a
    small enough fraction of the table -- around 2%. Deleting one knowledge base
    out of a large table falls under that, so a plain VACUUM never touches the
    vector index and finishes in milliseconds. That is real and worth reporting,
    but it is not the cost the partitioning argument is about: sooner or later
    the threshold is crossed, and then vacuum has to walk the whole graph.
    Reporting only the bypassed figure would make the problem look like it does
    not exist; reporting only the forced one would overstate how often it bites.
    """
    options = 'INDEX_CLEANUP ON' if force_index_cleanup else ''
    statement = f'VACUUM ({options}) {TABLE}' if options else f'VACUUM {TABLE}'
    started = time.monotonic()
    raw = engine.raw_connection()
    try:
        raw.set_isolation_level(0)  # VACUUM cannot run inside a transaction block
        cur = raw.cursor()
        cur.execute(statement)
        cur.close()
    finally:
        raw.close()
    return round(time.monotonic() - started, 1)


def vacuum_report(engine, churn_collection: Optional[str]) -> Dict:
    """Time a vacuum pass after deleting one collection's rows.

    Phase 2 of VACUUM walks every index on the table, and pgvector's HNSW bulk
    delete has to traverse the graph to repair links -- so on a single table,
    touching one small knowledge base eventually drags the whole global graph.
    """
    with engine.connect() as conn:
        if churn_collection is None:
            churn_collection = _churnable_collection(conn)

    with engine.begin() as conn:
        deleted = conn.execute(
            text(f'DELETE FROM {TABLE} WHERE collection_name = :c'), {'c': churn_collection}
        ).rowcount

    plain = _timed_vacuum(engine, force_index_cleanup=False)
    forced = _timed_vacuum(engine, force_index_cleanup=True)
    return {
        'churn_collection': churn_collection,
        'rows_deleted': deleted,
        'vacuum_seconds': plain,
        'vacuum_seconds_index_cleanup_forced': forced,
    }


def _churnable_collection(conn) -> str:
    """A knowledge base that may be destroyed by a churn or vacuum measurement.

    Never the one holding the needles: that collection ceasing to exist would
    make a later needle run report a total loss the index had nothing to do
    with.
    """
    reserved = None
    if conn.execute(text("SELECT to_regclass('_needle') IS NOT NULL")).scalar():
        reserved = conn.execute(text('SELECT collection_name FROM _needle LIMIT 1')).scalar()
    # From the end of the list, because measurements sample from the beginning:
    # the unpartitioned churn runs before the migration, so taking a collection
    # a later measurement would have used would leave the two layouts measuring
    # different data.
    candidates = [
        c
        for c in conn.execute(
            text(
                f'SELECT collection_name FROM {TABLE} WHERE collection_name LIKE :pattern '
                'GROUP BY 1 ORDER BY 1 DESC LIMIT 3'
            ),
            {'pattern': f'%{KB_SUFFIX}'},
        ).scalars()
        if c != reserved
    ]
    if not candidates:
        raise SystemExit('No knowledge base available to delete for this measurement.')
    return candidates[0]


def churn_report(engine, partitioned: bool) -> Dict:
    """Time what the application actually does when a knowledge base is removed.

    The `vacuum` command deletes rows behind the backend's back, which is the
    pessimistic case. The real path goes through `delete_collection()`, and that
    is where the layouts differ in kind rather than degree: unpartitioned it is
    a DELETE that leaves a dead tuple per row for autovacuum to reclaim through
    the vector index, partitioned it is a DROP of the collection's own
    partition, which leaves nothing behind at all.
    """
    from open_webui.retrieval.vector.dbs.pgvector import PgvectorClient

    with engine.connect() as conn:
        collection = _churnable_collection(conn)
        rows = conn.execute(
            text(f'SELECT count(*) FROM {TABLE} WHERE collection_name = :c'), {'c': collection}
        ).scalar()

    client = PgvectorClient()
    started = time.monotonic()
    client.delete_collection(collection)
    delete_seconds = time.monotonic() - started

    return {
        'layout': 'partitioned' if partitioned else 'unpartitioned',
        'collection': collection,
        'rows': rows,
        'delete_collection_seconds': round(delete_seconds, 2),
        'following_vacuum_seconds': _timed_vacuum(engine, force_index_cleanup=False),
        'following_vacuum_seconds_index_cleanup_forced': _timed_vacuum(engine, force_index_cleanup=True),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'command',
        choices=[
            'generate',
            'verify',
            'index',
            'partindex',
            'measure',
            'sizes',
            'vacuum',
            'churn',
            'embed',
            'needle',
            'multikb',
        ],
    )
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--rows', type=int, default=5_000_000)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--knowledge-bases', type=int, default=20)
    parser.add_argument('--kb-share', type=float, default=0.3, help='fraction of rows in knowledge bases')
    parser.add_argument('--centroids', type=int, default=500)
    parser.add_argument('--noises', type=int, default=5000)
    parser.add_argument('--method', choices=['hnsw', 'ivfflat'], default='hnsw')
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--ef-construction', type=int, default=64)
    parser.add_argument('--lists', default='1000', help="ivfflat lists; 'auto' sizes each partition from its rows")
    parser.add_argument('--work-mem', default='2GB', help='maintenance_work_mem for index builds')
    parser.add_argument('--config', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--queries-per-kb', type=int, default=25)
    parser.add_argument('--kb-limit', type=int, default=10)
    parser.add_argument('--ef-search', type=int, default=40)
    parser.add_argument(
        '--query-mode',
        choices=QUERY_MODES,
        default='exclude-self',
        help='self: stored vectors as queries; perturbed: nearby but not identical; exclude-self: drop the query row',
    )
    parser.add_argument('--partitioned', action='store_true')
    parser.add_argument(
        '--target',
        choices=['kb', 'buckets'],
        default='kb',
        help='measure knowledge bases (own partition and index) or bucketed collections (btree only)',
    )
    parser.add_argument('--memory-share', type=float, default=0.1)
    parser.add_argument('--search-share', type=float, default=0.1)
    # Any OpenAI-compatible embeddings endpoint; the campaigns used a local one.
    parser.add_argument('--embed-url', default=os.environ.get('OWUI_BENCH_EMBED_URL', ''))
    parser.add_argument('--embed-model', default=os.environ.get('OWUI_BENCH_EMBED_MODEL', ''))
    parser.add_argument('--needle-depth', type=int, default=50, help='how far down the result list to look')
    parser.add_argument(
        '--hard-distractors',
        type=int,
        default=20000,
        help='synthetic rows in the needle collection rewritten as real passage + noise',
    )
    parser.add_argument('--multikb-counts', default='1,3,10', help='how many knowledge bases to select at once')
    parser.add_argument('--rounds', type=int, default=20)
    args = parser.parse_args(argv)

    engine = create_engine(args.db_url)
    try:
        if args.command == 'generate':
            plan = generate(
                engine,
                args.rows,
                args.dim,
                args.knowledge_bases,
                args.kb_share,
                args.centroids,
                args.noises,
                args.memory_share,
                args.search_share,
            )
            # Verifying here rather than as a separate step means a bad dataset
            # cannot reach a measurement at all.
            verify_dataset(engine, plan)
        elif args.command == 'verify':
            verify_dataset(
                engine,
                _plan(args.rows, args.knowledge_bases, args.kb_share, args.memory_share, args.search_share),
            )
        elif args.command == 'index':
            build_flat_indexes(engine, args.method, args.m, args.ef_construction, int(args.lists), args.work_mem)
        elif args.command == 'partindex':
            print(
                json.dumps(
                    build_partition_indexes(
                        engine, args.method, args.m, args.ef_construction, args.lists, args.work_mem
                    ),
                    indent=2,
                )
            )
        elif args.command == 'measure':
            result = measure(
                engine,
                args.config,
                args.k,
                args.queries_per_kb,
                args.kb_limit,
                args.ef_search,
                args.partitioned,
                args.target,
                args.method,
                args.db_url,
                args.query_mode,
            )
            result.update(index_report(engine))
            result['configuration'] = describe_configuration(engine, args.method)
            print(json.dumps(result, indent=2))
        elif args.command == 'embed':
            if not args.embed_url or not args.embed_model:
                raise SystemExit('embed needs --embed-url and --embed-model (or OWUI_BENCH_EMBED_URL / _MODEL)')
            print(
                json.dumps(
                    implant_needles(engine, args.embed_url, args.embed_model, args.dim, args.hard_distractors),
                    indent=2,
                )
            )
        elif args.command == 'needle':
            result = needle_report(
                engine,
                args.db_url,
                args.config,
                args.method,
                args.ef_search,
                args.partitioned,
                args.needle_depth,
            )
            result['configuration'] = describe_configuration(engine, args.method)
            print(json.dumps(result, indent=2))
        elif args.command == 'multikb':
            result = multi_kb_report(
                engine,
                args.db_url,
                args.config,
                args.method,
                args.ef_search,
                args.partitioned,
                args.k,
                args.rounds,
                [int(n) for n in args.multikb_counts.split(',')],
            )
            result['configuration'] = describe_configuration(engine, args.method)
            print(json.dumps(result, indent=2))
        elif args.command == 'sizes':
            print(json.dumps(index_report(engine), indent=2))
        elif args.command == 'churn':
            print(json.dumps(churn_report(engine, args.partitioned), indent=2))
        else:
            print(json.dumps(vacuum_report(engine, None), indent=2))
        return 0
    finally:
        engine.dispose()


if __name__ == '__main__':
    sys.exit(main())
