"""Does building a partition's IVFFlat index before it holds rows cost recall?

The earlier answer (0.575 against 0.888) came from queries issued without a
collection filter, which is not a shape Open WebUI ever produces -- and the
client's own filtered queries usually do not reach an unpartitioned IVFFlat
index at all, so that comparison could not be reproduced through
`PgvectorClient`.

The case that *is* reachable, and the one the partitioning patch actually
creates, is a dedicated partition: `_create_dedicated_partition` builds its
vector index at creation time, when the partition is empty, while the migration
builds it after loading. Same install, same configuration, two behaviours. This
measures that difference the way the application would experience it -- through
`search()`, scoped to the collection.
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, '$REPO')
sys.path.insert(0, '$REPO/backend')

from sqlalchemy import create_engine, text  # noqa: E402

from tests.vector.bench_pgvector_partitioning import (  # noqa: E402
    client_search,
    exact_neighbours,
    knowledge_bases,
    make_client,
    plan_summary,
    query_vectors,
    _tune_scan,
)

DB_URL = os.environ['PGVECTOR_DB_URL']
LISTS = int(os.environ.get('LISTS', '15'))
PROBES = int(os.environ.get('PROBES', '3'))
K = 10


def partition_of(engine, collection):
    from open_webui.retrieval.vector.dbs.pgvector import part_key_for, partition_name_for

    return partition_name_for(part_key_for(collection)), part_key_for(collection)


def rebuild(engine, partition, when):
    """`when='empty'` reproduces the runtime path: index first, rows after."""
    with engine.begin() as conn:
        conn.execute(text(f'DROP INDEX IF EXISTS {partition}_vector_idx'))
        conn.execute(text(f'CREATE TABLE _stash AS SELECT * FROM {partition}'))
        conn.execute(text(f'DELETE FROM {partition}'))
    started = time.monotonic()
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL maintenance_work_mem = '2GB'"))
        if when == 'empty':
            conn.execute(
                text(
                    f'CREATE INDEX {partition}_vector_idx ON {partition} '
                    f'USING ivfflat (vector vector_cosine_ops) WITH (lists = {LISTS})'
                )
            )
        conn.execute(text(f'INSERT INTO {partition} SELECT * FROM _stash'))
        if when == 'loaded':
            conn.execute(
                text(
                    f'CREATE INDEX {partition}_vector_idx ON {partition} '
                    f'USING ivfflat (vector vector_cosine_ops) WITH (lists = {LISTS})'
                )
            )
    build = time.monotonic() - started
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE _stash'))
        conn.execute(text(f'ANALYZE {partition}'))
    return build


def measure(engine, client, collection, part_key, per_collection):
    with engine.connect() as conn:
        pairs = query_vectors(conn, collection, per_collection)
    recalls, latencies = [], []
    for _ in range(2):  # warm-up, then the measured pass
        recalls, latencies = [], []
        for qtext, vector in pairs:
            with engine.connect() as conn, conn.begin():
                truth = exact_neighbours(conn, collection, qtext, K, part_key)
            client.session.rollback()
            _tune_scan(client.session, 'ivfflat', PROBES)
            got, elapsed = client_search(client, collection, vector, K)
            if truth:
                recalls.append(len(set(got) & set(truth)) / len(truth))
            latencies.append(elapsed)
    plan = plan_summary(client, collection, pairs[0][1], K, 'ivfflat', PROBES)
    client.session.rollback()
    return {
        'recall_at_k': round(statistics.fmean(recalls), 4),
        'recall_p10': round(sorted(recalls)[len(recalls) // 10], 4),
        'latency_ms_median': round(statistics.median(latencies), 2),
        'uses_vector_index': plan.get('uses_vector_index'),
        'queries': len(recalls),
    }


def main():
    engine = create_engine(DB_URL)
    client = make_client(DB_URL, 'C')
    with engine.connect() as conn:
        collection = knowledge_bases(conn, 1)[0]
        rows = conn.execute(
            text('SELECT count(*) FROM document_chunk WHERE collection_name = :c'), {'c': collection}
        ).scalar()
    partition, part_key = partition_of(engine, collection)

    out = {'collection': collection, 'partition_rows': rows, 'lists': LISTS, 'probes': PROBES, 'cases': {}}
    for when in ('loaded', 'empty'):
        build = rebuild(engine, partition, when)
        result = measure(engine, client, collection, part_key, 40)
        result['index_built'] = when
        result['build_and_load_seconds'] = round(build, 1)
        out['cases'][when] = result
    print(json.dumps(out, indent=2))
    engine.dispose()


if __name__ == '__main__':
    main()
