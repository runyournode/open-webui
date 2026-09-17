"""Convert `document_chunk` to the partitioned layout, and back.

Run this with Open WebUI stopped. The copy is a snapshot: rows written to the
old table after their collection has been copied would not be carried over.

    python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate --plan
    python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate --migrate
    python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate --rollback

`PGVECTOR_PARTITION_BUCKETS` and `PGVECTOR_PARTITION_DEDICATED_PATTERN` must
hold the same values here as they will when the backend runs: the partition key
is recomputed from the collection name on every read, so migrating under one
setting and serving under another sends reads to partitions that hold nothing.

Why this is a script and not an Alembic revision:

  - Alembic is bound to `DATABASE_URL` (`migrations/env.py`), but the vector
    store is `PGVECTOR_DB_URL`, which only defaults to it and is routinely set
    to a different database.
  - `run_migrations()` is called at import time of `config.py`, before the app
    serves anything. A multi-hour rewrite there means failing health checks and
    a restart part-way through.
  - Alembic runs for every install, including those on SQLite or using Chroma,
    Qdrant or Milvus, where this table does not exist.
  - `DocumentChunk` is declared on its own `declarative_base()`, so autogenerate
    never sees it -- which is why `document_chunk` appears in no revision today.

The rewrite is resumable: progress is recorded per collection and per batch, so
an interrupted run continues where it stopped.
"""

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

from open_webui.config import PGVECTOR_DB_URL, PGVECTOR_PARTITION_BUCKETS
from open_webui.retrieval.vector.dbs.pgvector import (
    BUCKET_PART_KEY_PREFIX,
    VECTOR_OPCLASS,
    is_dedicated_collection,
    part_key_for,
    partition_name_for,
    vector_index_configuration,
)
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

TABLE = 'document_chunk'
STAGING = 'document_chunk_partitioned'
BACKUP = 'document_chunk_pre_partition'
PROGRESS = 'document_chunk_migration_progress'
PART_MAP = 'document_chunk_migration_part_map'
DEFAULT_BATCH_SIZE = 10_000


def render_ddl(conn, fmt: str, *args: str) -> str:
    """Build DDL with PostgreSQL's own identifier and literal quoting."""
    placeholders = ', '.join(f':a{i}' for i in range(len(args)))
    params = {f'a{i}': value for i, value in enumerate(args)}
    return conn.execute(text(f'SELECT format(:fmt, {placeholders})'), {'fmt': fmt, **params}).scalar()


def relkind(conn, name: str) -> Optional[str]:
    return conn.execute(
        text('SELECT relkind FROM pg_class WHERE relname = :name AND relnamespace = current_schema()::regnamespace'),
        {'name': name},
    ).scalar()


def collection_stats(conn, table: str) -> List[Tuple[str, int]]:
    return [
        (row[0], row[1])
        for row in conn.execute(text(f'SELECT collection_name, count(*) FROM {table} GROUP BY 1 ORDER BY 2 DESC')).all()
    ]


def build_plan(stats: List[Tuple[str, int]]) -> Dict[str, object]:
    dedicated, bucketed = {}, {}
    for collection, rows in stats:
        target = dedicated if is_dedicated_collection(collection) else bucketed
        target[collection] = rows
    return {
        'collections': len(stats),
        'rows': sum(rows for _, rows in stats),
        'dedicated': dedicated,
        'bucketed': bucketed,
        'partitions': len(dedicated) + PGVECTOR_PARTITION_BUCKETS,
    }


def print_plan(plan: Dict[str, object]) -> None:
    dedicated, bucketed = plan['dedicated'], plan['bucketed']
    print(f'  collections           : {plan["collections"]}')
    print(f'  rows                  : {plan["rows"]}')
    print(f'  dedicated partitions  : {len(dedicated)} (one per knowledge base, each with its own indexes)')
    print(f'  buckets               : {PGVECTOR_PARTITION_BUCKETS} (btree on collection_name only)')
    print(f'  total partitions      : {plan["partitions"]}')
    print(f'  rows into buckets     : {sum(bucketed.values())} across {len(bucketed)} collections')
    print(f'  rows into partitions  : {sum(dedicated.values())}')
    if plan['partitions'] > 200:
        print(
            f'  WARNING: {plan["partitions"]} partitions. Each query locks the partitions it touches; '
            'review max_locks_per_transaction before going ahead.'
        )


def create_staging(conn) -> None:
    """Create the partitioned table and its buckets.

    `LIKE` copies the column definitions exactly, so halfvec columns and the
    bytea columns used when PGVECTOR_PGCRYPTO is on are carried over without
    this script having to know which variant is in use.
    """
    conn.execute(
        text(
            f'CREATE TABLE IF NOT EXISTS {STAGING} '
            f'(LIKE {TABLE}, part_key text NOT NULL, PRIMARY KEY (id, part_key)) '
            f'PARTITION BY LIST (part_key)'
        )
    )
    for bucket in range(PGVECTOR_PARTITION_BUCKETS):
        part_key = f'{BUCKET_PART_KEY_PREFIX}{bucket}'
        partition = partition_name_for(part_key)
        conn.execute(
            text(
                render_ddl(
                    conn,
                    f'CREATE TABLE IF NOT EXISTS %I PARTITION OF {STAGING} FOR VALUES IN (%L)',
                    partition,
                    part_key,
                )
            )
        )
        conn.execute(
            text(f'CREATE INDEX IF NOT EXISTS {partition}_collection_name_idx ON {partition} (collection_name)')
        )

    conn.execute(
        text(
            f'CREATE TABLE IF NOT EXISTS {PROGRESS} ('
            '  id int PRIMARY KEY,'
            '  last_id text,'
            '  copied bigint NOT NULL DEFAULT 0)'
        )
    )


def create_dedicated_partition(conn, collection: str) -> str:
    part_key = part_key_for(collection)
    partition = partition_name_for(part_key)
    conn.execute(
        text(
            render_ddl(
                conn,
                f'CREATE TABLE IF NOT EXISTS %I PARTITION OF {STAGING} FOR VALUES IN (%L)',
                partition,
                part_key,
            )
        )
    )
    return partition


def load_part_key_map(engine, collections: List[str]) -> None:
    """Materialise collection_name -> part_key so the copy can join against it.

    The partition key is a Python function of the collection name (crc32, for
    stability across PostgreSQL versions), and SQL cannot call it. Computing it
    once per distinct collection and joining keeps the per-row work in the
    server: a realistic table has millions of rows but only thousands of
    collections.
    """
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {PART_MAP}'))
        conn.execute(text(f'CREATE TABLE {PART_MAP} (collection_name text PRIMARY KEY, part_key text NOT NULL)'))

    rows = [{'c': c, 'p': part_key_for(c)} for c in collections]
    with engine.begin() as conn:
        for chunk in (rows[i : i + 5000] for i in range(0, len(rows), 5000)):
            conn.execute(text(f'INSERT INTO {PART_MAP} (collection_name, part_key) VALUES (:c, :p)'), chunk)
        conn.execute(text(f'ANALYZE {PART_MAP}'))


def copy_rows(engine, batch_size: int, total: int) -> int:
    """Copy every row in one keyset-paginated pass, resuming where it stopped.

    An earlier version copied collection by collection, which is fine for a few
    hundred but not for the shape this table actually takes: Open WebUI creates
    a collection per uploaded file, so a 5M-row table can hold 200k of them and
    the per-collection round trips cost more than the copy itself. One ordered
    pass with a hash join against the part-key map is O(rows) regardless of how
    many collections there are.
    """
    with engine.begin() as conn:
        row = conn.execute(text(f'SELECT last_id, copied FROM {PROGRESS} WHERE id = 1')).one_or_none()
        if row is None:
            conn.execute(text(f'INSERT INTO {PROGRESS} (id, copied) VALUES (1, 0)'))
            last_id, copied = None, 0
        else:
            last_id, copied = row.last_id, row.copied

    columns = None
    started = time.monotonic()
    while True:
        with engine.begin() as conn:
            if columns is None:
                columns = [
                    r[0]
                    for r in conn.execute(
                        text(
                            'SELECT attname FROM pg_attribute '
                            'WHERE attrelid = to_regclass(:t) AND attnum > 0 AND NOT attisdropped '
                            'ORDER BY attnum'
                        ),
                        {'t': TABLE},
                    ).all()
                ]
            qualified = ', '.join(f'c.{name}' for name in columns)
            column_list = ', '.join(columns)
            bound = 'WHERE c.id > :last_id' if last_id is not None else ''
            # The cursor advances on what the batch *read*, not on what the
            # insert returned. `ON CONFLICT DO NOTHING ... RETURNING` reports
            # only the rows it actually wrote, so a batch that conflicted
            # entirely would look like the end of the table and silently leave
            # the rest of it uncopied.
            batch = conn.execute(
                text(
                    f'WITH batch AS ('
                    f'  SELECT {qualified}, m.part_key FROM {TABLE} c '
                    f'  JOIN {PART_MAP} m ON m.collection_name = c.collection_name '
                    f'  {bound} ORDER BY c.id LIMIT :batch_size'
                    f'), inserted AS ('
                    f'  INSERT INTO {STAGING} ({column_list}, part_key) '
                    f'  SELECT {column_list}, part_key FROM batch '
                    f'  ON CONFLICT DO NOTHING RETURNING id'
                    f') '
                    f'SELECT (SELECT count(*) FROM batch) AS read, '
                    f'       (SELECT max(id) FROM batch) AS max_id, '
                    f'       (SELECT count(*) FROM inserted) AS written'
                ),
                {'batch_size': batch_size, **({'last_id': last_id} if last_id is not None else {})},
            ).one()

            if not batch.read:
                return copied

            last_id = batch.max_id
            copied += batch.written
            conn.execute(
                text(f'UPDATE {PROGRESS} SET last_id = :last_id, copied = :copied WHERE id = 1'),
                {'last_id': last_id, 'copied': copied},
            )

        elapsed = time.monotonic() - started
        rate = copied / elapsed if elapsed else 0
        print(f'  copied {copied}/{total} rows ({rate:.0f}/s)', flush=True)


def text_is_encrypted(conn) -> bool:
    """Whether the `text` column is bytea, as it is under PGVECTOR_PGCRYPTO.

    Read from the table rather than from configuration: the table is what the
    index has to fit, whatever the environment this script happens to run in.
    """
    column_type = conn.execute(
        text(
            'SELECT format_type(atttypid, atttypmod) FROM pg_attribute '
            "WHERE attrelid = to_regclass(:t) AND attname = 'text'"
        ),
        {'t': STAGING},
    ).scalar()
    return column_type == 'bytea'


def build_partition_indexes(engine, collections: List[str], work_mem: str = '2GB') -> None:
    """Index the dedicated partitions once they are loaded.

    Left until after the copy: building an index over a populated table once is
    far cheaper than maintaining it across millions of inserts.

    The full-text index is skipped over an encrypted column, as the backend
    itself skips it at startup: there is no `to_tsvector` over bytea, and hybrid
    search declines under pgcrypto anyway.
    """
    index_method, index_options = vector_index_configuration()
    with engine.connect() as conn:
        encrypted = text_is_encrypted(conn)
    for collection in collections:
        partition = partition_name_for(part_key_for(collection))
        with engine.begin() as conn:
            conn.execute(text(f"SET LOCAL maintenance_work_mem = '{work_mem}'"))
            conn.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS {partition}_vector_idx ON {partition} '
                    f'USING {index_method} (vector {VECTOR_OPCLASS}) {index_options}'
                )
            )
            if not encrypted:
                conn.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS {partition}_text_idx ON {partition} '
                        f"USING GIN (to_tsvector('simple', coalesce(text, '')))"
                    )
                )
        log.info('Indexed partition %s', partition)


def row_counts(engine) -> Tuple[int, int]:
    """Rows in the source table and in the staging table."""
    with engine.connect() as conn:
        source = conn.execute(text(f'SELECT count(*) FROM {TABLE}')).scalar()
        staged = conn.execute(text(f'SELECT count(*) FROM {STAGING}')).scalar()
    return source, staged


def swap(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {TABLE} RENAME TO {BACKUP}'))
        conn.execute(text(f'ALTER TABLE {STAGING} RENAME TO {TABLE}'))
    log.info("Swapped in the partitioned table. The original is kept as '%s'.", BACKUP)


def migrate(engine, batch_size: int, work_mem: str = '2GB') -> int:
    with engine.connect() as conn:
        kind = relkind(conn, TABLE)
        if kind is None:
            print(f"'{TABLE}' does not exist; nothing to migrate.")
            return 1
        if kind == 'p':
            print(f"'{TABLE}' is already partitioned; nothing to do.")
            return 0
        stats = collection_stats(conn, TABLE)

    plan = build_plan(stats)
    print('Migration plan:')
    print_plan(plan)

    with engine.connect() as conn:
        if relkind(conn, STAGING) is not None:
            print(
                f"\n'{STAGING}' already exists. A rolled-back migration leaves it behind, along with\n"
                'its partitions, whose names would collide with the ones this run needs.\n'
                f'Drop it first:  DROP TABLE {STAGING} CASCADE;'
            )
            return 1

    with engine.begin() as conn:
        create_staging(conn)

    started = time.monotonic()
    print(f'Creating {len(plan["dedicated"])} dedicated partitions')
    with engine.begin() as conn:
        for collection in plan['dedicated']:
            create_dedicated_partition(conn, collection)

    print(f'Mapping {len(stats)} collections to partition keys')
    load_part_key_map(engine, [collection for collection, _ in stats])

    total = copy_rows(engine, batch_size, plan['rows'])

    # The copy joins against a map of the collections that existed when the
    # plan was made. A row written to a new collection afterwards -- the app
    # was not stopped as instructed -- is silently left out of the copy, and
    # a swap would then lose it. Refuse before spending time on the indexes.
    source, staged = row_counts(engine)
    if staged != source:
        print(
            f"\n'{STAGING}' holds {staged} rows but '{TABLE}' holds {source}. Rows were written after "
            'the migration started, so the copy is not complete and the table is left untouched.\n'
            f'Stop Open WebUI, drop the staging table (DROP TABLE {STAGING} CASCADE;) and run again.'
        )
        return 1

    build_partition_indexes(engine, list(plan['dedicated']), work_mem)
    swap(engine)

    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {PROGRESS}'))
        conn.execute(text(f'DROP TABLE IF EXISTS {PART_MAP}'))

    elapsed = time.monotonic() - started
    print(f'\nMigrated {total} rows in {elapsed:.1f}s.')
    print(f"Set PGVECTOR_PARTITIONING=true and restart. The previous table is kept as '{BACKUP}';")
    print('drop it once you are satisfied, or use --rollback to go back.')
    return 0


def rollback(engine) -> int:
    with engine.connect() as conn:
        if relkind(conn, BACKUP) is None:
            print(f"No '{BACKUP}' table found; nothing to roll back to.")
            return 1
        if relkind(conn, TABLE) != 'p':
            print(f"'{TABLE}' is not partitioned; the rollback appears to have happened already.")
            return 1

    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {TABLE} RENAME TO {STAGING}'))
        conn.execute(text(f'ALTER TABLE {BACKUP} RENAME TO {TABLE}'))
    print(f"Restored the unpartitioned '{TABLE}'. Unset PGVECTOR_PARTITIONING and restart.")
    print(f"The partitioned copy is kept as '{STAGING}'; drop it when you no longer need it.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true', help='report what the migration would do, and change nothing')
    action.add_argument('--migrate', action='store_true', help='rewrite the table into the partitioned layout')
    action.add_argument('--rollback', action='store_true', help='restore the table saved by a previous --migrate')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='rows per committed batch')
    parser.add_argument('--db-url', default=PGVECTOR_DB_URL, help='defaults to PGVECTOR_DB_URL')
    parser.add_argument('--work-mem', default='2GB', help='maintenance_work_mem for building partition indexes')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    engine = create_engine(args.db_url)
    try:
        if args.plan:
            with engine.connect() as conn:
                if relkind(conn, TABLE) is None:
                    print(f"'{TABLE}' does not exist.")
                    return 1
                print_plan(build_plan(collection_stats(conn, TABLE)))
            return 0
        if args.migrate:
            return migrate(engine, args.batch_size, args.work_mem)
        return rollback(engine)
    finally:
        engine.dispose()


if __name__ == '__main__':
    sys.exit(main())
