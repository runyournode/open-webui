"""The migration from the single-table layout to the partitioned one, and back.

The rewrite has to be lossless in both directions: every row readable before is
readable after, through the same public API, and a rollback restores the
original table untouched.
"""

import importlib
import uuid

import pytest
from sqlalchemy import text

from tests.conftest import PGVECTOR_TEST_DB_URL, TEST_VECTOR_LENGTH
from tests.vector.conftest import TEST_BUCKETS

KB_A = '0a1b2c3d-4e5f-4a6b-8c7d-00000000ab01'
KB_B = '0a1b2c3d-4e5f-4a6b-8c7d-00000000ab02'
BUCKETED = [
    'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000cd01',
    'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000cd02',
    'user-memory-someone',
    'knowledge-bases',
]


def item(index: int, text_value: str):
    vector = [0.0] * TEST_VECTOR_LENGTH
    vector[index % TEST_VECTOR_LENGTH] = 1.0
    return {'id': str(uuid.uuid4()), 'text': text_value, 'vector': vector, 'metadata': {'index': index}}


def snapshot(client, collections):
    """Documents per collection, order-independent."""
    return {c: sorted(client.get(c).documents[0]) if client.has_collection(c) else [] for c in collections}


def load_migration_module():
    import open_webui.retrieval.vector.dbs.pgvector_partition_migrate as migrate

    return importlib.reload(migrate)


@pytest.fixture
def populated(reload_pgvector, drop_schema):
    """An unpartitioned table holding every collection shape, as an upgrade finds it."""
    drop_schema()
    module = reload_pgvector(partitioning=False)
    client = module.PgvectorClient()

    for n, collection in enumerate([KB_A, KB_B] + BUCKETED):
        client.insert(collection, [item(i, f'{collection} chunk {i}') for i in range(n + 2)])

    collections = [KB_A, KB_B] + BUCKETED
    return client, collections, snapshot(client, collections)


def test_plan_reports_the_split(populated, capsys):
    _, _, _ = populated
    migrate = load_migration_module()

    assert migrate.main(['--plan', '--db-url', PGVECTOR_TEST_DB_URL]) == 0
    out = capsys.readouterr().out
    assert 'dedicated partitions  : 3' in out, 'two knowledge bases plus the knowledge-bases meta-collection'
    assert f'buckets               : {TEST_BUCKETS}' in out


def test_migration_preserves_every_row(populated, reload_pgvector):
    _, collections, before = populated

    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    partitioned = reload_pgvector(partitioning=True)
    client = partitioned.PgvectorClient()
    assert snapshot(client, collections) == before


def test_migration_produces_the_expected_layout(populated, reload_pgvector):
    _, _, _ = populated
    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    partitioned = reload_pgvector(partitioning=True)
    client = partitioned.PgvectorClient()

    kb_partition = partitioned.partition_name_for(partitioned.part_key_for(KB_A))
    defs = ' '.join(
        client.session.execute(
            text('SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = :t'),
            {'t': kb_partition},
        )
        .scalars()
        .all()
    )
    assert 'USING hnsw' in defs or 'USING ivfflat' in defs
    assert 'USING gin' in defs

    bucket_defs = ' '.join(
        client.session.execute(
            text('SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = :t'),
            {'t': 'document_chunk_b0'},
        )
        .scalars()
        .all()
    )
    assert 'USING hnsw' not in bucket_defs and 'USING ivfflat' not in bucket_defs


def test_migration_keeps_the_original_table(populated, reload_pgvector):
    _, _, _ = populated
    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    partitioned = reload_pgvector(partitioning=True)
    client = partitioned.PgvectorClient()
    kept = client.session.execute(
        text("SELECT relkind FROM pg_class WHERE relname = 'document_chunk_pre_partition'")
    ).scalar()
    assert kept == 'r', 'the untouched original must survive the migration for rollback'


def test_migration_is_idempotent(populated, reload_pgvector):
    _, collections, before = populated
    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0, 'a second run should be a no-op'

    partitioned = reload_pgvector(partitioning=True)
    assert snapshot(partitioned.PgvectorClient(), collections) == before


def test_rollback_restores_the_original(populated, reload_pgvector):
    _, collections, before = populated
    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0
    assert migrate.main(['--rollback', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    unpartitioned = reload_pgvector(partitioning=False)
    client = unpartitioned.PgvectorClient()
    assert snapshot(client, collections) == before

    relkind = client.session.execute(text("SELECT relkind FROM pg_class WHERE relname = 'document_chunk'")).scalar()
    assert relkind == 'r'


def test_the_backend_works_after_migrating(populated, reload_pgvector):
    _, _, _ = populated
    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    partitioned = reload_pgvector(partitioning=True)
    client = partitioned.PgvectorClient()

    new_kb = '0a1b2c3d-4e5f-4a6b-8c7d-00000000ff99'
    client.insert(new_kb, [item(0, 'written after the migration')])
    assert client.get(new_kb).documents[0] == ['written after the migration']

    client.insert(KB_A, [item(1, 'appended to a migrated collection')])
    assert 'appended to a migrated collection' in client.get(KB_A).documents[0]


def test_migration_under_pgcrypto(reload_pgvector, drop_schema):
    """An encrypted table migrates too, and ends up without a text index.

    Under PGVECTOR_PGCRYPTO the `text` column is bytea, so the GIN index the
    backend builds for hybrid search cannot exist -- the runtime path skips it.
    The migration built it unconditionally and failed after copying every row,
    leaving the staging table behind and nothing to say why.
    """
    drop_schema()
    module = reload_pgvector(partitioning=False, PGVECTOR_PGCRYPTO='true', PGVECTOR_PGCRYPTO_KEY='test-key')
    client = module.PgvectorClient()
    collections = [KB_A, KB_B, BUCKETED[0]]
    for n, collection in enumerate(collections):
        client.insert(collection, [item(i, f'{collection} secret {i}') for i in range(n + 2)])
    before = snapshot(client, collections)

    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 0

    partitioned = reload_pgvector(partitioning=True, PGVECTOR_PGCRYPTO='true', PGVECTOR_PGCRYPTO_KEY='test-key')
    client = partitioned.PgvectorClient()
    assert snapshot(client, collections) == before

    kb_partition = partitioned.partition_name_for(partitioned.part_key_for(KB_A))
    defs = ' '.join(
        client.session.execute(
            text('SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = :t'),
            {'t': kb_partition},
        )
        .scalars()
        .all()
    )
    assert 'USING hnsw' in defs or 'USING ivfflat' in defs, 'the vector index is still wanted'
    assert 'USING gin' not in defs, 'no full-text index can exist over an encrypted column'

    reload_pgvector(partitioning=False, PGVECTOR_PGCRYPTO='false')


def test_migration_refuses_to_swap_when_rows_are_missing(reload_pgvector, drop_schema, monkeypatch):
    """A row written after the plan was made must stop the migration, not vanish.

    The copy joins against the collections known when the plan was computed.
    The script says to stop the application first; if that instruction is not
    followed, a collection created in between is not in the map, its rows are
    never copied, and a swap would drop them. Refusing is the only safe answer.
    """
    drop_schema()
    module = reload_pgvector(partitioning=False)
    client = module.PgvectorClient()
    client.insert(KB_A, [item(i, f'row {i}') for i in range(10)])

    migrate = load_migration_module()
    original = migrate.load_part_key_map

    def write_a_late_row_then_map(engine, collections):
        client.insert(KB_B, [item(0, 'written after the plan was made')])
        original(engine, collections)

    monkeypatch.setattr(migrate, 'load_part_key_map', write_a_late_row_then_map)
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL]) == 1

    relkind = client.session.execute(text("SELECT relkind FROM pg_class WHERE relname = 'document_chunk'")).scalar()
    client.session.rollback()
    assert relkind == 'r', 'the table must be left untouched when the copy is incomplete'
    assert client.has_collection(KB_B), 'the late row is still where it was written'


# --------------------------------------------------------------------------
# The copy loop itself
#
# The fixture above holds 27 rows against a 10 000-row default batch, so the
# loop runs exactly once and returns. Keyset pagination, the resume cursor and
# the "advance on rows read, not rows written" behaviour were all unexercised.
# --------------------------------------------------------------------------


def _rows_in(engine, table):
    with engine.connect() as conn:
        return (
            conn.execute(text(f'SELECT count(*) FROM {table}')).scalar(),
            conn.execute(text(f'SELECT count(DISTINCT id) FROM {table}')).scalar(),
        )


def test_migration_copies_in_many_batches(reload_pgvector, drop_schema):
    """A batch size far below the row count, so the loop must page."""
    drop_schema()
    module = reload_pgvector(partitioning=False)
    client = module.PgvectorClient()

    rows = [item(i, f'row {i}') for i in range(300)]
    client.insert(KB_A, rows[:150])
    client.insert(BUCKETED[0], rows[150:])

    migrate = load_migration_module()
    assert migrate.main(['--migrate', '--db-url', PGVECTOR_TEST_DB_URL, '--batch-size', '25']) == 0

    partitioned = reload_pgvector(partitioning=True)
    client = partitioned.PgvectorClient()
    assert len(client.get(KB_A).ids[0]) == 150
    assert len(client.get(BUCKETED[0]).ids[0]) == 150


def test_migration_resumes_from_its_cursor(reload_pgvector, drop_schema):
    """An interrupted copy must continue rather than restart or skip.

    The cursor is seeded by hand to the state a crash would have left, which is
    the only way to reach the resume path deterministically.
    """
    from sqlalchemy import create_engine

    drop_schema()
    module = reload_pgvector(partitioning=False)
    client = module.PgvectorClient()
    rows = [item(i, f'row {i}') for i in range(200)]
    client.insert(KB_A, rows)

    migrate = load_migration_module()
    engine = create_engine(PGVECTOR_TEST_DB_URL)
    try:
        with engine.begin() as conn:
            migrate.create_staging(conn)
            migrate.create_dedicated_partition(conn, KB_A)
        migrate.load_part_key_map(engine, [KB_A])

        ids = sorted(r['id'] for r in rows)
        midpoint = ids[len(ids) // 2]
        remaining = len([i for i in ids if i > midpoint])
        with engine.begin() as conn:
            conn.execute(
                text(f'INSERT INTO {migrate.PROGRESS} (id, last_id, copied) VALUES (1, :last, 0)'),
                {'last': midpoint},
            )

        migrate.copy_rows(engine, batch_size=25, total=len(rows))
        staged, _ = _rows_in(engine, migrate.STAGING)
        assert staged == remaining, 'resume must copy only what the cursor says is outstanding'

        # Rewind the cursor and run again: ON CONFLICT DO NOTHING has to make a
        # replayed range idempotent, which is what makes a crash recoverable.
        with engine.begin() as conn:
            conn.execute(text(f'UPDATE {migrate.PROGRESS} SET last_id = NULL WHERE id = 1'))
        migrate.copy_rows(engine, batch_size=25, total=len(rows))

        staged, distinct = _rows_in(engine, migrate.STAGING)
        assert staged == len(rows), 'a full replay must end with every row present'
        assert distinct == staged, 'a replayed range must not duplicate rows'
    finally:
        engine.dispose()


def test_copy_advances_on_rows_read_not_rows_written(reload_pgvector, drop_schema):
    """A batch whose rows all conflict must not look like the end of the table.

    `INSERT ... ON CONFLICT DO NOTHING ... RETURNING` reports only the rows it
    wrote, so paging on that would stop the copy early and silently leave the
    rest behind. Pre-seeding the staging table with the first rows reproduces
    exactly that.
    """
    from sqlalchemy import create_engine

    drop_schema()
    module = reload_pgvector(partitioning=False)
    client = module.PgvectorClient()
    rows = [item(i, f'row {i}') for i in range(120)]
    client.insert(KB_A, rows)

    migrate = load_migration_module()
    engine = create_engine(PGVECTOR_TEST_DB_URL)
    try:
        with engine.begin() as conn:
            migrate.create_staging(conn)
            migrate.create_dedicated_partition(conn, KB_A)
        migrate.load_part_key_map(engine, [KB_A])

        # Copy the first batch twice over: the second pass sees it as conflicts.
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO {migrate.STAGING} (id, vector, collection_name, text, vmetadata, part_key) '
                    f'SELECT c.id, c.vector, c.collection_name, c.text, c.vmetadata, m.part_key '
                    f'FROM {migrate.TABLE} c JOIN {migrate.PART_MAP} m USING (collection_name) '
                    f'ORDER BY c.id LIMIT 40'
                )
            )

        migrate.copy_rows(engine, batch_size=20, total=len(rows))

        staged, distinct = _rows_in(engine, migrate.STAGING)
        assert staged == len(rows), 'the copy must not stop at a fully-conflicting batch'
        assert distinct == staged
    finally:
        engine.dispose()
