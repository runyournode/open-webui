"""Operational edges of the partitioned layout.

These cover the failure modes that are silent rather than loud: a partition
created twice, a flag flipped without migrating, a bucket count lowered after
rows were written. Each of those either corrupts routing or strands data, and
none of them announce themselves.
"""

import threading
import uuid

import pytest
from sqlalchemy import text

from tests.conftest import TEST_VECTOR_LENGTH
from tests.vector.conftest import TEST_BUCKETS

KB = '0a1b2c3d-4e5f-4a6b-8c7d-0000000000ff'
# A bucketed collection, to check an operation scoped to one collection does not
# reach into the partition that holds all the others.
BUCKETED = ['file-0a1b2c3d-4e5f-4a6b-8c7d-0000000000fe']


def item(index: int = 0, text_value: str = 'chunk'):
    vector = [0.0] * TEST_VECTOR_LENGTH
    vector[index % TEST_VECTOR_LENGTH] = 1.0
    return {'id': str(uuid.uuid4()), 'text': text_value, 'vector': vector, 'metadata': {'index': index}}


def test_concurrent_workers_creating_one_partition(pgvector_module, mode):
    """Several workers ingesting into a brand-new knowledge base at once.

    Each client keeps its own cache, so they all believe they must create the
    partition. `IF NOT EXISTS` alone does not survive this -- the existence
    check and the create are not atomic -- which is why the DDL takes an
    advisory lock and still treats duplicate_table as success.
    """
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')

    workers = 8
    clients = [pgvector_module.PgvectorClient() for _ in range(workers)]
    barrier = threading.Barrier(workers)
    failures = []

    def ingest(client):
        barrier.wait()
        try:
            client.insert(KB, [item()])
        except Exception as e:  # noqa: BLE001 - the assertion is that there are none
            failures.append(e)

    threads = [threading.Thread(target=ingest, args=(c,)) for c in clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, f'concurrent partition creation failed: {failures}'
    assert len(clients[0].get(KB).ids[0]) == workers, 'every worker should have written exactly one row'


def test_concurrent_startup_creates_buckets_once(pgvector_module, mode):
    """Several workers starting at the same time against a table with no buckets.

    `UVICORN_WORKERS` starts that many processes, and each one runs
    `PgvectorClient.__init__`, which creates the bucket partitions with
    `CREATE TABLE IF NOT EXISTS ... PARTITION OF`. The existence check and the
    create are not atomic, so two workers can both decide the bucket is missing
    and one of them then fails its startup with duplicate_table. The dedicated
    partitions already take an advisory lock for exactly this reason; the
    buckets have to as well.

    One client is created first so the parent table exists: its creation races
    in the same way on `dev` today, and that is not what this test is about.
    """
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')

    first = pgvector_module.PgvectorClient()
    for bucket in range(TEST_BUCKETS):
        first.session.execute(text(f'DROP TABLE IF EXISTS document_chunk_b{bucket}'))
    first.session.commit()

    workers = 8
    barrier = threading.Barrier(workers)
    failures = []

    def start():
        barrier.wait()
        try:
            pgvector_module.PgvectorClient()
        except Exception as e:  # noqa: BLE001 - the assertion is that there are none
            failures.append(e)

    threads = [threading.Thread(target=start) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, f'concurrent startup failed: {failures}'
    buckets = first.session.execute(
        text(
            'SELECT count(*) FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid '
            "WHERE i.inhparent = to_regclass('document_chunk') AND c.relname LIKE 'document_chunk_b%'"
        )
    ).scalar()
    first.session.rollback()
    assert buckets == TEST_BUCKETS


def test_enabling_the_flag_without_migrating_is_refused(reload_pgvector, drop_schema):
    """create_all(checkfirst=True) skips an existing table, so without an explicit
    guard the flag would appear to work and then fail on every query."""
    drop_schema()
    unpartitioned = reload_pgvector(partitioning=False)
    unpartitioned.PgvectorClient()

    partitioned = reload_pgvector(partitioning=True)
    with pytest.raises(RuntimeError, match='not a partitioned table'):
        partitioned.PgvectorClient()


def test_disabling_the_flag_without_rolling_back_is_refused(reload_pgvector, drop_schema):
    drop_schema()
    partitioned = reload_pgvector(partitioning=True)
    partitioned.PgvectorClient()

    unpartitioned = reload_pgvector(partitioning=False)
    with pytest.raises(RuntimeError, match='PGVECTOR_PARTITIONING is not enabled'):
        unpartitioned.PgvectorClient()


def test_lowering_the_bucket_count_is_refused(reload_pgvector, drop_schema):
    """Rows hash into buckets by count. Lowering it leaves the rows in the
    removed buckets unreachable, with no error at read time."""
    drop_schema()
    wide = reload_pgvector(partitioning=True, PGVECTOR_PARTITION_BUCKETS=TEST_BUCKETS)
    wide.PgvectorClient()

    narrow = reload_pgvector(partitioning=True, PGVECTOR_PARTITION_BUCKETS=TEST_BUCKETS - 2)
    with pytest.raises(RuntimeError, match='beyond the configured'):
        narrow.PgvectorClient()


def test_raising_the_bucket_count_adds_partitions(reload_pgvector, drop_schema):
    """Raising it is safe for the schema, though existing rows keep their old
    bucket until they are moved."""
    drop_schema()
    narrow = reload_pgvector(partitioning=True, PGVECTOR_PARTITION_BUCKETS=TEST_BUCKETS)
    narrow.PgvectorClient()

    wide = reload_pgvector(partitioning=True, PGVECTOR_PARTITION_BUCKETS=TEST_BUCKETS + 2)
    client = wide.PgvectorClient()

    buckets = client.session.execute(
        text(
            'SELECT count(*) FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid '
            "WHERE i.inhparent = to_regclass('document_chunk') AND c.relname LIKE 'document_chunk_b%'"
        )
    ).scalar()
    assert buckets == TEST_BUCKETS + 2


def test_an_invalid_dedicated_pattern_falls_back_to_bucketing(reload_pgvector, drop_schema):
    """A bad regex must not produce a partition per collection."""
    drop_schema()
    module = reload_pgvector(partitioning=True, PGVECTOR_PARTITION_DEDICATED_PATTERN='([unclosed')

    assert not module.is_dedicated_collection(KB)
    assert module.part_key_for(KB).startswith(module.BUCKET_PART_KEY_PREFIX)


@pytest.mark.parametrize('partitioning', [False, True])
def test_pgcrypto_round_trip(reload_pgvector, drop_schema, partitioning):
    """The encrypted path uses raw SQL with an ON CONFLICT target, which has to
    name the composite primary key when the table is partitioned."""
    drop_schema()
    module = reload_pgvector(
        partitioning=partitioning,
        PGVECTOR_PGCRYPTO='true',
        PGVECTOR_PGCRYPTO_KEY='test-key',
    )
    client = module.PgvectorClient()

    first = item(0, 'secret one')
    client.insert(KB, [first])
    assert client.get(KB).documents[0] == ['secret one']

    # Same id again: insert must not overwrite, upsert must.
    client.insert(KB, [dict(first, text='ignored')])
    assert client.get(KB).documents[0] == ['secret one']

    client.upsert(KB, [dict(first, text='secret two')])
    assert client.get(KB).documents[0] == ['secret two']

    # Restore the default for the tests that follow.
    reload_pgvector(partitioning=False, PGVECTOR_PGCRYPTO='false')


@pytest.mark.parametrize('partitioning', [False, True])
def test_pgcrypto_read_and_delete_paths(reload_pgvector, drop_schema, partitioning):
    """The encrypted read and delete paths, which the round-trip test does not reach.

    Under pgcrypto these are separate branches in `search`, `query`, `delete` and
    `has_collection`, each building its own WHERE clause -- and each therefore
    needing the partition key added independently. Covering only insert/upsert/get
    left the branches that partitioning actually had to modify untested.
    """
    drop_schema()
    module = reload_pgvector(
        partitioning=partitioning,
        PGVECTOR_PGCRYPTO='true',
        PGVECTOR_PGCRYPTO_KEY='test-key',
    )
    client = module.PgvectorClient()

    kept, dropped = item(0, 'keep me'), item(1, 'drop me')
    kept['metadata']['tag'], dropped['metadata']['tag'] = 'keep', 'drop'
    client.insert(KB, [kept, dropped])
    client.insert(BUCKETED[0], [item(2, 'in a bucket')])

    # search: decrypts text and metadata, and must stay scoped to its collection
    found = client.search(KB, vectors=[item(0)['vector']], limit=10)
    assert found is not None
    assert sorted(found.documents[0]) == ['drop me', 'keep me']

    # query: metadata filter applied after decryption
    filtered = client.query(KB, filter={'tag': 'keep'})
    assert filtered.documents[0] == ['keep me']

    # hybrid_search must still decline under pgcrypto, exactly as upstream does
    assert client.hybrid_search(KB, query='keep', vectors=[item(0)['vector']], limit=5) is None

    # delete by metadata filter, then the collection, without touching the bucket
    client.delete(KB, filter={'tag': 'drop'})
    assert client.get(KB).documents[0] == ['keep me']

    client.delete_collection(KB)
    assert client.has_collection(KB) is False
    assert client.has_collection(BUCKETED[0]) is True

    reload_pgvector(partitioning=False, PGVECTOR_PGCRYPTO='false')


def test_write_recovers_when_another_worker_dropped_the_partition(pgvector_module, mode):
    """One worker drops a knowledge base while another still has it cached.

    Reindexing a knowledge base deletes the collection and writes it again, so
    on a multi-worker install this is a normal sequence, not an edge case. The
    write must re-create the partition rather than failing the ingestion.
    """
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')

    writer = pgvector_module.PgvectorClient()
    dropper = pgvector_module.PgvectorClient()

    writer.insert(KB, [item(0, 'before')])
    # `dropper` removes the partition; `writer` still believes it exists.
    dropper.delete_collection(KB)
    assert pgvector_module.part_key_for(KB) in writer._known_part_keys

    writer.insert(KB, [item(1, 'after')])
    assert writer.get(KB).documents[0] == ['after']


def test_a_real_check_violation_is_not_mistaken_for_a_missing_partition(pgvector_module, mode):
    """The two share SQLSTATE 23514, and the message is translated under NLS,
    so they are told apart by the diagnostic fields instead."""
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')

    client = pgvector_module.PgvectorClient()
    # The partition only exists once something has been written to it, and the
    # constraint has to hold for the rows already there, so it forbids a value
    # only the second insert uses.
    client.insert(KB, [item(0, 'allowed')])
    partition = pgvector_module.partition_name_for(pgvector_module.part_key_for(KB))
    client.session.execute(text(f"ALTER TABLE {partition} ADD CONSTRAINT no_forbidden CHECK (text <> 'forbidden')"))
    client.session.commit()

    with pytest.raises(Exception) as caught:
        client.insert(KB, [item(1, 'forbidden')])
    assert not pgvector_module.PgvectorClient._is_missing_partition(caught.value), (
        'a genuine CHECK violation must not be retried as a dropped partition'
    )
