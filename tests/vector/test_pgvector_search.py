"""Search paths that had no coverage: bucketed collections, multiple query
vectors, and metadata filters.

The bucketed collections matter most. Under partitioning they keep only a btree
on `collection_name` -- no vector index, no full-text index -- and they hold the
majority of rows in a real install. Nothing here asserted that searching them
still worked, which is precisely where a regression would have gone unnoticed.
"""

import uuid

import pytest
from sqlalchemy import text

from tests.conftest import TEST_VECTOR_LENGTH
from tests.vector.conftest import scanned_relations

KB = '0a1b2c3d-4e5f-4a6b-8c7d-00000000aa01'
FILE_A = 'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000bb01'
FILE_B = 'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000bb02'
MEMORY = 'user-memory-00000000bb03'
WEB_SEARCH = 'web-search-u1-00000000bb04'

BUCKETED = [FILE_A, MEMORY, WEB_SEARCH]


def unit_vector(axis: int):
    """A vector along one axis: nearest-neighbour order is then obvious by
    construction, so the assertions do not depend on the index implementation."""
    vector = [0.0] * TEST_VECTOR_LENGTH
    vector[axis % TEST_VECTOR_LENGTH] = 1.0
    return vector


def item(axis: int, text_value: str = None, **metadata):
    return {
        'id': str(uuid.uuid4()),
        'text': text_value if text_value is not None else f'chunk on axis {axis}',
        'vector': unit_vector(axis),
        'metadata': {'axis': axis, **metadata},
    }


# --------------------------------------------------------------------------
# Bucketed collections
# --------------------------------------------------------------------------


@pytest.mark.parametrize('collection', BUCKETED)
def test_search_works_on_a_bucketed_collection(client, collection):
    """Buckets carry no vector index, so this is an exact scan over a handful of
    rows. It must still return the right answer."""
    items = [item(axis) for axis in range(TEST_VECTOR_LENGTH)]
    client.insert(collection, items)

    result = client.search(collection, vectors=[unit_vector(3)], limit=1)
    assert result is not None
    assert result.ids[0][0] == items[3]['id']


@pytest.mark.parametrize('collection', BUCKETED)
def test_bucket_search_is_exact(client, collection):
    """Without an approximate index there is nothing to be approximate about:
    the ranking must match the true distance order exactly."""
    items = [item(axis) for axis in range(TEST_VECTOR_LENGTH)]
    client.insert(collection, items)

    result = client.search(collection, vectors=[unit_vector(0)], limit=TEST_VECTOR_LENGTH)
    assert result.ids[0][0] == items[0]['id'], 'the identical vector must rank first'
    assert len(result.ids[0]) == TEST_VECTOR_LENGTH
    # Distances are returned normalised to [0, 1] and must be non-increasing.
    scores = result.distances[0]
    assert scores == sorted(scores, reverse=True)


def test_bucketed_collections_sharing_a_partition_do_not_leak(client, pgvector_module):
    """Two collections in one bucket are separated only by collection_name."""
    colliding, seen = None, {}
    for i in range(500):
        name = f'file-leak-{i}'
        key = pgvector_module.part_key_for(name)
        if key in seen:
            colliding = (seen[key], name)
            break
        seen[key] = name
    assert colliding, 'expected two collections to hash into the same bucket'

    client.insert(colliding[0], [item(0, 'first')])
    client.insert(colliding[1], [item(0, 'second')])

    result = client.search(colliding[0], vectors=[unit_vector(0)], limit=10)
    assert result.documents[0] == ['first']


@pytest.mark.parametrize('collection', BUCKETED)
def test_hybrid_search_works_on_a_bucketed_collection(client, collection):
    """Buckets have no GIN index either, so the lexical half runs unindexed. It
    still has to return correct results."""
    client.insert(
        collection,
        [
            item(0, 'the quick brown fox jumps over the lazy dog'),
            item(1, 'entirely unrelated filler'),
            item(2, 'more unrelated filler'),
        ],
    )

    result = client.hybrid_search(collection, query='quick brown fox', vectors=[unit_vector(0)], limit=3)
    assert result is not None, 'native hybrid search must remain available for bucketed collections'
    assert len(result.ids[0]) >= 1


def test_bucket_search_prunes_to_one_partition(client, pgvector_module, mode):
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    for collection in (FILE_A, FILE_B, MEMORY, WEB_SEARCH, KB):
        client.insert(collection, [item(0)])

    part_key = pgvector_module.part_key_for(FILE_A)
    scanned = scanned_relations(
        client,
        'SELECT id FROM document_chunk WHERE collection_name = :c AND part_key = :p',
        {'c': FILE_A, 'p': part_key},
    )
    expected = pgvector_module.partition_name_for(part_key)
    assert scanned == {expected}, f'expected only {expected} to be scanned, got {sorted(scanned)}'


# --------------------------------------------------------------------------
# Multiple query vectors
# --------------------------------------------------------------------------


def test_search_with_multiple_query_vectors(client):
    """`query_collection` searches with several query vectors at once. The
    lateral join fans out per vector and the results are grouped by `qid`;
    nothing previously exercised more than one."""
    items = [item(axis) for axis in range(TEST_VECTOR_LENGTH)]
    client.insert(KB, items)

    axes = [0, 3, 5]
    result = client.search(KB, vectors=[unit_vector(a) for a in axes], limit=2)

    assert len(result.ids) == len(axes), 'one result list per query vector'
    for position, axis in enumerate(axes):
        assert result.ids[position][0] == items[axis]['id'], f'query {position} should match its own axis'
        assert len(result.ids[position]) == 2


def test_multiple_query_vectors_on_a_bucketed_collection(client):
    items = [item(axis) for axis in range(TEST_VECTOR_LENGTH)]
    client.insert(FILE_A, items)

    result = client.search(FILE_A, vectors=[unit_vector(1), unit_vector(4)], limit=1)
    assert [result.ids[0][0], result.ids[1][0]] == [items[1]['id'], items[4]['id']]


# --------------------------------------------------------------------------
# Metadata filters, and hybrid-search parity with upstream
# --------------------------------------------------------------------------


@pytest.mark.parametrize('collection', [KB, FILE_A])
def test_search_applies_a_metadata_filter(client, collection):
    """The post-filter path. Partitioning adds a part_key predicate alongside
    the metadata one, so this checks the two compose correctly."""
    client.insert(collection, [item(0, 'keep me', tag='keep'), item(1, 'drop me', tag='drop')])

    result = client.search(collection, vectors=[unit_vector(1)], filter={'tag': 'keep'}, limit=10)
    assert result is not None
    assert result.documents[0] == ['keep me']


def test_search_filter_supports_in_operator(client):
    client.insert(KB, [item(0, 'a', tag='x'), item(1, 'b', tag='y'), item(2, 'c', tag='z')])

    result = client.search(KB, vectors=[unit_vector(0)], filter={'tag': {'$in': ['x', 'z']}}, limit=10)
    assert sorted(result.documents[0]) == ['a', 'c']


def test_hybrid_search_declines_exactly_when_upstream_does(client):
    """Parity, not new behaviour.

    Upstream returns None from hybrid_search when a metadata filter is passed
    (or pgcrypto is on), which is what drops Open WebUI onto its Python BM25
    fallback. Partitioning must not change *when* native hybrid search engages
    -- neither narrowing it (a regression) nor widening it (a behaviour change
    nobody asked for).
    """
    client.insert(KB, [item(0, 'the quick brown fox', tag='keep')])

    without_filter = client.hybrid_search(KB, query='quick fox', vectors=[unit_vector(0)], limit=5)
    assert without_filter is not None, 'native hybrid search must still engage when no filter is passed'

    with_filter = client.hybrid_search(KB, query='quick fox', vectors=[unit_vector(0)], filter={'tag': 'keep'}, limit=5)
    assert with_filter is None, 'a metadata filter must still decline to the caller, as upstream does'


# --------------------------------------------------------------------------
# Iterative index scans
# --------------------------------------------------------------------------


def test_iterative_scan_is_off_by_default(client, pgvector_module):
    """Nothing changes for an install that did not ask for it.

    Read through pg_settings rather than SHOW: the parameter is registered only
    once pgvector's library loads into the backend, and SHOW on an unregistered
    one raises, which would put an ERROR in the server log for a case that is
    working exactly as intended. Either `None` (never registered) or `off`
    proves the helper did nothing.
    """
    method = pgvector_module.vector_index_configuration()[0]
    client.session.rollback()
    client._apply_iterative_scan()
    applied = client.session.execute(
        text('SELECT setting FROM pg_settings WHERE name = :name'),
        {'name': f'{method}.iterative_scan'},
    ).scalar()
    client.session.rollback()

    assert applied in (None, 'off'), f'expected no iterative scan by default, found {applied!r}'


def test_iterative_scan_is_applied_when_enabled(reload_pgvector, drop_schema):
    """The GUC has to reach the session, with the mode the method supports.

    Checked on the helper rather than after `search()`, because `SET LOCAL` is
    scoped to the transaction and `search()` rolls back when it is done -- so by
    the time a caller could look, the setting is gone. It is in force for the
    query itself, which is what matters.
    """
    drop_schema()
    module = reload_pgvector(partitioning=False, PGVECTOR_ITERATIVE_SCAN='relaxed_order')
    client = module.PgvectorClient()
    method = module.vector_index_configuration()[0]

    client.session.rollback()
    client._apply_iterative_scan()
    applied = client.session.execute(
        text('SELECT setting FROM pg_settings WHERE name = :name'),
        {'name': f'{method}.iterative_scan'},
    ).scalar()
    client.session.rollback()

    assert applied in ('relaxed_order', 'strict_order'), f'expected an iterative scan mode, got {applied!r}'
    if method == 'ivfflat':
        assert applied == 'relaxed_order', 'ivfflat has no strict_order'


def test_search_still_works_with_iterative_scan(reload_pgvector, drop_schema):
    """A regression guard: the GUC must not break the search path itself."""
    drop_schema()
    module = reload_pgvector(partitioning=False, PGVECTOR_ITERATIVE_SCAN='relaxed_order')
    client = module.PgvectorClient()

    items = [item(axis) for axis in range(TEST_VECTOR_LENGTH)]
    client.insert(KB, items)
    result = client.search(KB, vectors=[unit_vector(3)], limit=1)
    assert result.ids[0][0] == items[3]['id']


def test_empty_result_returns_the_connection_to_the_pool(client):
    """A read that finds nothing must still end its transaction.

    `search`, `query` and `get` each returned early on an empty result set
    without the rollback their populated paths take, so the session kept its
    transaction -- and with it a pooled connection. One thread reusing one
    session never notices. `query_collection` fans out over threads, giving each
    its own session, and a thread that ends still holding a connection never
    gives it back: fifteen empty searches and the backend stops answering.

    An empty result is not an exotic case. It is what a filtered index scan
    returns when none of the candidates it walked belong to the collection being
    searched -- the behaviour this whole suite exists to measure.
    """
    pool = client.session.get_bind().pool
    missing = f'file-{uuid.uuid4()}'

    client.search(collection_name=missing, vectors=[unit_vector(0)], limit=10)
    assert pool.checkedout() == 0, 'search() kept its connection after an empty result'

    client.query(collection_name=missing, filter={'axis': 0})
    assert pool.checkedout() == 0, 'query() kept its connection after an empty result'

    client.get(collection_name=missing)
    assert pool.checkedout() == 0, 'get() kept its connection after an empty result'
