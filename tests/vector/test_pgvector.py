"""Behaviour of the pgvector backend, asserted identically in both storage layouts.

Every test here runs twice: once against the single-table layout and once
against the partitioned one. That is the point -- partitioning is meant to
change where rows live and nothing a caller can observe.
"""

import uuid

import pytest
from sqlalchemy import text

from tests.conftest import TEST_VECTOR_LENGTH
from tests.vector.conftest import TEST_BUCKETS, scanned_relations

KB_A = '0a1b2c3d-4e5f-4a6b-8c7d-000000000001'
KB_B = '0a1b2c3d-4e5f-4a6b-8c7d-000000000002'
FILE_A = 'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000000a'
FILE_B = 'file-0a1b2c3d-4e5f-4a6b-8c7d-00000000000b'


def item(index: int, text_value: str = None, **metadata):
    """A VectorItem whose vector points along one axis, so nearest-neighbour
    order is predictable without depending on the index implementation."""
    vector = [0.0] * TEST_VECTOR_LENGTH
    vector[index % TEST_VECTOR_LENGTH] = 1.0
    return {
        'id': str(uuid.uuid4()),
        'text': text_value if text_value is not None else f'chunk number {index}',
        'vector': vector,
        'metadata': {'index': index, **metadata},
    }


def unit_vector(index: int):
    vector = [0.0] * TEST_VECTOR_LENGTH
    vector[index % TEST_VECTOR_LENGTH] = 1.0
    return vector


# --------------------------------------------------------------------------
# Core contract
# --------------------------------------------------------------------------


def test_insert_and_get_round_trip(client):
    items = [item(i) for i in range(5)]
    client.insert(KB_A, items)

    result = client.get(KB_A)
    assert sorted(result.ids[0]) == sorted(i['id'] for i in items)
    assert len(result.documents[0]) == 5


def test_search_returns_nearest_first(client):
    items = [item(i) for i in range(TEST_VECTOR_LENGTH)]
    client.insert(KB_A, items)

    result = client.search(KB_A, vectors=[unit_vector(3)], limit=1)
    assert result.ids[0][0] == items[3]['id']


def test_upsert_updates_in_place(client):
    original = item(0, 'before')
    client.upsert(KB_A, [original])
    updated = dict(original, text='after')
    client.upsert(KB_A, [updated])

    result = client.get(KB_A)
    assert result.ids[0] == [original['id']]
    assert result.documents[0] == ['after']


def test_query_filters_on_metadata(client):
    client.insert(KB_A, [item(0, tag='keep'), item(1, tag='drop')])

    result = client.query(KB_A, filter={'tag': 'keep'})
    assert len(result.ids[0]) == 1
    assert result.metadatas[0][0]['tag'] == 'keep'


def test_delete_by_id(client):
    items = [item(0), item(1)]
    client.insert(KB_A, items)

    client.delete(KB_A, ids=[items[0]['id']])
    assert client.get(KB_A).ids[0] == [items[1]['id']]


def test_delete_by_metadata_filter(client):
    client.insert(KB_A, [item(0, tag='keep'), item(1, tag='drop')])

    client.delete(KB_A, filter={'tag': 'drop'})
    remaining = client.get(KB_A)
    assert len(remaining.ids[0]) == 1
    assert remaining.metadatas[0][0]['tag'] == 'keep'


def test_has_collection(client):
    assert client.has_collection(KB_A) is False
    client.insert(KB_A, [item(0)])
    assert client.has_collection(KB_A) is True


def test_hybrid_search_combines_both_signals(client):
    client.insert(
        KB_A,
        [
            item(0, 'the quick brown fox jumps'),
            item(1, 'unrelated filler content'),
            item(2, 'another unrelated chunk'),
        ],
    )

    result = client.hybrid_search(KB_A, query='quick brown fox', vectors=[unit_vector(0)], limit=3)
    assert result is not None, 'pgvector is the only backend implementing native hybrid search'
    assert len(result.ids[0]) >= 1


# --------------------------------------------------------------------------
# Isolation -- the property partitioning must not break
# --------------------------------------------------------------------------


def test_collections_are_isolated(client):
    client.insert(KB_A, [item(0, 'in a')])
    client.insert(KB_B, [item(0, 'in b')])

    assert client.get(KB_A).documents[0] == ['in a']
    assert client.get(KB_B).documents[0] == ['in b']
    assert client.search(KB_A, vectors=[unit_vector(0)], limit=10).documents[0] == ['in a']


def test_collections_sharing_a_bucket_stay_isolated(client, pgvector_module):
    """Bucketed collections share a partition, so the collection_name predicate
    is the only thing keeping them apart."""
    colliding = []
    seen = {}
    for i in range(500):
        name = f'file-collide-{i}'
        key = pgvector_module.part_key_for(name)
        if key in seen:
            colliding = [seen[key], name]
            break
        seen[key] = name
    assert colliding, 'expected two collections to hash into the same bucket'

    client.insert(colliding[0], [item(0, 'first')])
    client.insert(colliding[1], [item(1, 'second')])

    assert client.get(colliding[0]).documents[0] == ['first']
    assert client.get(colliding[1]).documents[0] == ['second']


def test_delete_collection_leaves_others_intact(client):
    client.insert(KB_A, [item(0)])
    client.insert(KB_B, [item(0)])
    client.insert(FILE_A, [item(0)])

    client.delete_collection(KB_A)

    assert client.has_collection(KB_A) is False
    assert client.has_collection(KB_B) is True
    assert client.has_collection(FILE_A) is True


def test_delete_collection_on_a_bucketed_collection(client):
    """Bucket members cannot be dropped, only deleted -- the partition is shared."""
    client.insert(FILE_A, [item(0)])
    client.insert(FILE_B, [item(0)])

    client.delete_collection(FILE_A)

    assert client.has_collection(FILE_A) is False
    assert client.has_collection(FILE_B) is True


def test_collection_can_be_recreated_after_deletion(client):
    """Reindexing a knowledge base deletes the collection and writes it again."""
    client.insert(KB_A, [item(0, 'first generation')])
    client.delete_collection(KB_A)
    client.insert(KB_A, [item(1, 'second generation')])

    assert client.get(KB_A).documents[0] == ['second generation']


def test_reset_clears_everything(client):
    client.insert(KB_A, [item(0)])
    client.insert(FILE_A, [item(0)])

    client.reset()

    assert client.has_collection(KB_A) is False
    assert client.has_collection(FILE_A) is False


def test_reset_leaves_the_backend_usable(client):
    client.insert(KB_A, [item(0)])
    client.reset()
    client.insert(KB_A, [item(1, 'after reset')])

    assert client.get(KB_A).documents[0] == ['after reset']


# --------------------------------------------------------------------------
# Layout -- only meaningful when partitioning is on
# --------------------------------------------------------------------------


def partitions_of(client, parent='document_chunk'):
    return set(
        client.session.execute(
            text(
                'SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid '
                'WHERE i.inhparent = to_regclass(:parent)'
            ),
            {'parent': parent},
        )
        .scalars()
        .all()
    )


def indexes_on(client, table):
    return (
        client.session.execute(
            text('SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = :t'),
            {'t': table},
        )
        .scalars()
        .all()
    )


def test_unpartitioned_layout_is_unchanged(client, mode):
    if mode != 'unpartitioned':
        pytest.skip('layout assertions for the single-table mode')
    client.insert(KB_A, [item(0)])

    relkind = client.session.execute(text("SELECT relkind FROM pg_class WHERE relname = 'document_chunk'")).scalar()
    assert relkind == 'r', 'the flag is off; the table must stay an ordinary one'
    assert partitions_of(client) == set()

    names = {d.split(' ON ')[0].split()[-1] for d in indexes_on(client, 'document_chunk')}
    assert 'idx_document_chunk_vector' in names
    assert 'idx_document_chunk_text_search' in names
    assert 'idx_document_chunk_collection_name' in names


def test_buckets_exist_from_startup(client, mode):
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    assert {p for p in partitions_of(client) if p.startswith('document_chunk_b')} == {
        f'document_chunk_b{i}' for i in range(TEST_BUCKETS)
    }


def test_ingesting_a_file_creates_no_partition(client, mode):
    """The hot path must never run DDL: uploads, memories and web searches all
    land in buckets that already exist."""
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    before = partitions_of(client)
    client.insert(FILE_A, [item(0)])
    client.insert('user-memory-abc', [item(0)])
    client.insert('web-search-user-' + 'f' * 40, [item(0)])
    assert partitions_of(client) == before


def test_knowledge_base_gets_its_own_indexed_partition(client, pgvector_module, mode):
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    client.insert(KB_A, [item(0)])

    partition = pgvector_module.partition_name_for(pgvector_module.part_key_for(KB_A))
    assert partition in partitions_of(client)

    defs = ' '.join(indexes_on(client, partition))
    assert 'USING hnsw' in defs or 'USING ivfflat' in defs, 'a knowledge base needs a vector index'
    assert 'USING gin' in defs, 'a knowledge base needs the full-text index for hybrid search'


def test_buckets_carry_no_vector_index(client, mode):
    """The whole point: per-file rows must stay out of any vector graph."""
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    client.insert(FILE_A, [item(0)])

    defs = ' '.join(indexes_on(client, 'document_chunk_b0'))
    assert 'USING hnsw' not in defs and 'USING ivfflat' not in defs
    assert 'USING gin' not in defs
    assert 'collection_name' in defs


def test_parent_carries_no_vector_index(client, mode):
    """An index on the parent cascades to every partition, which would put a
    vector index back onto the buckets and undo the feature."""
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    client.insert(KB_A, [item(0)])

    defs = ' '.join(indexes_on(client, 'document_chunk'))
    assert 'USING hnsw' not in defs and 'USING ivfflat' not in defs
    assert 'USING gin' not in defs


def test_queries_prune_to_one_partition(client, pgvector_module, mode):
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    for collection in (KB_A, KB_B, FILE_A, FILE_B):
        client.insert(collection, [item(0)])

    part_key = pgvector_module.part_key_for(KB_A)
    scanned = scanned_relations(
        client,
        'SELECT id FROM document_chunk WHERE collection_name = :c AND part_key = :p',
        {'c': KB_A, 'p': part_key},
    )
    expected = pgvector_module.partition_name_for(part_key)
    assert scanned == {expected}, f'expected only {expected} to be scanned, got {sorted(scanned)}'


def test_dropping_a_collection_drops_its_partition(client, pgvector_module, mode):
    if mode != 'partitioned':
        pytest.skip('partitioned layout only')
    client.insert(KB_A, [item(0)])
    partition = pgvector_module.partition_name_for(pgvector_module.part_key_for(KB_A))
    assert partition in partitions_of(client)

    client.delete_collection(KB_A)
    assert partition not in partitions_of(client), 'DROP, not DELETE: a delete would leave dead tuples'
