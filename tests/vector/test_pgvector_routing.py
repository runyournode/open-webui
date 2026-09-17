"""Partition routing rules.

Pure functions, so these run without a database. They pin the behaviour that
matters most: which of Open WebUI's collection shapes earn a partition of their
own, and which are bucketed. Getting this wrong does not fail loudly -- it
quietly creates a partition per uploaded file or per web search.
"""

import zlib

import pytest


@pytest.fixture
def routing(reload_pgvector):
    """Reloaded per test: `importlib.reload` mutates the module in place, so a
    cached copy would pick up configuration from whichever test ran last."""
    return reload_pgvector(partitioning=True, PGVECTOR_PARTITION_BUCKETS=16)


# The collection shapes Open WebUI actually creates, and where each belongs.
# Knowledge bases are the only high-volume, vector-searched collections, so they
# are the only ones worth a dedicated partition and its own index.
DEDICATED = [
    '0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d',  # knowledge base id
    '0A1B2C3D-4E5F-4A6B-8C7D-9E0F1A2B3C4D',  # ... case-insensitively
    'knowledge-bases',  # the system meta-collection
]
BUCKETED = [
    'file-0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d',  # one per uploaded file
    'user-memory-abc123',  # one per user
    'web-search-user1-' + 'a' * 40,  # one per web search
    'a' * 64,  # sha256 of /process/text content
    'b' * 63,  # truncated sha256 of a /process/web URL
    'some-client-supplied-name',  # unscoped collections, when enabled
]


@pytest.mark.parametrize('collection', DEDICATED)
def test_knowledge_bases_get_their_own_partition(routing, collection):
    assert routing.is_dedicated_collection(collection)
    assert routing.part_key_for(collection) == collection


@pytest.mark.parametrize('collection', BUCKETED)
def test_high_cardinality_collections_are_bucketed(routing, collection):
    assert not routing.is_dedicated_collection(collection)
    assert routing.part_key_for(collection).startswith(routing.BUCKET_PART_KEY_PREFIX)


def test_bucket_assignment_is_stable(routing):
    """Readers recompute the key instead of reading it back, so it must not drift.

    The constant is pinned on purpose: crc32 is fixed by the algorithm rather
    than by the Python or PostgreSQL version, which is exactly why it was chosen
    over `hashtext()`. If this value ever changes, existing rows have silently
    become unreachable.
    """
    assert routing.part_key_for('file-abc') == f'{routing.BUCKET_PART_KEY_PREFIX}12'
    assert zlib.crc32(b'file-abc') == 0xDB51A74C


def test_buckets_are_spread(routing):
    keys = {routing.part_key_for(f'file-{i}') for i in range(2000)}
    assert len(keys) == 16, 'every bucket should be used'


@pytest.mark.parametrize(
    'collection',
    DEDICATED + BUCKETED + ['x' * 255, 'MiXeD-CaSe', 'a-b_c-d'],
)
def test_partition_names_are_valid_identifiers(routing, collection):
    """PostgreSQL truncates identifiers at 63 bytes and folds them to lower case.

    Partition names are derived rather than interpolated so that a long or
    oddly-cased collection name cannot collide with another or be silently cut.
    """
    part_key = routing.part_key_for(collection)
    name = routing.partition_name_for(part_key)

    # Longest index suffix the module appends to each kind of partition.
    suffix = '_collection_name_idx' if part_key.startswith(routing.BUCKET_PART_KEY_PREFIX) else '_vector_idx'
    assert len(name) + len(suffix) <= 63, 'PostgreSQL would silently truncate the index name'
    assert name == name.lower(), 'PostgreSQL folds unquoted identifiers to lower case'
    assert all(c.isalnum() or c == '_' for c in name)


def test_distinct_collections_get_distinct_partition_names(routing):
    names = {
        routing.partition_name_for(routing.part_key_for(f'{i:08x}-0000-4000-8000-000000000000')) for i in range(500)
    }
    assert len(names) == 500


def test_case_differing_collections_do_not_collide(routing):
    """Lower-casing the slug must not merge two distinct collections."""
    lower = routing.partition_name_for('aaaaaaaa-0000-4000-8000-000000000000')
    upper = routing.partition_name_for('AAAAAAAA-0000-4000-8000-000000000000')
    assert lower != upper
