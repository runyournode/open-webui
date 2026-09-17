"""Fixtures for the pgvector backend tests.

`pgvector.py` reads its configuration at import time, which is what lets the
unpartitioned path stay byte-for-byte unchanged when the flag is off. To cover
both modes in one run, each parametrisation re-imports the module with a
different environment rather than trying to toggle it at runtime.
"""

import importlib
import json
import os

import pytest
from sqlalchemy import text

from tests.conftest import PGVECTOR_TEST_DB_URL, TEST_VECTOR_LENGTH

MODES = ['unpartitioned', 'partitioned']
TEST_BUCKETS = 4

# Every setting `_reload_pgvector` touches. `importlib.reload` mutates the module
# in place, so a test that reloads with a non-default value changes it for every
# later test in the process unless the environment is restored.
PGVECTOR_ENV_KEYS = (
    'VECTOR_DB',
    'ENABLE_DB_MIGRATIONS',
    'DATABASE_URL',
    'PGVECTOR_DB_URL',
    'PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH',
    'PGVECTOR_PARTITIONING',
    'PGVECTOR_PARTITION_BUCKETS',
    'PGVECTOR_PARTITION_DEDICATED_PATTERN',
    'PGVECTOR_PGCRYPTO',
    'PGVECTOR_PGCRYPTO_KEY',
    'PGVECTOR_INDEX_METHOD',
    'PGVECTOR_ITERATIVE_SCAN',
)


def _reload_pgvector(partitioning: bool, **overrides):
    """Re-import config and the pgvector backend under a given environment."""
    if not PGVECTOR_TEST_DB_URL:
        pytest.skip('PGVECTOR_TEST_DB_URL is not set')
    os.environ.update(
        {
            'VECTOR_DB': 'pgvector',
            'ENABLE_DB_MIGRATIONS': 'false',
            'DATABASE_URL': PGVECTOR_TEST_DB_URL,
            'PGVECTOR_DB_URL': PGVECTOR_TEST_DB_URL,
            'PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH': str(TEST_VECTOR_LENGTH),
            'PGVECTOR_PARTITIONING': 'true' if partitioning else 'false',
            'PGVECTOR_PARTITION_BUCKETS': str(TEST_BUCKETS),
        }
    )
    for key, value in overrides.items():
        os.environ[key] = str(value)

    import open_webui.config as config
    import open_webui.retrieval.vector.dbs.pgvector as pgvector

    importlib.reload(config)
    return importlib.reload(pgvector)


def _drop_schema(url: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.begin() as conn:
        # Everything the backend or the migration script can create. Leaving any
        # of it behind makes the next test inherit stale partitions.
        for table in (
            'document_chunk',
            'document_chunk_partitioned',
            'document_chunk_pre_partition',
            'document_chunk_migration_progress',
            'document_chunk_migration_part_map',
        ):
            conn.execute(text(f'DROP TABLE IF EXISTS {table} CASCADE'))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    engine.dispose()


def scanned_relations(client, sql: str, params: dict) -> set:
    """The distinct tables a plan actually reads.

    Taken from EXPLAIN (FORMAT JSON) rather than by grepping the text plan: a
    bitmap scan names both the partition and its index, so counting lines that
    mention the table prefix over-counts and makes a correctly pruned query look
    like it touched two partitions.
    """
    plan = client.session.execute(
        text(f'EXPLAIN (FORMAT JSON) {sql}'),
        params,
    ).scalar()
    if isinstance(plan, str):
        plan = json.loads(plan)

    relations = set()

    def walk(node):
        name = node.get('Relation Name')
        if name:
            relations.add(name)
        for child in node.get('Plans', []):
            walk(child)

    walk(plan[0]['Plan'])
    return relations


@pytest.fixture(autouse=True)
def _restore_pgvector_env():
    """Undo configuration changes a test made, and reload the module under the
    restored environment so the next test starts from a known state."""
    if not PGVECTOR_TEST_DB_URL:
        # Nothing was loaded, so there is nothing to restore -- and a reload
        # here would turn every skipped test into an error.
        yield
        return
    saved = {key: os.environ.get(key) for key in PGVECTOR_ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _reload_pgvector(partitioning=os.environ.get('PGVECTOR_PARTITIONING') == 'true')


@pytest.fixture(params=MODES)
def mode(request):
    """Runs every test that uses it once per storage layout."""
    return request.param


@pytest.fixture
def pgvector_module(mode):
    if not PGVECTOR_TEST_DB_URL:
        pytest.skip('PGVECTOR_TEST_DB_URL is not set')
    _drop_schema(PGVECTOR_TEST_DB_URL)
    return _reload_pgvector(partitioning=(mode == 'partitioned'))


@pytest.fixture
def client(pgvector_module):
    instance = pgvector_module.PgvectorClient()
    yield instance
    instance.session.rollback()


@pytest.fixture
def reload_pgvector():
    """Escape hatch for tests that need a non-default configuration."""
    return _reload_pgvector


@pytest.fixture
def drop_schema():
    if not PGVECTOR_TEST_DB_URL:
        pytest.skip('PGVECTOR_TEST_DB_URL is not set')
    return lambda: _drop_schema(PGVECTOR_TEST_DB_URL)
