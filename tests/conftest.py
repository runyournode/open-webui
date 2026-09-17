"""Shared fixtures for the backend test suite.

The pgvector tests need a real PostgreSQL with the `vector` extension; there is
no meaningful in-memory substitute for partition pruning or index selection.
Point `PGVECTOR_TEST_DB_URL` at one (see docker-compose.pgvector-test.yaml) or
the tests that need it are skipped.
"""

import os
import warnings

import pytest

PGVECTOR_TEST_DB_URL = os.environ.get('PGVECTOR_TEST_DB_URL')

# Small vectors keep index builds instant; nothing under test depends on the
# dimension.
TEST_VECTOR_LENGTH = 8


def pytest_configure(config):
    config.addinivalue_line('markers', 'pgvector: needs a PostgreSQL server with the vector extension')

    # The driver and ORM versions decide how statements are parameterised, and
    # therefore whether partition pruning happens at plan time at all -- so a
    # result taken against some other dependency set says less than it seems.
    # In the reference environment (the compose file sets OWUI_TEST_STRICT_ENV)
    # a mismatch is an error; on a developer's machine it is a warning, so the
    # suite can still be run.
    from tests.environment import mismatches

    drift = mismatches()
    if drift:
        message = (
            "The environment does not match this branch's pins: "
            + '; '.join(f'{package}: {detail}' for package, detail in sorted(drift.items()))
            + '. Run the suite on ghcr.io/open-webui/open-webui:dev (see docker-compose.pgvector-test.yaml).'
        )
        if os.environ.get('OWUI_TEST_STRICT_ENV'):
            raise pytest.UsageError(message)
        warnings.warn(message, stacklevel=1)
