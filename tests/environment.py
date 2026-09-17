"""What the tests and benchmarks are actually running against.

A measurement that does not record its environment cannot be reproduced or
defended. This collects the pieces that matter for the pgvector backend and
checks them against the branch's own pins, so a mismatch between the image and
the source tree is an error rather than a silent variable.

It matters in particular because the recommended runner is
`ghcr.io/open-webui/open-webui:dev`, which is rebuilt nightly: two campaigns run
a day apart can otherwise sit on two different dependency sets with nothing in
the results to show it.
"""

import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# The packages this backend's behaviour actually depends on. Driver and ORM
# versions decide how statements are parameterised and therefore whether
# partition pruning happens at plan time or execution time; the pgvector version
# decides which index methods and GUCs exist at all.
CRITICAL_PACKAGES = ('sqlalchemy', 'psycopg2-binary', 'psycopg', 'pgvector', 'alembic')


def installed_versions() -> Dict[str, Optional[str]]:
    versions = {}
    for package in CRITICAL_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def pinned_versions(pyproject: Optional[Path] = None) -> Dict[str, str]:
    """The `name==version` pins from pyproject.toml, for the critical packages.

    Parsed rather than hardcoded so this cannot drift from the branch it claims
    to describe.
    """
    if pyproject is None:
        pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if not pyproject.is_file():
        return {}

    pins: Dict[str, str] = {}
    for raw in pyproject.read_text().splitlines():
        line = raw.strip().strip(',').strip('"')
        if '==' not in line:
            continue
        name, _, version = line.partition('==')
        # Strip any extras marker, e.g. `psycopg[binary]` -> `psycopg`.
        name = name.split('[', 1)[0].strip().lower()
        if name in CRITICAL_PACKAGES and name not in pins:
            pins[name] = version.strip()
    return pins


def describe() -> Dict[str, object]:
    """Everything worth recording alongside a measurement."""
    return {
        'python': sys.version.split()[0],
        'packages': installed_versions(),
        'pins': pinned_versions(),
        # Set by the compose file so results can name the exact image build.
        'image': os.environ.get('OWUI_TEST_IMAGE', ''),
        'image_digest': os.environ.get('OWUI_TEST_IMAGE_DIGEST', ''),
    }


def mismatches() -> Dict[str, str]:
    """Critical packages whose installed version differs from the branch pin."""
    installed, pins = installed_versions(), pinned_versions()
    out = {}
    for package, pinned in pins.items():
        actual = installed.get(package)
        if actual is not None and actual != pinned:
            out[package] = f'installed {actual}, branch pins {pinned}'
    return out
