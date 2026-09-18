"""Which way round is it: does an empty result retain the connection, or does an
exhausted pool produce empty results?

The two hypotheses make opposite predictions at low concurrency. With three
threads against a pool of fifteen, a checkout timeout is impossible -- so if
connections are still retained, nothing about pool exhaustion can explain it.
And if a NON-empty search with the identical thread pattern returns its
connection, then concurrency is not the trigger either. What remains is the
result being empty.
"""

import os
import sys
import threading

sys.path.insert(0, '/app/backend')

from sqlalchemy import create_engine, text  # noqa: E402

from open_webui.retrieval.vector.dbs.pgvector import PgvectorClient  # noqa: E402

client = PgvectorClient()
pool = client.session.get_bind().pool
vector = [0.1] * 1024

engine = create_engine(os.environ['PGVECTOR_DB_URL'])
with engine.connect() as conn:
    populated = conn.execute(
        text("SELECT collection_name FROM document_chunk WHERE collection_name LIKE '%-0000-4000-8000-%' GROUP BY 1 LIMIT 1")
    ).scalar()
engine.dispose()

EMPTY = 'file-this-collection-does-not-exist'


def probe(collection, label, n=3):
    """n is far below the pool's capacity, so a timeout cannot occur."""
    got = []

    def one():
        r = client.search(collection_name=collection, vectors=[vector], limit=10)
        got.append(0 if r is None else len(r.ids[0]))

    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'{label:<44} rows returned={got}  checked out={pool.checkedout()}  free={pool.checkedin()}')


print(f'pool: size {pool.size()}, max overflow {pool._max_overflow}, so {pool.size() + pool._max_overflow} connections in total\n')
probe(populated, '3 threads, NON-EMPTY result')
probe(EMPTY, '3 threads, EMPTY result')
probe(populated, '3 threads, NON-EMPTY result (again)')
