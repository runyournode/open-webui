import hashlib
import logging
import re
import threading
import zlib
from typing import Any, Dict, List, Optional, Tuple

from open_webui.config import (
    PGVECTOR_CREATE_EXTENSION,
    PGVECTOR_DB_URL,
    PGVECTOR_HNSW_EF_CONSTRUCTION,
    PGVECTOR_HNSW_M,
    PGVECTOR_INDEX_METHOD,
    PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH,
    PGVECTOR_ITERATIVE_SCAN,
    PGVECTOR_IVFFLAT_LISTS,
    PGVECTOR_PARTITION_BUCKETS,
    PGVECTOR_PARTITION_DEDICATED_PATTERN,
    PGVECTOR_PARTITIONING,
    PGVECTOR_PGCRYPTO,
    PGVECTOR_PGCRYPTO_KEY,
    PGVECTOR_POOL_MAX_OVERFLOW,
    PGVECTOR_POOL_RECYCLE,
    PGVECTOR_POOL_SIZE,
    PGVECTOR_POOL_TIMEOUT,
    PGVECTOR_USE_HALFVEC,
)
from open_webui.internal.db import ScopedSession, enable_iam_token_auth
from open_webui.retrieval.vector.main import (
    GetResult,
    SearchResult,
    VectorDBBase,
    VectorItem,
)
from open_webui.retrieval.vector.utils import merge_hybrid_search_results, process_metadata
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import sanitize_text_for_db
from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import (
    Column,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    cast,
    column,
    create_engine,
    func,
    literal,
    select,
    text,
    values,
)
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.sql import true

VECTOR_LENGTH = PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH
USE_HALFVEC = PGVECTOR_USE_HALFVEC

VECTOR_TYPE_FACTORY = HALFVEC if USE_HALFVEC else Vector
VECTOR_OPCLASS = 'halfvec_cosine_ops' if USE_HALFVEC else 'vector_cosine_ops'
Base = declarative_base()

log = logging.getLogger(__name__)

# PostgreSQL folds unquoted identifiers to lower case and truncates them at 63
# bytes, while collection names are application strings of up to 255 characters.
# Partition names are therefore derived, never interpolated: a readable slug for
# operators plus a digest of the full name to keep them unique.
PARTITION_PREFIX = 'document_chunk_p_'
BUCKET_PREFIX = 'document_chunk_b'
BUCKET_PART_KEY_PREFIX = 'bucket_'
# `pg_advisory_xact_lock(classid, objid)` is namespaced by this classid so it
# cannot collide with any other advisory lock taken against the same database.
PARTITION_ADVISORY_CLASSID = 0x4F57_5043 - 2**31
# Bucket creation at startup takes one fixed key; dedicated partitions take one
# derived from their part_key.
BUCKETS_ADVISORY_OBJID = zlib.crc32(BUCKET_PART_KEY_PREFIX.encode('utf-8')) - 2**31
PARTITION_LOCK_TIMEOUT_MS = 3000
PARTITION_LOCK_RETRIES = 3
_SLUG_MAX = 16
_UNSAFE_IDENT = re.compile(r'[^a-z0-9_]+')

try:
    _DEDICATED_RE = re.compile(PGVECTOR_PARTITION_DEDICATED_PATTERN)
except re.error:
    log.exception(
        'PGVECTOR_PARTITION_DEDICATED_PATTERN is not a valid regular expression; '
        'falling back to bucketing every collection.'
    )
    # Matches nothing, so every collection is bucketed. Degrading to "all
    # buckets" keeps the install working; the alternative, a partition per
    # collection, would not.
    _DEDICATED_RE = re.compile(r'(?!)')


def is_dedicated_collection(collection_name: str) -> bool:
    """Whether this collection gets a partition (and vector index) of its own.

    Allow-list by design. Open WebUI creates a collection per uploaded file, per
    user memory, per web search and per processed text or URL; giving each of
    those a partition would put DDL in the ingestion path and leave the table
    with an unbounded number of partitions. Anything not matched here is hashed
    into a bucket instead.
    """
    return bool(_DEDICATED_RE.search(collection_name))


def part_key_for(collection_name: str) -> str:
    """Map a collection name to its partition key.

    Must stay a pure function of the name: readers recompute it locally rather
    than reading it back, so any history- or size-dependent routing would send
    reads to the wrong partition. `zlib.crc32` is used rather than PostgreSQL's
    `hashtext()` because its output is stable across server versions -- rows
    hashed by `hashtext()` would silently land in the wrong bucket after a major
    upgrade.
    """
    if is_dedicated_collection(collection_name):
        return collection_name
    bucket = zlib.crc32(collection_name.encode('utf-8')) % PGVECTOR_PARTITION_BUCKETS
    return f'{BUCKET_PART_KEY_PREFIX}{bucket}'


def partition_name_for(part_key: str) -> str:
    """Name of the partition holding `part_key`. A safe SQL identifier."""
    if part_key.startswith(BUCKET_PART_KEY_PREFIX):
        return f'{BUCKET_PREFIX}{part_key[len(BUCKET_PART_KEY_PREFIX) :]}'
    slug = _UNSAFE_IDENT.sub('_', part_key.lower())[:_SLUG_MAX].strip('_')
    digest = hashlib.sha1(part_key.encode('utf-8')).hexdigest()[:12]
    return f'{PARTITION_PREFIX}{slug}_{digest}' if slug else f'{PARTITION_PREFIX}{digest}'


def pgcrypto_encrypt(val, key):
    return func.pgp_sym_encrypt(val, literal(key))


def pgcrypto_decrypt(col, key, outtype='text'):
    return func.cast(func.pgp_sym_decrypt(col, literal(key)), outtype)


def vector_index_configuration() -> Tuple[str, str]:
    """Index method and options for a vector index, from configuration.

    Module level so the partition migration builds indexes by exactly the same
    rules the running backend would.
    """
    if PGVECTOR_INDEX_METHOD:
        index_method = PGVECTOR_INDEX_METHOD
        log.info(
            "Using vector index method '%s' from PGVECTOR_INDEX_METHOD.",
            index_method,
        )
    elif USE_HALFVEC:
        index_method = 'hnsw'
        log.info(
            'VECTOR_LENGTH=%s exceeds 2000; using halfvec column type with hnsw index.',
            VECTOR_LENGTH,
        )
    else:
        index_method = 'ivfflat'

    if index_method == 'hnsw':
        index_options = f'WITH (m = {PGVECTOR_HNSW_M}, ef_construction = {PGVECTOR_HNSW_EF_CONSTRUCTION})'
    else:
        index_options = f'WITH (lists = {PGVECTOR_IVFFLAT_LISTS})'

    return index_method, index_options


class DocumentChunk(Base):
    __tablename__ = 'document_chunk'

    id = Column(Text, primary_key=True)
    vector = Column(VECTOR_TYPE_FACTORY(dim=VECTOR_LENGTH), nullable=True)
    collection_name = Column(Text, nullable=False)

    if PGVECTOR_PARTITIONING:
        # PostgreSQL requires the partition key in every unique constraint, so
        # the primary key becomes composite. Chunk ids are uuid4 and the
        # knowledge-base copy of a file is inserted with fresh ids, so nothing
        # relies on `id` being unique on its own.
        part_key = Column(Text, primary_key=True, nullable=False)
        __table_args__ = {'postgresql_partition_by': 'LIST (part_key)'}

    if PGVECTOR_PGCRYPTO:
        text = Column(LargeBinary, nullable=True)
        vmetadata = Column(LargeBinary, nullable=True)
    else:
        text = Column(Text, nullable=True)
        vmetadata = Column(MutableDict.as_mutable(JSONB), nullable=True)


class PgvectorClient(VectorDBBase):
    def __init__(self) -> None:
        # Partitions already known to exist. `AsyncVectorDBClient` dispatches
        # every call through `asyncio.to_thread`, so this is shared between
        # threads of one worker as well as being per-process.
        self._known_part_keys: set[str] = set()
        self._partition_lock = threading.Lock()
        # Whether this pgvector build exposes <method>.iterative_scan, per method.
        self._iterative_scan_support: Dict[str, bool] = {}

        # if no pgvector uri, use the existing database connection
        if not PGVECTOR_DB_URL:
            self.session = ScopedSession
        else:
            if isinstance(PGVECTOR_POOL_SIZE, int):
                if PGVECTOR_POOL_SIZE > 0:
                    engine = create_engine(
                        PGVECTOR_DB_URL,
                        pool_size=PGVECTOR_POOL_SIZE,
                        max_overflow=PGVECTOR_POOL_MAX_OVERFLOW,
                        pool_timeout=PGVECTOR_POOL_TIMEOUT,
                        pool_recycle=PGVECTOR_POOL_RECYCLE,
                        pool_pre_ping=True,
                        poolclass=QueuePool,
                    )
                else:
                    engine = create_engine(PGVECTOR_DB_URL, pool_pre_ping=True, poolclass=NullPool)
            else:
                engine = create_engine(PGVECTOR_DB_URL, pool_pre_ping=True)

            enable_iam_token_auth(engine)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
            self.session = scoped_session(SessionLocal)

        try:
            # Ensure the pgvector extension is available
            # Use a conditional check to avoid permission issues on Azure PostgreSQL
            if PGVECTOR_CREATE_EXTENSION:
                self.session.execute(
                    text("""
                    DO $$
                    BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                        CREATE EXTENSION IF NOT EXISTS vector;
                    END IF;
                    END $$;
                """)
                )

            if PGVECTOR_PGCRYPTO:
                # Ensure the pgcrypto extension is available for encryption
                # Use a conditional check to avoid permission issues on Azure PostgreSQL
                self.session.execute(
                    text("""
                    DO $$
                    BEGIN
                       IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto') THEN
                          CREATE EXTENSION IF NOT EXISTS pgcrypto;
                       END IF;
                    END $$;
                """)
                )

                if not PGVECTOR_PGCRYPTO_KEY:
                    raise ValueError('PGVECTOR_PGCRYPTO_KEY must be set when PGVECTOR_PGCRYPTO is enabled.')

            # Refuse to run against a table whose shape contradicts the flag.
            # Must happen before create_all, which silently skips a table that
            # already exists.
            self.check_partitioning_matches_schema()

            # Check vector length consistency
            self.check_vector_length()

            # Create the tables if they do not exist
            # Base.metadata.create_all requires a bind (engine or connection)
            # Get the connection from the session
            connection = self.session.connection()
            Base.metadata.create_all(bind=connection)

            index_method, index_options = self._vector_index_configuration()
            self._index_method = index_method
            self._index_options = index_options

            if PGVECTOR_PARTITIONING:
                # Indexes must never be created on the parent: a partitioned
                # index cascades to every partition, which would put a vector
                # index back on the buckets and undo the whole point. Each
                # partition is indexed individually instead.
                self._ensure_buckets()
            else:
                self._ensure_vector_index(index_method, index_options)
                self._ensure_text_search_index()

                self.session.execute(
                    text(
                        'CREATE INDEX IF NOT EXISTS idx_document_chunk_collection_name '
                        'ON document_chunk (collection_name);'
                    )
                )
            self.session.commit()
            log.info('Initialization complete.')
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during initialization: {e}')
            raise

    @staticmethod
    def _extract_index_method(index_def: Optional[str]) -> Optional[str]:
        if not index_def:
            return None
        try:
            after_using = index_def.lower().split('using ', 1)[1]
            return after_using.split()[0]
        except (IndexError, AttributeError):
            return None

    def _vector_index_configuration(self) -> Tuple[str, str]:
        return vector_index_configuration()

    def _ensure_vector_index(
        self,
        index_method: str,
        index_options: str,
        table_name: str = 'document_chunk',
        index_name: str = 'idx_document_chunk_vector',
    ) -> None:
        existing_index_def = self.session.execute(
            text("""
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = current_schema()
                  AND tablename = :table_name
                  AND indexname = :index_name
                """),
            {'table_name': table_name, 'index_name': index_name},
        ).scalar()

        existing_method = self._extract_index_method(existing_index_def)
        if existing_method and existing_method != index_method:
            raise RuntimeError(
                f"Existing pgvector index '{index_name}' uses method '{existing_method}' but configuration now "
                f"requires '{index_method}'. Automatic rebuild is disabled to prevent long-running maintenance. "
                'Drop the index manually (optionally after tuning maintenance_work_mem/max_parallel_maintenance_workers) '
                'and recreate it with the new method before restarting Open WebUI.'
            )

        if not existing_index_def:
            index_sql = (
                f'CREATE INDEX IF NOT EXISTS {index_name} '
                f'ON {table_name} USING {index_method} (vector {VECTOR_OPCLASS})'
            )
            if index_options:
                index_sql = f'{index_sql} {index_options}'
            self.session.execute(text(index_sql))
            log.info(
                "Ensured vector index '%s' using %s%s.",
                index_name,
                index_method,
                f' {index_options}' if index_options else '',
            )

    def _ensure_text_search_index(
        self,
        table_name: str = 'document_chunk',
        index_name: str = 'idx_document_chunk_text_search',
    ) -> None:
        if PGVECTOR_PGCRYPTO:
            return

        self.session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS {index_name} '
                f"ON {table_name} USING GIN (to_tsvector('simple', coalesce(text, '')));"
            )
        )
        log.info("Ensured text search index '%s'.", index_name)

    def check_partitioning_matches_schema(self) -> None:
        """Refuse to start when PGVECTOR_PARTITIONING disagrees with the table.

        `Base.metadata.create_all(checkfirst=True)` silently skips a table that
        already exists, so without this check flipping the flag on an existing
        install would leave every query referencing a `part_key` column that is
        not there -- and flipping it off would leave rows stranded in partitions
        nothing reads.
        """
        relkind = self.session.execute(
            text(
                'SELECT relkind FROM pg_class WHERE relname = :name AND relnamespace = current_schema()::regnamespace'
            ),
            {'name': DocumentChunk.__tablename__},
        ).scalar()

        if relkind is None:
            # Fresh install: create_all builds whichever shape the flag asks for.
            return

        if PGVECTOR_PARTITIONING and relkind != 'p':
            raise RuntimeError(
                "PGVECTOR_PARTITIONING is enabled but 'document_chunk' is not a partitioned table. "
                'Convert it with: python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate '
                '--migrate  (or unset PGVECTOR_PARTITIONING).'
            )
        if not PGVECTOR_PARTITIONING and relkind == 'p':
            raise RuntimeError(
                "'document_chunk' is a partitioned table but PGVECTOR_PARTITIONING is not enabled. "
                'Set PGVECTOR_PARTITIONING=true, or roll the migration back with: '
                'python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate --rollback'
            )

    def _render_ddl(self, fmt: str, *args: str) -> str:
        """Render DDL with PostgreSQL's own identifier and literal quoting.

        DDL cannot take bound parameters and partition keys are application
        strings, so the statement is built server-side through `format()` with
        `%I`/`%L` rather than by escaping in Python.
        """
        placeholders = ', '.join(f':a{i}' for i in range(len(args)))
        params = {f'a{i}': value for i, value in enumerate(args)}
        return self.session.execute(
            text(f'SELECT format(:fmt, {placeholders})'),
            {'fmt': fmt, **params},
        ).scalar()

    def _ensure_buckets(self) -> None:
        """Create the fixed bucket partitions. Runs once per process, at startup.

        Buckets get a btree on `collection_name` and nothing else. They hold the
        many small collections -- per file, per user memory, per web search, per
        processed text -- which are looked up by exact collection name and then
        distance-sorted over a handful of rows. A vector index over them would
        only inflate the graph and lengthen every vacuum pass.

        Workers starting together race here exactly as they do on dedicated
        partitions: `IF NOT EXISTS` is not atomic, and worse, a worker that
        read the partition list before another finished creating the buckets
        would then find them in pg_class and misreport them as orphans of a
        rolled-back migration. So the lock comes first and the state is read
        under it. It is released when __init__ commits.
        """
        self.session.execute(
            text('SELECT pg_advisory_xact_lock(:classid, :objid)'),
            {'classid': PARTITION_ADVISORY_CLASSID, 'objid': BUCKETS_ADVISORY_OBJID},
        )
        existing = (
            self.session.execute(
                text(
                    'SELECT c.relname FROM pg_inherits i '
                    'JOIN pg_class c ON c.oid = i.inhrelid '
                    'WHERE i.inhparent = to_regclass(:parent) AND c.relname LIKE :pattern'
                ),
                {'parent': DocumentChunk.__tablename__, 'pattern': f'{BUCKET_PREFIX}%'},
            )
            .scalars()
            .all()
        )

        stale = sorted(
            int(suffix)
            for suffix in (name[len(BUCKET_PREFIX) :] for name in existing)
            if suffix.isdigit() and int(suffix) >= PGVECTOR_PARTITION_BUCKETS
        )
        if stale:
            raise RuntimeError(
                f'document_chunk has bucket partitions {stale} beyond the configured '
                f'PGVECTOR_PARTITION_BUCKETS={PGVECTOR_PARTITION_BUCKETS}. Lowering the bucket count '
                'strands their rows: collections that hashed into them would silently stop being '
                'found. Restore the previous value, or move the rows first.'
            )

        attached = set(existing)
        for bucket in range(PGVECTOR_PARTITION_BUCKETS):
            part_key = f'{BUCKET_PART_KEY_PREFIX}{bucket}'
            partition = partition_name_for(part_key)

            if partition not in attached:
                # `CREATE TABLE IF NOT EXISTS ... PARTITION OF` matches on the
                # name alone: if a table of that name exists but hangs off a
                # different parent -- what a rolled-back migration leaves behind
                # -- it would quietly do nothing and leave the bucket missing.
                orphan = self.session.execute(
                    text(
                        'SELECT relkind FROM pg_class '
                        'WHERE relname = :name AND relnamespace = current_schema()::regnamespace'
                    ),
                    {'name': partition},
                ).scalar()
                if orphan is not None:
                    raise RuntimeError(
                        f"Table '{partition}' already exists but is not a partition of "
                        f"'{DocumentChunk.__tablename__}'. A rolled-back migration leaves the old "
                        'partitioned table and its partitions in place; drop them before starting.'
                    )

            self.session.execute(
                text(
                    self._render_ddl(
                        'CREATE TABLE IF NOT EXISTS %I PARTITION OF document_chunk FOR VALUES IN (%L)',
                        partition,
                        part_key,
                    )
                )
            )
            self.session.execute(
                text(f'CREATE INDEX IF NOT EXISTS {partition}_collection_name_idx ON {partition} (collection_name)')
            )
            self._known_part_keys.add(part_key)

        log.info('Ensured %s bucket partitions.', PGVECTOR_PARTITION_BUCKETS)

    def _ensure_partition(self, collection_name: str) -> Optional[str]:
        """Return the partition key for a collection, creating its partition if needed.

        Only dedicated partitions are ever created here. Buckets exist from
        startup, so the high-volume ingestion paths -- uploads, memories, web
        search -- never run DDL.
        """
        if not PGVECTOR_PARTITIONING:
            return None

        part_key = part_key_for(collection_name)
        if part_key in self._known_part_keys:
            return part_key

        if part_key.startswith(BUCKET_PART_KEY_PREFIX):
            self._known_part_keys.add(part_key)
            return part_key

        with self._partition_lock:
            if part_key not in self._known_part_keys:
                self._create_dedicated_partition(part_key)
                self._known_part_keys.add(part_key)
        return part_key

    def _create_dedicated_partition(self, part_key: str) -> None:
        partition = partition_name_for(part_key)
        lock_key = zlib.crc32(part_key.encode('utf-8')) - 2**31

        for attempt in range(PARTITION_LOCK_RETRIES):
            try:
                # Creating a partition takes ACCESS EXCLUSIVE on the parent, so it
                # runs in its own transaction and commits before the caller writes
                # any rows. Holding that lock for the length of an insert would
                # stall every concurrent reader.
                self.session.rollback()
                self.session.execute(text(f"SET LOCAL lock_timeout = '{PARTITION_LOCK_TIMEOUT_MS}ms'"))
                # `IF NOT EXISTS` is checked and acted on non-atomically: twelve
                # workers racing for one partition still produce eleven
                # duplicate_table errors without this lock.
                self.session.execute(
                    text('SELECT pg_advisory_xact_lock(:classid, :objid)'),
                    {'classid': PARTITION_ADVISORY_CLASSID, 'objid': lock_key},
                )
                self.session.execute(
                    text(
                        self._render_ddl(
                            'CREATE TABLE IF NOT EXISTS %I PARTITION OF document_chunk FOR VALUES IN (%L)',
                            partition,
                            part_key,
                        )
                    )
                )
                # Indexed while empty, so this is instant and safe to keep in the
                # same transaction as the partition itself.
                self._ensure_vector_index(
                    self._index_method,
                    self._index_options,
                    table_name=partition,
                    index_name=f'{partition}_vector_idx',
                )
                self._ensure_text_search_index(table_name=partition, index_name=f'{partition}_text_idx')
                self.session.commit()
                log.info("Created partition '%s' for collection '%s'.", partition, part_key)
                return
            except Exception as e:
                self.session.rollback()
                sqlstate = getattr(getattr(e, 'orig', None), 'pgcode', None)
                if sqlstate == '42P07':
                    # duplicate_table: another worker created it first, which is
                    # exactly the outcome we wanted.
                    log.debug("Partition '%s' was created concurrently.", partition)
                    return
                if sqlstate == '55P03' and attempt < PARTITION_LOCK_RETRIES - 1:
                    # lock_not_available: a long-running reader holds the parent.
                    # Back off rather than blocking ingestion indefinitely.
                    log.warning(
                        "Timed out locking 'document_chunk' to create partition '%s' (attempt %s/%s); retrying.",
                        partition,
                        attempt + 1,
                        PARTITION_LOCK_RETRIES,
                    )
                    continue
                log.exception('Error creating partition %s: %s', partition, e)
                raise

    @staticmethod
    def _is_missing_partition(error: Exception) -> bool:
        """Whether a write failed because its partition no longer exists.

        Expected across workers: one can drop a knowledge base's partition --
        reindexing does exactly that -- while another still has the key cached.

        PostgreSQL reports this as check_violation (23514), the same code as a
        real CHECK failure, but the two are distinguishable without reading the
        message text, which is translated when the server runs with NLS:
        routing failures name the *parent* table and carry no constraint, while
        a CHECK failure names the partition and its constraint.
        """
        orig = getattr(error, 'orig', None)
        if getattr(orig, 'pgcode', None) != '23514':
            return False
        diag = getattr(orig, 'diag', None)
        if diag is None:
            return False
        return diag.constraint_name is None and diag.table_name == DocumentChunk.__tablename__

    def _write_with_partition_retry(self, collection_name: str, write, items: List[VectorItem]) -> None:
        """Run a write, re-creating the partition once if it vanished underneath us."""
        if not PGVECTOR_PARTITIONING:
            write(collection_name, items)
            return

        try:
            write(collection_name, items)
        except Exception as e:
            if not self._is_missing_partition(e):
                raise
            self.session.rollback()
            with self._partition_lock:
                self._known_part_keys.discard(part_key_for(collection_name))
            log.warning(
                "Partition for collection '%s' was dropped by another worker; recreating it.",
                collection_name,
            )
            write(collection_name, items)

    def _collection_scope(self, collection_name: str) -> List[Any]:
        """WHERE clauses restricting a query to one collection.

        Adds the partition key so PostgreSQL can prune, which is what makes the
        partitioning worth anything on the read path. Nothing about this reaches
        `VectorDBBase`: the key is derived locally from the collection name.
        """
        clauses: List[Any] = [DocumentChunk.collection_name == collection_name]
        if PGVECTOR_PARTITIONING:
            clauses.append(DocumentChunk.part_key == part_key_for(collection_name))
        return clauses

    def check_vector_length(self) -> None:
        """
        Check if the VECTOR_LENGTH matches the existing vector column dimension in the database.
        Raises an exception if there is a mismatch.
        """
        metadata = MetaData()
        try:
            # Attempt to reflect the 'document_chunk' table
            document_chunk_table = Table('document_chunk', metadata, autoload_with=self.session.bind)
        except NoSuchTableError:
            # Table does not exist; no action needed
            return

        # Proceed to check the vector column
        if 'vector' in document_chunk_table.columns:
            vector_column = document_chunk_table.columns['vector']
            vector_type = vector_column.type
            expected_type = HALFVEC if USE_HALFVEC else Vector

            if not isinstance(vector_type, expected_type):
                raise Exception(
                    "The 'vector' column type does not match the expected type "
                    f"('{expected_type.__name__}') for VECTOR_LENGTH {VECTOR_LENGTH}."
                )

            db_vector_length = getattr(vector_type, 'dim', None)
            if db_vector_length is not None and db_vector_length != VECTOR_LENGTH:
                raise Exception(
                    f'VECTOR_LENGTH {VECTOR_LENGTH} does not match existing vector column dimension {db_vector_length}. '
                    'Cannot change vector size after initialization without migrating the data.'
                )
        else:
            raise Exception("The 'vector' column does not exist in the 'document_chunk' table.")

    def _pgcrypto_insert_sql(self, conflict_action: str) -> str:
        """INSERT statement for the pgcrypto path.

        The ON CONFLICT target has to name the whole primary key, and under
        partitioning that key is composite: PostgreSQL requires the partition
        key in every unique constraint.
        """
        if PGVECTOR_PARTITIONING:
            columns = 'id, part_key, vector, collection_name, text, vmetadata'
            values = ':id, :part_key, :vector, :collection_name'
            conflict_target = '(id, part_key)'
        else:
            columns = 'id, vector, collection_name, text, vmetadata'
            values = ':id, :vector, :collection_name'
            conflict_target = '(id)'

        return (
            f'INSERT INTO document_chunk ({columns}) '
            f'VALUES ({values}, pgp_sym_encrypt(:text, :key), pgp_sym_encrypt(:metadata_text, :key)) '
            f'ON CONFLICT {conflict_target} {conflict_action}'
        )

    @staticmethod
    def _chunk_fields(part_key: Optional[str]) -> Dict[str, Any]:
        """Extra model/statement fields carried only when partitioning is on."""
        return {'part_key': part_key} if PGVECTOR_PARTITIONING else {}

    def adjust_vector_length(self, vector: List[float]) -> List[float]:
        # Adjust vector to have length VECTOR_LENGTH
        current_length = len(vector)
        if current_length < VECTOR_LENGTH:
            # Pad the vector with zeros
            vector += [0.0] * (VECTOR_LENGTH - current_length)
        elif current_length > VECTOR_LENGTH:
            # Truncate the vector to VECTOR_LENGTH
            vector = vector[:VECTOR_LENGTH]
        return vector

    def _insert_items(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            part_key = self._ensure_partition(collection_name)
            extra = self._chunk_fields(part_key)

            if PGVECTOR_PGCRYPTO:
                insert_sql = self._pgcrypto_insert_sql('DO NOTHING')
                for item in items:
                    vector = self.adjust_vector_length(item['vector'])
                    # Use raw SQL for BYTEA/pgcrypto
                    # Ensure metadata is converted to its JSON text representation
                    # Sanitize to strip null bytes / surrogates that PostgreSQL cannot store
                    json_metadata = sanitize_text_for_db(JSONCodec.dumps(item['metadata']))
                    item_text = sanitize_text_for_db(item['text'])
                    self.session.execute(
                        text(insert_sql),
                        {
                            'id': item['id'],
                            'vector': vector,
                            'collection_name': collection_name,
                            'text': item_text,
                            'metadata_text': json_metadata,
                            'key': PGVECTOR_PGCRYPTO_KEY,
                            **extra,
                        },
                    )
                self.session.commit()
                log.info("Encrypted & inserted %s into '%s'", len(items), collection_name)

            else:
                new_items = []
                for item in items:
                    vector = self.adjust_vector_length(item['vector'])
                    new_chunk = DocumentChunk(
                        id=item['id'],
                        vector=vector,
                        collection_name=collection_name,
                        text=item['text'],
                        vmetadata=process_metadata(item['metadata']),
                        **extra,
                    )
                    new_items.append(new_chunk)
                self.session.bulk_save_objects(new_items)
                self.session.commit()
                log.info("Inserted %s items into collection '%s'.", len(new_items), collection_name)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during insert: {e}')
            raise

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        self._write_with_partition_retry(collection_name, self._insert_items, items)

    def _upsert_items(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            part_key = self._ensure_partition(collection_name)
            extra = self._chunk_fields(part_key)

            if PGVECTOR_PGCRYPTO:
                upsert_sql = self._pgcrypto_insert_sql(
                    'DO UPDATE SET vector = EXCLUDED.vector, collection_name = EXCLUDED.collection_name, '
                    'text = EXCLUDED.text, vmetadata = EXCLUDED.vmetadata'
                )
                for item in items:
                    vector = self.adjust_vector_length(item['vector'])
                    # Sanitize to strip null bytes / surrogates that PostgreSQL cannot store
                    json_metadata = sanitize_text_for_db(JSONCodec.dumps(item['metadata']))
                    item_text = sanitize_text_for_db(item['text'])
                    self.session.execute(
                        text(upsert_sql),
                        {
                            'id': item['id'],
                            'vector': vector,
                            'collection_name': collection_name,
                            'text': item_text,
                            'metadata_text': json_metadata,
                            'key': PGVECTOR_PGCRYPTO_KEY,
                            **extra,
                        },
                    )
                self.session.commit()
                log.info("Encrypted & upserted %s into '%s'", len(items), collection_name)
            else:
                for item in items:
                    vector = self.adjust_vector_length(item['vector'])
                    # Scoped by part_key as well: looking up by id alone would
                    # scan every partition.
                    id_filter = [DocumentChunk.id == item['id']]
                    if PGVECTOR_PARTITIONING:
                        id_filter.append(DocumentChunk.part_key == part_key)
                    existing = self.session.query(DocumentChunk).filter(*id_filter).first()
                    if existing:
                        existing.vector = vector
                        existing.text = item['text']
                        existing.vmetadata = process_metadata(item['metadata'])
                        existing.collection_name = collection_name  # Update collection_name if necessary
                    else:
                        new_chunk = DocumentChunk(
                            id=item['id'],
                            vector=vector,
                            collection_name=collection_name,
                            text=item['text'],
                            vmetadata=process_metadata(item['metadata']),
                            **extra,
                        )
                        self.session.add(new_chunk)
                self.session.commit()
                log.info("Upserted %s items into collection '%s'.", len(items), collection_name)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during upsert: {e}')
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        self._write_with_partition_retry(collection_name, self._upsert_items, items)

    def _apply_iterative_scan(self) -> None:
        """Let pgvector keep walking the index until enough rows survive the filter.

        Without this the scan stops after the first `ef_search` (or `probes`)
        candidates, and whatever fraction of them belongs to the collection being
        searched is all the caller gets. The GUC is method-specific, and
        `strict_order` exists only for HNSW.

        pgvector registers its settings when its library is first loaded into a
        backend, and that only happens on first use of the vector type. Setting
        one before then raises `unrecognized configuration parameter`, which
        would abort the caller's transaction -- so the type is touched once per
        connection first. Loading is process-level, so it survives the rollback
        a read-only search ends with.
        """
        if PGVECTOR_ITERATIVE_SCAN == 'off':
            return

        # Resolved once in __init__: vector_index_configuration() logs at INFO
        # when PGVECTOR_INDEX_METHOD is set, and this runs on every search.
        method = self._index_method
        connection = self.session.connection()
        if not connection.info.get('pgvector_settings_registered'):
            self.session.execute(text("SELECT '[1]'::vector"))
            connection.info['pgvector_settings_registered'] = True

        if not self._supports_iterative_scan(method):
            return

        mode = 'relaxed_order' if method == 'ivfflat' else PGVECTOR_ITERATIVE_SCAN
        self.session.execute(text(f"SET LOCAL {method}.iterative_scan = '{mode}'"))

    def _supports_iterative_scan(self, method: str) -> bool:
        """Whether this pgvector build knows the setting. Looked up once."""
        if method in self._iterative_scan_support:
            return self._iterative_scan_support[method]

        supported = bool(
            self.session.execute(
                text('SELECT 1 FROM pg_settings WHERE name = :name'),
                {'name': f'{method}.iterative_scan'},
            ).scalar()
        )
        if not supported:
            log.info(
                'This pgvector build has no %s.iterative_scan; searches keep the default scan behaviour.',
                method,
            )
        self._iterative_scan_support[method] = supported
        return supported

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        try:
            if not vectors:
                return None

            # Adjust query vectors to VECTOR_LENGTH
            vectors = [self.adjust_vector_length(vector) for vector in vectors]
            num_queries = len(vectors)

            def vector_expr(vector):
                return cast(array(vector), VECTOR_TYPE_FACTORY(VECTOR_LENGTH))

            # Create the values for query vectors
            qid_col = column('qid', Integer)
            q_vector_col = column('q_vector', VECTOR_TYPE_FACTORY(VECTOR_LENGTH))
            query_vectors = (
                values(qid_col, q_vector_col)
                .data([(idx, vector_expr(vector)) for idx, vector in enumerate(vectors)])
                .alias('query_vectors')
            )

            result_fields = [
                DocumentChunk.id,
            ]
            if PGVECTOR_PGCRYPTO:
                result_fields.append(pgcrypto_decrypt(DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text).label('text'))
                result_fields.append(
                    pgcrypto_decrypt(DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB).label('vmetadata')
                )
            else:
                result_fields.append(DocumentChunk.text)
                result_fields.append(DocumentChunk.vmetadata)
            result_fields.append((DocumentChunk.vector.cosine_distance(query_vectors.c.q_vector)).label('distance'))

            # Build the lateral subquery for each query vector
            where_clauses = self._collection_scope(collection_name)

            # Apply metadata filter if provided
            if filter:
                for key, value in filter.items():
                    if isinstance(value, dict) and '$in' in value:
                        # Handle $in operator: {"field": {"$in": [values]}}
                        in_values = value['$in']
                        if PGVECTOR_PGCRYPTO:
                            where_clauses.append(
                                pgcrypto_decrypt(
                                    DocumentChunk.vmetadata,
                                    PGVECTOR_PGCRYPTO_KEY,
                                    JSONB,
                                )[key].astext.in_([str(v) for v in in_values])
                            )
                        else:
                            where_clauses.append(DocumentChunk.vmetadata[key].astext.in_([str(v) for v in in_values]))
                    else:
                        # Handle simple equality: {"field": "value"}
                        if PGVECTOR_PGCRYPTO:
                            where_clauses.append(
                                pgcrypto_decrypt(
                                    DocumentChunk.vmetadata,
                                    PGVECTOR_PGCRYPTO_KEY,
                                    JSONB,
                                )[key].astext
                                == str(value)
                            )
                        else:
                            where_clauses.append(DocumentChunk.vmetadata[key].astext == str(value))

            subq = (
                select(*result_fields)
                .where(*where_clauses)
                .order_by((DocumentChunk.vector.cosine_distance(query_vectors.c.q_vector)))
            )
            if limit is not None:
                subq = subq.limit(limit)
            subq = subq.lateral('result')

            # Build the main query by joining query_vectors and the lateral subquery
            stmt = (
                select(
                    query_vectors.c.qid,
                    subq.c.id,
                    subq.c.text,
                    subq.c.vmetadata,
                    subq.c.distance,
                )
                .select_from(query_vectors)
                .join(subq, true())
                .order_by(query_vectors.c.qid, subq.c.distance)
            )

            self._apply_iterative_scan()
            result_proxy = self.session.execute(stmt)
            results = result_proxy.all()

            ids = [[] for _ in range(num_queries)]
            distances = [[] for _ in range(num_queries)]
            documents = [[] for _ in range(num_queries)]
            metadatas = [[] for _ in range(num_queries)]

            if not results:
                # Same rollback as the populated path below. Without it the
                # session keeps its transaction, and with it the pooled
                # connection: one thread per empty search, until the pool is
                # gone. An empty result is not an unusual case -- it is what a
                # filtered index scan returns when none of its candidates
                # belong to the collection being searched.
                self.session.rollback()
                return SearchResult(
                    ids=ids,
                    distances=distances,
                    documents=documents,
                    metadatas=metadatas,
                )

            for row in results:
                qid = int(row.qid)
                ids[qid].append(row.id)
                # normalize and re-orders pgvec distance from [2, 0] to [0, 1] score range
                # https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
                distances[qid].append((2.0 - row.distance) / 2.0)
                documents[qid].append(row.text)
                metadatas[qid].append(row.vmetadata)

            self.session.rollback()  # read-only transaction
            return SearchResult(ids=ids, distances=distances, documents=documents, metadatas=metadatas)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during search: {e}')
            return None

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        vectors: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        hybrid_bm25_weight: float = 0.5,
    ) -> Optional[SearchResult]:
        if PGVECTOR_PGCRYPTO or filter:
            return None

        try:
            limit = max(1, limit)
            vectors = [self.adjust_vector_length(vector) for vector in vectors] if vectors else []
            num_queries = len(vectors) if vectors else 1
            bm25_weight = min(max(hybrid_bm25_weight, 0.0), 1.0)
            vector_weight = 1.0 - bm25_weight

            vector_result = None
            if vector_weight > 0 and vectors:
                vector_result = self.search(collection_name=collection_name, vectors=vectors, limit=limit)

            fts_results = []
            if bm25_weight > 0 and query and query.strip():
                # The part_key predicate sits in the main query, not the CTE, so
                # the planner still prunes to a single partition.
                part_key_clause = ''
                params = {'collection_name': collection_name, 'query': query, 'limit': limit}
                if PGVECTOR_PARTITIONING:
                    part_key_clause = 'AND document_chunk.part_key = :part_key'
                    params['part_key'] = part_key_for(collection_name)

                fts_rows = self.session.execute(
                    text(f"""
                        WITH fts_query AS (
                            SELECT plainto_tsquery('simple', :query) AS query
                        )
                        SELECT
                            document_chunk.id AS id,
                            document_chunk.text AS text,
                            document_chunk.vmetadata AS vmetadata,
                            ts_rank_cd(
                                to_tsvector('simple', coalesce(document_chunk.text, '')),
                                fts_query.query
                            ) AS rank
                        FROM document_chunk, fts_query
                        WHERE document_chunk.collection_name = :collection_name
                          {part_key_clause}
                          AND to_tsvector('simple', coalesce(document_chunk.text, '')) @@ fts_query.query
                        ORDER BY rank DESC
                        LIMIT :limit
                    """),
                    params,
                )
                fts_results = [dict(row) for row in fts_rows.mappings().all()]
                self.session.rollback()

            return merge_hybrid_search_results(
                vector_result=vector_result,
                fts_results=fts_results,
                num_queries=num_queries,
                limit=limit,
                hybrid_bm25_weight=hybrid_bm25_weight,
            )
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during hybrid search: {e}')
            return None

    def query(self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None) -> Optional[GetResult]:
        try:
            if PGVECTOR_PGCRYPTO:
                # Build where clause for vmetadata filter
                where_clauses = self._collection_scope(collection_name)
                for key, value in filter.items():
                    # decrypt then check key: JSON filter after decryption
                    where_clauses.append(
                        pgcrypto_decrypt(DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB)[key].astext
                        == str(value)
                    )
                stmt = select(
                    DocumentChunk.id,
                    pgcrypto_decrypt(DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text).label('text'),
                    pgcrypto_decrypt(DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB).label('vmetadata'),
                ).where(*where_clauses)
                if limit is not None:
                    stmt = stmt.limit(limit)
                results = self.session.execute(stmt).all()
            else:
                query = self.session.query(DocumentChunk).filter(*self._collection_scope(collection_name))

                for key, value in filter.items():
                    query = query.filter(DocumentChunk.vmetadata[key].astext == str(value))

                if limit is not None:
                    query = query.limit(limit)

                results = query.all()

            if not results:
                self.session.rollback()  # read-only transaction
                return None

            ids = [[result.id for result in results]]
            documents = [[result.text for result in results]]
            metadatas = [[result.vmetadata for result in results]]

            self.session.rollback()  # read-only transaction
            return GetResult(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during query: {e}')
            return None

    def get(self, collection_name: str, limit: Optional[int] = None) -> Optional[GetResult]:
        try:
            if PGVECTOR_PGCRYPTO:
                stmt = select(
                    DocumentChunk.id,
                    pgcrypto_decrypt(DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text).label('text'),
                    pgcrypto_decrypt(DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB).label('vmetadata'),
                ).where(*self._collection_scope(collection_name))
                if limit is not None:
                    stmt = stmt.limit(limit)
                results = self.session.execute(stmt).all()
                ids = [[row.id for row in results]]
                documents = [[row.text for row in results]]
                metadatas = [[row.vmetadata for row in results]]
            else:
                query = self.session.query(DocumentChunk).filter(*self._collection_scope(collection_name))
                if limit is not None:
                    query = query.limit(limit)

                results = query.all()

                if not results:
                    self.session.rollback()  # read-only transaction
                    return None

                ids = [[result.id for result in results]]
                documents = [[result.text for result in results]]
                metadatas = [[result.vmetadata for result in results]]

            self.session.rollback()  # read-only transaction
            return GetResult(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during get: {e}')
            return None

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if PGVECTOR_PGCRYPTO:
                wheres = self._collection_scope(collection_name)
                if ids:
                    wheres.append(DocumentChunk.id.in_(ids))
                if filter:
                    for key, value in filter.items():
                        wheres.append(
                            pgcrypto_decrypt(DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB)[key].astext
                            == str(value)
                        )
                stmt = DocumentChunk.__table__.delete().where(*wheres)
                result = self.session.execute(stmt)
                deleted = result.rowcount
            else:
                query = self.session.query(DocumentChunk).filter(*self._collection_scope(collection_name))
                if ids:
                    query = query.filter(DocumentChunk.id.in_(ids))
                if filter:
                    for key, value in filter.items():
                        query = query.filter(DocumentChunk.vmetadata[key].astext == str(value))
                deleted = query.delete(synchronize_session=False)
            self.session.commit()
            log.info("Deleted %s items from collection '%s'.", deleted, collection_name)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during delete: {e}')
            raise

    def reset(self) -> None:
        try:
            if PGVECTOR_PARTITIONING:
                # Drop the per-collection partitions outright and truncate the
                # buckets. A plain DELETE would leave one dead tuple per row for
                # autovacuum to chew through, which is the cost this whole
                # feature exists to avoid.
                for partition in self._dedicated_partitions():
                    self.session.execute(text(self._render_ddl('DROP TABLE IF EXISTS %I', partition)))
                self.session.execute(text(f'TRUNCATE {DocumentChunk.__tablename__}'))
                self.session.commit()
                with self._partition_lock:
                    self._known_part_keys = {
                        f'{BUCKET_PART_KEY_PREFIX}{bucket}' for bucket in range(PGVECTOR_PARTITION_BUCKETS)
                    }
                log.info("Reset complete. Dropped all collection partitions of 'document_chunk'.")
                return

            deleted = self.session.query(DocumentChunk).delete()
            self.session.commit()
            log.info("Reset complete. Deleted %s items from 'document_chunk' table.", deleted)
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error during reset: {e}')
            raise

    def _dedicated_partitions(self) -> List[str]:
        """Names of the per-collection partitions, buckets excluded."""
        return list(
            self.session.execute(
                text(
                    'SELECT c.relname FROM pg_inherits i '
                    'JOIN pg_class c ON c.oid = i.inhrelid '
                    'WHERE i.inhparent = to_regclass(:parent) AND c.relname NOT LIKE :buckets'
                ),
                {'parent': DocumentChunk.__tablename__, 'buckets': f'{BUCKET_PREFIX}%'},
            )
            .scalars()
            .all()
        )

    def close(self) -> None:
        pass

    def has_collection(self, collection_name: str) -> bool:
        try:
            exists = (
                self.session.query(DocumentChunk).filter(*self._collection_scope(collection_name)).first() is not None
            )
            self.session.rollback()  # read-only transaction
            return exists
        except Exception as e:
            self.session.rollback()
            log.exception(f'Error checking collection existence: {e}')
            return False

    def delete_collection(self, collection_name: str) -> None:
        if PGVECTOR_PARTITIONING and is_dedicated_collection(collection_name):
            self._drop_dedicated_partition(collection_name)
        else:
            # Bucketed collections share a partition with many others, so only
            # their own rows can go.
            self.delete(collection_name)
        log.info("Collection '%s' deleted.", collection_name)

    def _drop_dedicated_partition(self, collection_name: str) -> None:
        """Drop a collection's partition instead of deleting its rows.

        Instant, and it leaves nothing behind for autovacuum -- unlike a DELETE
        over a whole knowledge base, which leaves a dead tuple per row and drags
        every later vacuum pass through the vector index.
        """
        part_key = part_key_for(collection_name)
        partition = partition_name_for(part_key)
        try:
            self.session.rollback()
            self.session.execute(text(f"SET LOCAL lock_timeout = '{PARTITION_LOCK_TIMEOUT_MS}ms'"))
            self.session.execute(text(self._render_ddl('DROP TABLE IF EXISTS %I', partition)))
            self.session.commit()
            with self._partition_lock:
                self._known_part_keys.discard(part_key)
        except Exception as e:
            self.session.rollback()
            log.exception('Error dropping partition %s: %s', partition, e)
            raise
