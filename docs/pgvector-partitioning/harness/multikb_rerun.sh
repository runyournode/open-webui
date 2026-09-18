#!/bin/bash
# The one block of the campaigns that produced no valid figure: the multi-KB
# case at N=10 in configuration A under hnsw, at 1M and 3M. Those searches hit
# the connection leak in `search()` -- an empty result did not release its
# transaction -- and the pool's 30 s checkout timeout was measured instead of a
# latency.
#
# The leak is fixed. The migration is done in place, so the unpartitioned table
# of those campaigns no longer exists: a dataset is regenerated and A, B *and*
# C are measured on it, so the comparison stays internal to one dataset.
set -u
cd $REPO
SP=$EVIDENCE_DIR

ROWS=$1
TAG="multikb-${ROWS}"
DB="multikb_${ROWS}"
URL="postgresql://owui_test:owui_test@pgvector-test:5432/${DB}"
OUT="$SP/results/${TAG}.log"

. "$SP/runtime/pg_profile.env"
export PG_SHARED_BUFFERS PG_MAINTENANCE_WORK_MEM PG_WORK_MEM PG_SHM_SIZE
export OWUI_TEST_IMAGE_DIGEST=$(docker image inspect ghcr.io/open-webui/open-webui:dev --format '{{.Id}}')
if [ "$ROWS" -le 1200000 ]; then BUILD_MEM=6GB; else BUILD_MEM=16GB; fi

B="python -m tests.vector.bench_pgvector_partitioning"
M="python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate"
say() { echo; echo "######## $(date -Is) [$TAG] $*"; }

RUN() {
  timeout 36000 docker compose -f docker-compose.pgvector-test.yaml run --rm \
    -e PYTHONPATH=/app/backend -e VECTOR_DB=pgvector -e PGVECTOR_DB_URL="$URL" \
    -e PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH=1024 \
    -e PGVECTOR_INDEX_METHOD=hnsw -e PGVECTOR_HNSW_M="${HNSW_M:-16}" \
    -e PGVECTOR_HNSW_EF_CONSTRUCTION="${HNSW_EFC:-64}" \
    -e PGVECTOR_PARTITION_BUCKETS=32 -e PGVECTOR_PARTITIONING="${2:-false}" \
    tests "$1" 2>&1 | grep -vE "CORS_ALLOW|^\s*$|Container |^WARNING: Running pip"
  local status=${PIPESTATUS[0]}
  [ "$status" -ne 0 ] && { echo "!!!! $(date -Is) [$TAG] STEP FAILED (exit $status): $1"; exit "$status"; }
}

{
say "multi-KB re-run, $ROWS rows, generous profile, build_mem=$BUILD_MEM"
docker exec open-webui-pgvector-test-1 psql -U owui_test -d owui_test -q \
  -c "DROP DATABASE IF EXISTS $DB;" -c "CREATE DATABASE $DB;"

say "generate + verify"
RUN "$B generate --db-url '$URL' --rows $ROWS --dim 1024 --knowledge-bases 20 --kb-share 0.3 --centroids 500 --noises 5000"
say "embed"
RUN "$B embed --db-url '$URL' --dim 1024 --embed-url $OWUI_BENCH_EMBED_URL --embed-model $OWUI_BENCH_EMBED_MODEL"

for BUILD in 16:64 32:128; do
  HNSW_M=${BUILD%%:*}; HNSW_EFC=${BUILD##*:}; export HNSW_M HNSW_EFC
  say "global hnsw index (m=$HNSW_M, efc=$HNSW_EFC)"
  docker exec open-webui-pgvector-test-1 psql -U owui_test -d "$DB" -q -c "DROP INDEX IF EXISTS idx_document_chunk_vector;"
  RUN "$B index --db-url '$URL' --method hnsw --m $HNSW_M --ef-construction $HNSW_EFC --work-mem $BUILD_MEM"
  for config in A B; do
    say "multikb $config hnsw m=$HNSW_M,efc=$HNSW_EFC effort=400"
    RUN "$B multikb --db-url '$URL' --config $config --method hnsw --ef-search 400 --rounds 10"
  done
done

say "MIGRATE"
HNSW_M=16 HNSW_EFC=64 RUN "$M --migrate --db-url '$URL' --work-mem $BUILD_MEM --batch-size 50000" true
docker exec open-webui-pgvector-test-1 psql -U owui_test -d "$DB" -q -c "DROP TABLE IF EXISTS document_chunk_pre_partition CASCADE;"

for BUILD in 16:64 32:128; do
  HNSW_M=${BUILD%%:*}; HNSW_EFC=${BUILD##*:}; export HNSW_M HNSW_EFC
  say "partition hnsw indexes (m=$HNSW_M, efc=$HNSW_EFC)"
  RUN "$B partindex --db-url '$URL' --method hnsw --m $HNSW_M --ef-construction $HNSW_EFC --work-mem $BUILD_MEM" true
  say "multikb C hnsw m=$HNSW_M,efc=$HNSW_EFC effort=400"
  RUN "$B multikb --db-url '$URL' --config C --partitioned --method hnsw --ef-search 400 --rounds 10" true
done
say "re-run done"
} >> "$OUT" 2>&1
