#!/bin/bash
# Query-mode trial: does recall change when the query is not a stored row?
# 200k rows, HNSW m=16/efc=64, ef_search=40, three query modes, A/B/C.
set -u
cd $REPO
SP=$EVIDENCE_DIR
. "$SP/runtime/pg_profile.env"; export PG_SHARED_BUFFERS PG_MAINTENANCE_WORK_MEM PG_WORK_MEM PG_SHM_SIZE
export OWUI_TEST_IMAGE_DIGEST=$(docker image inspect ghcr.io/open-webui/open-webui:dev --format '{{.Id}}')
DB=bench_200000_d5; URL="postgresql://owui_test:owui_test@pgvector-test:5432/${DB}"
OUT="$SP/results/d5-query-mode-200000.log"
B="python -m tests.vector.bench_pgvector_partitioning"
M="python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate"
say() { echo; echo "######## $(date -Is) [d5] $*"; }
RUN() {
  timeout 3600 docker compose -f docker-compose.pgvector-test.yaml run --rm \
    -e PYTHONPATH=/app/backend -e VECTOR_DB=pgvector -e PGVECTOR_DB_URL="$URL" \
    -e PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH=1024 -e PGVECTOR_INDEX_METHOD=hnsw \
    -e PGVECTOR_HNSW_M=16 -e PGVECTOR_HNSW_EF_CONSTRUCTION=64 \
    -e PGVECTOR_PARTITION_BUCKETS=32 -e PGVECTOR_PARTITIONING="${2:-false}" \
    tests "$1" 2>&1 | grep -vE "CORS_ALLOW|^\s*$|Container |^WARNING: Running pip"
  local status=${PIPESTATUS[0]}
  [ "$status" -ne 0 ] && { echo "!!!! $(date -Is) [d5] STEP FAILED (exit $status): $1"; exit "$status"; }
}
{
say "start"
docker exec open-webui-pgvector-test-1 psql -U owui_test -d owui_test -q -c "DROP DATABASE IF EXISTS $DB;" -c "CREATE DATABASE $DB;"
say "generate"; RUN "$B generate --db-url '$URL' --rows 200000 --dim 1024 --knowledge-bases 20 --kb-share 0.3 --centroids 500 --noises 5000"
say "embed";    RUN "$B embed --db-url '$URL' --dim 1024 --embed-url $OWUI_BENCH_EMBED_URL --embed-model $OWUI_BENCH_EMBED_MODEL"
say "index hnsw m=16"; RUN "$B index --db-url '$URL' --method hnsw --m 16 --ef-construction 64 --work-mem 2GB"
for mode in self perturbed exclude-self; do
  for config in A B; do
    say "$config kb ef=40 mode=$mode"
    RUN "$B measure --db-url '$URL' --config $config --method hnsw --target kb --kb-limit 10 --queries-per-kb 20 --ef-search 40 --query-mode $mode"
  done
done
say "migrate"; RUN "$M --migrate --db-url '$URL' --work-mem 2GB --batch-size 50000" true
docker exec open-webui-pgvector-test-1 psql -U owui_test -d "$DB" -q -c "DROP TABLE IF EXISTS document_chunk_pre_partition CASCADE;"
say "partindex hnsw m=16"; RUN "$B partindex --db-url '$URL' --method hnsw --m 16 --ef-construction 64 --work-mem 2GB" true
for mode in self perturbed exclude-self; do
  say "C kb ef=40 mode=$mode"
  RUN "$B measure --db-url '$URL' --config C --partitioned --method hnsw --target kb --kb-limit 10 --queries-per-kb 20 --ef-search 40 --query-mode $mode" true
done
say "end"
} >> "$OUT" 2>&1
