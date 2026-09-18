#!/bin/bash
set -u
SP=$EVIDENCE_DIR
cd $REPO
cat > "$SP/runtime/pg_profile.env" <<'P'
PG_SHARED_BUFFERS=2GB
PG_MAINTENANCE_WORK_MEM=24GB
PG_WORK_MEM=256MB
PG_SHM_SIZE=26gb
P
. "$SP/runtime/pg_profile.env"; export PG_SHARED_BUFFERS PG_MAINTENANCE_WORK_MEM PG_WORK_MEM PG_SHM_SIZE
echo "$(date -Is) preflight: recreating postgres"
docker compose -f docker-compose.pgvector-test.yaml down -v >/dev/null 2>&1
docker compose -f docker-compose.pgvector-test.yaml up -d pgvector-test >/dev/null 2>&1
for _ in $(seq 1 60); do docker exec open-webui-pgvector-test-1 pg_isready -U owui_test -d owui_test >/dev/null 2>&1 && break; sleep 2; done
docker compose -f docker-compose.pgvector-test.yaml run --rm \
  -e PYTHONPATH=/app/backend -e DATABASE_URL="postgresql://owui_test:owui_test@pgvector-test:5432/owui_test" \
  tests "cd /app/backend/open_webui && alembic upgrade head" >/dev/null 2>&1
docker compose -f docker-compose.pgvector-test.yaml run --rm tests 2>&1 | tail -2
unexpected=$(docker logs open-webui-pgvector-test-1 2>&1 | grep -E "ERROR|FATAL|PANIC" \
  | grep -vE 'no partition of relation|violates check constraint "no_forbidden"|does not exist, skipping')
[ -n "$unexpected" ] && { echo "!!!! preflight: unexpected errors in the server log"; echo "$unexpected" | head; exit 1; }
echo "$(date -Is) preflight clean"
bash $EVIDENCE_DIR/harness/multikb_rerun.sh 1000000
echo "$(date -Is) 1M done (exit $?)"
bash $EVIDENCE_DIR/harness/multikb_rerun.sh 3000000
echo "$(date -Is) 3M done (exit $?)"
