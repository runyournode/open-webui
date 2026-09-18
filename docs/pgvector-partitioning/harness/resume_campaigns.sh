#!/bin/bash
# Every campaign, one after another. Never two at once: a second campaign would
# compete for the same page cache and the same disk, and the latencies would
# describe the contention rather than the index.
set -u
cd $REPO
SP=$EVIDENCE_DIR
mkdir -p "$SP/results"

write_profile() {
  case "$1" in
    generous) cat > "$SP/runtime/pg_profile.env" <<'P'
PG_SHARED_BUFFERS=2GB
PG_MAINTENANCE_WORK_MEM=24GB
PG_WORK_MEM=256MB
PG_SHM_SIZE=26gb
P
    ;;
    modest) cat > "$SP/runtime/pg_profile.env" <<'P'
PG_SHARED_BUFFERS=512MB
PG_MAINTENANCE_WORK_MEM=256MB
PG_WORK_MEM=16MB
PG_SHM_SIZE=1gb
P
    ;;
  esac
}

# A campaign that starts on a dirty server log does not start at all. The
# exercise is the functional suite, which drives every operation of the client
# in both layouts -- so anything the code does wrong at the SQL level shows up
# here rather than being discovered in a log three hours into a campaign.
preflight() {
  cd $REPO
  . "$SP/runtime/pg_profile.env"; export PG_SHARED_BUFFERS PG_MAINTENANCE_WORK_MEM PG_WORK_MEM PG_SHM_SIZE
  echo "$(date -Is) preflight: recreating postgres on the $1 profile"
  docker compose -f docker-compose.pgvector-test.yaml down -v >/dev/null 2>&1
  docker compose -f docker-compose.pgvector-test.yaml up -d pgvector-test >/dev/null 2>&1
  for _ in $(seq 1 60); do
    docker exec open-webui-pgvector-test-1 pg_isready -U owui_test -d owui_test >/dev/null 2>&1 && break
    sleep 2
  done
  docker exec open-webui-pgvector-test-1 psql -U owui_test -d owui_test -tAc \
    "select name||'='||setting from pg_settings where name in ('shared_buffers','maintenance_work_mem','work_mem')"

  # query_collection reads the application's own settings table, so the app
  # schema has to exist before the multi-knowledge-base step can run.
  docker compose -f docker-compose.pgvector-test.yaml run --rm \
    -e PYTHONPATH=/app/backend \
    -e DATABASE_URL="postgresql://owui_test:owui_test@pgvector-test:5432/owui_test" \
    tests "cd /app/backend/open_webui && alembic upgrade head" >/dev/null 2>&1

  docker compose -f docker-compose.pgvector-test.yaml run --rm tests 2>&1 | tail -2

  # Two of these the suite raises on purpose, and they are the point of the
  # tests that raise them: one asserts that a write to a collection whose
  # partition does not exist yet is retried rather than lost, the other that a
  # constraint violation from inside a partition is *not* mistaken for a
  # missing partition. Anything else is a defect, and this is where it has to
  # be caught -- not three hours into a campaign.
  local expected='no partition of relation|violates check constraint "no_forbidden"|does not exist, skipping'
  local unexpected
  unexpected=$(docker logs open-webui-pgvector-test-1 2>&1 | grep -E "ERROR|FATAL|PANIC" | grep -vE "$expected")
  if [ -n "$unexpected" ]; then
    echo "!!!! preflight failed: unexpected errors in the server log, refusing to start a campaign"
    echo "$unexpected" | head
    return 1
  fi
  echo "$(date -Is) preflight clean (only the errors the suite provokes deliberately)"
}

# Resume after a host restart: a campaign whose log ends with "campaign done"
# is complete and is not redone. Everything else starts over, because an
# interrupted campaign left a half-indexed database, and measuring on it would
# give plausible, wrong numbers.
skip_or_run() {
  local log="$SP/results/$1-$2.log"
  if [ -f "$log" ] && tail -3 "$log" | grep -q "campaign done"; then
    echo "$(date -Is) ==== campaign $1/$2 already complete, skipped ===="
    return 0
  fi
  run_one "$1" "$2"
}

run_one() {  # rows, profile
  write_profile "$2"
  preflight "$2" || exit 1
  echo "$(date -Is) ==== campaign $1 rows, $2 profile ===="
  bash $EVIDENCE_DIR/harness/campaign.sh "$1" "$2"
  echo "$(date -Is) ==== campaign $1/$2 finished with status $? ===="
}

skip_or_run 200000 modest
skip_or_run 200000 generous
skip_or_run 1000000 generous
skip_or_run 3000000 generous
echo "$(date -Is) all campaigns done"
