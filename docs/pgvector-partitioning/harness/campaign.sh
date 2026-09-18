#!/bin/bash
# One campaign = one (row count x memory profile), measured end to end for both
# index methods, two build parameterisations each, and four search efforts each.
#
#   campaign.sh <rows> <profile>
#
# Strictly sequential: nothing else may touch the database while it runs, and a
# step that is still progressing is never killed. A step that *fails* stops the
# campaign, because measuring past a failed index build produces numbers that
# look plausible and describe a sequential scan.
set -u
cd $REPO
SP=$EVIDENCE_DIR

ROWS=$1; PROFILE=$2
TAG="${ROWS}-${PROFILE}"
DB="bench_${ROWS}_${PROFILE}"
URL="postgresql://owui_test:owui_test@pgvector-test:5432/${DB}"
OUT="$SP/results/${TAG}.log"
STEP_TIMEOUT=${STEP_TIMEOUT:-36000}   # 10h per step; progress, not patience, is the stop condition
KB_LIMIT=10
QUERIES=20

# Every `docker compose` call must resolve the same PG_* values, or compose sees
# a config drift and silently recreates postgres with the compose defaults.
. "$SP/runtime/pg_profile.env"
export PG_SHARED_BUFFERS PG_MAINTENANCE_WORK_MEM PG_WORK_MEM PG_SHM_SIZE
export OWUI_TEST_IMAGE_DIGEST=$(docker image inspect ghcr.io/open-webui/open-webui:dev --format '{{.Id}}')

# maintenance_work_mem for index builds, sized to the data rather than to the
# host: pgvector reserves the whole amount as shared memory for the duration.
case "$PROFILE" in
  modest)   BUILD_MEM=256MB ;;
  *)        if   [ "$ROWS" -le 100000  ]; then BUILD_MEM=1GB
            elif [ "$ROWS" -le 300000  ]; then BUILD_MEM=2GB
            elif [ "$ROWS" -le 1200000 ]; then BUILD_MEM=6GB
            else                               BUILD_MEM=16GB
            fi ;;
esac

B="python -m tests.vector.bench_pgvector_partitioning"
M="python -m open_webui.retrieval.vector.dbs.pgvector_partition_migrate"
say() { echo; echo "######## $(date -Is) [$TAG] $*"; }

RUN() {  # $1 = command, $2 = partitioned flag, $3.. = extra env
  timeout "$STEP_TIMEOUT" docker compose -f docker-compose.pgvector-test.yaml run --rm \
    -e PYTHONPATH=/app/backend \
    -e VECTOR_DB=pgvector \
    -e PGVECTOR_DB_URL="$URL" \
    -e PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH=1024 \
    -e PGVECTOR_INDEX_METHOD="${METHOD:-hnsw}" \
    -e PGVECTOR_IVFFLAT_LISTS="${LISTS:-100}" \
    -e PGVECTOR_HNSW_M="${HNSW_M:-16}" \
    -e PGVECTOR_HNSW_EF_CONSTRUCTION="${HNSW_EFC:-64}" \
    -e PGVECTOR_PARTITION_BUCKETS=32 \
    -e PGVECTOR_PARTITIONING="${2:-false}" \
    tests "$1" 2>&1 | grep -vE "CORS_ALLOW|^\s*$|Container |^WARNING: Running pip"
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "!!!! $(date -Is) [$TAG] STEP FAILED (exit $status), abandoning this campaign: $1"
    exit "$status"
  fi
}

# Build parameterisations. Two per method, so a conclusion that only holds at
# one point in the tuning range shows up as one.
hnsw_builds() { echo "16:64 32:128"; }
ivfflat_builds() {
  local a b
  a=$(( ROWS / 1000 )); [ "$a" -lt 1 ] && a=1          # pgvector's guidance up to 1M rows
  b=$(python3 -c "import math;print(round(math.sqrt($ROWS)))")   # and over 1M
  [ "$a" -eq "$b" ] && b=$(( a * 4 ))                  # they coincide at exactly 1M
  echo "$a $b"
}
hnsw_efforts() { echo "10 40 100 400"; }
ivfflat_efforts() {  # 1, 10, sqrt(lists), lists/4
  python3 -c "
import math
l = $1
print(' '.join(str(v) for v in sorted({1, 10, round(math.sqrt(l)), max(1, l // 4)})))"
}

measure_set() {  # $1 = config, $2 = partitioned flag, $3 = efforts
  local config=$1 part=$2 efforts=$3 target effort
  for target in kb buckets; do
    for effort in $efforts; do
      say "$config $METHOD build=$BUILD_LABEL $target effort=$effort"
      RUN "$B measure --db-url '$URL' --config $config ${part:+--partitioned} --method $METHOD \
           --target $target --kb-limit $KB_LIMIT --queries-per-kb $QUERIES --ef-search $effort --query-mode self" "$part"
    done
  done
}

{
say "campaign start: rows=$ROWS profile=$PROFILE build_mem=$BUILD_MEM"
say "postgres settings in force"
docker exec open-webui-pgvector-test-1 psql -U owui_test -d owui_test -tAc \
  "select name||' = '||setting||' '||unit from pg_settings where name in ('shared_buffers','maintenance_work_mem','work_mem')"

docker exec open-webui-pgvector-test-1 psql -U owui_test -d owui_test -q \
  -c "DROP DATABASE IF EXISTS $DB;" -c "CREATE DATABASE $DB;"

say "generate + verify (30% knowledge base, 70% bucketed)"
RUN "$B generate --db-url '$URL' --rows $ROWS --dim 1024 --knowledge-bases 20 --kb-share 0.3 \
     --centroids 500 --noises 5000"

say "implant real embeddings in one knowledge base"
RUN "$B embed --db-url '$URL' --dim 1024 --embed-url $OWUI_BENCH_EMBED_URL --embed-model $OWUI_BENCH_EMBED_MODEL"

# ---------------------------------------------------------------- unpartitioned
for METHOD in hnsw ivfflat; do
  export METHOD
  if [ "$METHOD" = hnsw ]; then BUILDS=$(hnsw_builds); else BUILDS=$(ivfflat_builds); fi
  for BUILD in $BUILDS; do
    if [ "$METHOD" = hnsw ]; then
      HNSW_M=${BUILD%%:*}; HNSW_EFC=${BUILD##*:}; LISTS=100
      BUILD_LABEL="m=$HNSW_M,efc=$HNSW_EFC"; EFFORTS=$(hnsw_efforts)
      BUILD_ARGS="--m $HNSW_M --ef-construction $HNSW_EFC"
    else
      LISTS=$BUILD; HNSW_M=16; HNSW_EFC=64
      BUILD_LABEL="lists=$LISTS"; EFFORTS=$(ivfflat_efforts "$LISTS")
      BUILD_ARGS="--lists $LISTS"
    fi
    export HNSW_M HNSW_EFC LISTS

    say "drop any previous global vector index"
    docker exec open-webui-pgvector-test-1 psql -U owui_test -d "$DB" -q \
      -c "DROP INDEX IF EXISTS idx_document_chunk_vector;"

    say "build global $METHOD index ($BUILD_LABEL)"
    RUN "$B index --db-url '$URL' --method $METHOD $BUILD_ARGS --work-mem $BUILD_MEM"
    say "sizes unpartitioned ($METHOD $BUILD_LABEL)"
    RUN "$B sizes --db-url '$URL'"

    measure_set A '' "$EFFORTS"
    measure_set B '' "$EFFORTS"

    LOW=${EFFORTS%% *}; HIGH=${EFFORTS##* }
    for config in A B; do
      for effort in $LOW $HIGH; do
        say "needle $config $METHOD $BUILD_LABEL effort=$effort"
        RUN "$B needle --db-url '$URL' --config $config --method $METHOD --ef-search $effort"
      done
      say "multikb $config $METHOD $BUILD_LABEL effort=$HIGH"
      RUN "$B multikb --db-url '$URL' --config $config --method $METHOD --ef-search $HIGH --rounds 10"
    done
  done
done

say "churn unpartitioned (delete_collection + vacuum)"; RUN "$B churn --db-url '$URL'"
say "vacuum unpartitioned (raw DELETE + vacuum)";        RUN "$B vacuum --db-url '$URL'"

# ------------------------------------------------------------------ partitioned
say "MIGRATE"
METHOD=hnsw HNSW_M=16 HNSW_EFC=64 RUN "$M --migrate --db-url '$URL' --work-mem $BUILD_MEM --batch-size 50000" true
docker exec open-webui-pgvector-test-1 psql -U owui_test -d "$DB" -q \
  -c "DROP TABLE IF EXISTS document_chunk_pre_partition CASCADE;"

for METHOD in hnsw ivfflat; do
  export METHOD
  if [ "$METHOD" = hnsw ]; then BUILDS=$(hnsw_builds); else BUILDS=$(ivfflat_builds); fi
  for BUILD in $BUILDS; do
    if [ "$METHOD" = hnsw ]; then
      HNSW_M=${BUILD%%:*}; HNSW_EFC=${BUILD##*:}; LISTS=100
      BUILD_LABEL="m=$HNSW_M,efc=$HNSW_EFC"; EFFORTS=$(hnsw_efforts)
      BUILD_ARGS="--m $HNSW_M --ef-construction $HNSW_EFC"
    else
      LISTS=$BUILD; HNSW_M=16; HNSW_EFC=64
      BUILD_LABEL="lists=$LISTS"; EFFORTS=$(ivfflat_efforts "$LISTS")
      BUILD_ARGS="--lists $LISTS"
    fi
    export HNSW_M HNSW_EFC LISTS

    say "rebuild partition indexes ($METHOD $BUILD_LABEL)"
    RUN "$B partindex --db-url '$URL' --method $METHOD $BUILD_ARGS --work-mem $BUILD_MEM" true
    say "sizes partitioned ($METHOD $BUILD_LABEL)"
    RUN "$B sizes --db-url '$URL'" true

    measure_set C 'true' "$EFFORTS"

    LOW=${EFFORTS%% *}; HIGH=${EFFORTS##* }
    for effort in $LOW $HIGH; do
      say "needle C $METHOD $BUILD_LABEL effort=$effort"
      RUN "$B needle --db-url '$URL' --config C --partitioned --method $METHOD --ef-search $effort" true
    done
    say "multikb C $METHOD $BUILD_LABEL effort=$HIGH"
    RUN "$B multikb --db-url '$URL' --config C --partitioned --method $METHOD --ef-search $HIGH --rounds 10" true

    # ivfflat only: the same partitions with `lists` sized from each partition's
    # own row count, which is what a global setting cannot express.
    if [ "$METHOD" = ivfflat ] && [ "$BUILD" = "$(ivfflat_builds | cut -d' ' -f1)" ]; then
      say "rebuild partition indexes (ivfflat, lists sized per partition)"
      RUN "$B partindex --db-url '$URL' --method ivfflat --lists auto --work-mem $BUILD_MEM" true
      BUILD_LABEL="lists=auto"
      measure_set C 'true' "$EFFORTS"
    fi
  done
done

say "churn partitioned (delete_collection + vacuum)"; RUN "$B churn --db-url '$URL' --partitioned" true
say "vacuum partitioned (raw DELETE + vacuum)";        RUN "$B vacuum --db-url '$URL'" true
say "campaign done"
} >> "$OUT" 2>&1
