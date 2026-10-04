#!/bin/sh
set -u
CORES="$(nproc 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="$CORES"
export MKL_NUM_THREADS="$CORES"
export OPENBLAS_NUM_THREADS="$CORES"
export NUMEXPR_NUM_THREADS="$CORES"
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
