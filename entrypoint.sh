#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

# Start Uvicorn
echo "Starting Uvicorn..."

PORT=${PORT}
HOST=0.0.0.0
WORKERS=1
TIMEOUT=60

exec uvicorn app.main:create_app --factory \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --timeout-keep-alive $TIMEOUT \
    --log-level info