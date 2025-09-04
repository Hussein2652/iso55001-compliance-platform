#!/usr/bin/env sh
set -e

cd /app

if [ -n "$DATABASE_URL" ]; then
  echo "Running Alembic migrations..."
  alembic -c backend/alembic.ini upgrade head || echo "Warning: Alembic failed; continuing"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000

