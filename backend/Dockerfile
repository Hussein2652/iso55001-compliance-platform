FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

# System deps (optional, kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r requirements.txt

# Copy the whole repo so we can access the seed JSON in ../data/
COPY . /app

# Expose and run
EXPOSE 8000
ENV DATABASE_URL="sqlite:////app/backend/app.db"
COPY backend/start.sh /app/backend/start.sh
RUN chmod +x /app/backend/start.sh
CMD ["/app/backend/start.sh"]
