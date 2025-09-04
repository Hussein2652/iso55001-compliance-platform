ISO 55001 Compliance Platform (MVP)

Overview
- Goal: Stand up a minimal, production-ready API to manage ISO 55001 clauses and capture compliance assessments (evidence, status, owners, due dates). Build on provided standard materials in `data/`.
- Inputs: `data/project_files/iso55001_clauses_seed.json` provides the seed list of ISO 55001:2024 clauses.

MVP Scope
- Clauses: Serve and search ISO 55001 clauses from the seed JSON.
- Assessments: Create/read/update assessment records per clause with status, evidence, owner, and due date.
- Health + CORS: Basic health endpoint and CORS enabled for a future frontend.

Security
- Bearer token (optional): Set `API_TOKEN` to protect write endpoints. If unset, dev mode allows writes without auth. In production, set a strong `API_TOKEN` or enable OIDC.
- OIDC/JWT (HS256): Set `OIDC_HS256_SECRET` (and optionally `OIDC_ISSUER`, `OIDC_AUDIENCE`) to require a JWT in `Authorization: Bearer`. If both `OIDC_HS256_SECRET` and `API_TOKEN` are set, non‑JWT tokens use static token validation and JWT-shaped tokens use JWT validation.
- Roles (RBAC):
  - Roles: `viewer` (read-only), `editor` (create/update, upload), `admin` (all + delete).
  - Static token role via `API_ROLE` (default: `admin`).
  - JWT roles from `roles` array, `role` string, or `scope` (space-separated).
 - RS256 OIDC via JWKS:
   - Provide `OIDC_JWKS` (inline JSON), or `OIDC_JWKS_PATH` (file path), or `OIDC_JWKS_URL`.
   - If both JWT and static token are configured, non-JWT tokens use static token; JWT tokens choose HS256/RS256 based on `alg` header.

Tech (backend)
- FastAPI + Uvicorn for a lightweight REST API.
- SQLite by default (file DB) for dev; designed to swap to Postgres via `DATABASE_URL`.
- Zero external services required to run locally.

Run (local)
- Prereqs: Python 3.10+ recommended.
- From repo root:
  - `cd backend`
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn app.main:app --reload` (default http://127.0.0.1:8000)

Run (Docker)
- From repo root:
  - `docker build -t iso55001-backend ./backend`
  - `docker run -p 8000:8000 --name iso55001-backend iso55001-backend`

Run (Compose)
- From repo root:
  - SQLite (default): `docker compose -f infrastructure/compose.yaml up --build`
  - Postgres: uncomment `DATABASE_URL` pointing to `db` and set `API_TOKEN` in `infrastructure/compose.yaml`, then run the same command.

API (high level)
- `GET /health` — service status.
- `GET /metrics` — Prometheus metrics (http_requests_total).
  - Includes `http_request_duration_seconds` histogram labeled by `method`, `route`, and `status`.
- `GET /clauses?q=&limit=&offset=` — list/search with pagination.
- `GET /clauses/{clause_id}` — clause detail.
- `POST /assessments` — create assessment (status enum, optional ISO date).
- `GET /assessments?clause_id=&status=&limit=&offset=` — list with filters + pagination.
- `GET /assessments/{id}` — assessment detail.
- `PATCH /assessments/{id}` — partial update.
- `POST /assessments/{id}/attachments` — upload evidence file (auth).
- `GET /assessments/{id}/attachments` — list evidence attachments.
- `GET /attachments/{attachment_id}/download` — download evidence file.
- `DELETE /attachments/{attachment_id}` — delete evidence (auth).
- `POST /audits` — create audit (editor role).
- `GET /audits?status=&limit=&offset=` — list audits.
- `GET /audits/{id}` — audit detail.
- `PATCH /audits/{id}` — update title/description/status/dates (editor role).
- `POST /nonconformities` — create nonconformity linked to audit and/or clause (editor role).
- `GET /nonconformities?status=&severity=&clause_id=&audit_id=&limit=&offset=` — list/filter.
- `GET /nonconformities/{id}` — detail.
- `PATCH /nonconformities/{id}` — update fields, including status and closure info (editor role).
- `POST /management-reviews` — create review (editor role).
- `GET /management-reviews?limit=&offset=` — list reviews.
- `GET /management-reviews/{id}` — review detail.
- `PATCH /management-reviews/{id}` — update review (editor role).
- `GET /kpi/overview` — summary counts (audits by status, NCs by status/severity, overdue NCs, assessments by status).
  - Filters: `created_from`, `created_to`, `as_of` (for overdue calculations).
- `GET /kpi/nc_trends?from_date=&to_date=&severity=` — monthly opened/closed counts (YYYY-MM) for Nonconformities.
- `GET /kpi/overdue_nonconformities?days=` — list open overdue NCs with `days_overdue`.
- `GET /export/audits.csv` — CSV export of audits.
- Filters: `status`, `scheduled_from`, `scheduled_to`, `completed_from`, `completed_to`.
- `GET /export/nonconformities.csv` — CSV export of NCs.
- Filters: `status`, `severity`, `clause_id`, `audit_id`, `created_from`, `created_to`, `due_from`, `due_to`.
- `GET /export/audits.xlsx` — Excel export of audits.
- Filters: `status`.
- `GET /export/nonconformities.xlsx` — Excel export of NCs.
  - Filters: `status`, `severity`.
- `GET /export/assessments.csv|.xlsx` — export assessments.
  - Filters: CSV `clause_id`, `status`, `owner`, `created_from`, `created_to`, `due_from`, `due_to`; XLSX `status`.
- `GET /export/audit_nonconformities.csv|.xlsx` — joined export of Audits with associated NCs.
  - Filters: CSV `audit_id`, `status`, `severity`.
- `GET /dashboard` — static dashboard with KPIs and overdue NCs.
  - Interactive filters: date range and severity; renders a simple trend chart.

Pagination
- All list endpoints support `limit` and `offset` and set `X-Total-Count` header with the total rows for the applied filters.

Production Notes
- Switch to Postgres by setting `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@host/db`).
- Put Uvicorn behind a reverse proxy (e.g., Nginx) and set `--workers` for concurrency.
- Logging: Structured JSON logs to stdout, include `X-Request-ID`, method, path, status, duration_ms, user-agent, and client IP.
- Harden CORS and auth (e.g., SSO/OIDC) before exposing publicly.
- Backups for the DB and evidence attachments (to be added if we manage files).

Migrations (Alembic)
- Configure DB: `export DATABASE_URL=sqlite:////absolute/path/to/app.db` (or Postgres URL)
- Upgrade: `alembic -c backend/alembic.ini upgrade head`
- In Docker: `docker compose exec backend alembic -c backend/alembic.ini upgrade head`
 - Includes initial tables and follow-up migrations for `audits`, `nonconformities`, and `management_reviews`.

Environment Variables
- `DATABASE_URL`: SQLAlchemy URL (SQLite or Postgres). Examples:
  - `sqlite:////absolute/path/to/app.db`
  - `postgresql+psycopg2://user:pass@host:5432/dbname`
- `API_TOKEN`: Optional Bearer token for write endpoints. Example: `export API_TOKEN=$(openssl rand -hex 24)`
- `EVIDENCE_DIR`: Filesystem path to store uploaded evidence (default: `backend/evidence`). Ensure durable storage in production (e.g., mounted volume or S3 gateway).
- `OIDC_HS256_SECRET`: Use HS256 JWT validation for writes (instead of `API_TOKEN`).
- `OIDC_ISSUER`: Optional issuer claim to validate.
- `OIDC_AUDIENCE`: Optional audience claim to validate.
- `API_ROLE`: Role used when `API_TOKEN` is active (`viewer`|`editor`|`admin`, default `admin`).
- `LOG_LEVEL`: Logging level (default `INFO`).
 - `OIDC_JWKS`: Inline JWKS JSON string for RS256 verification.
 - `OIDC_JWKS_PATH`: Filesystem path to JWKS JSON.
 - `OIDC_JWKS_URL`: JWKS URL (requires network access).

Next Steps
- Add authentication/authorization (OIDC/JWT).
- Add document/evidence file attachments and storage (S3 compatible).
- Expand data model: audits, management reviews, corrective actions, KPIs.
- Frontend: simple React/Vue app for assessments dashboard.
