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
- `/v1` envelope routes — consistent pagination `{ items, total, limit, offset }` (legacy routes still supported):
  - `GET /v1/assessments`
  - `GET /v1/audits`
  - `GET /v1/nonconformities`
  - `GET /v1/management-reviews`
  - `GET /v1/clauses`
 - Organizations (admin):
   - `POST /v1/orgs` — create organization (idempotent)
   - `GET /v1/orgs?limit=&offset=` — list
   - `GET /v1/orgs/{id}` — detail
   - `PATCH /v1/orgs/{id}` — update (name)
 - Evidence (S3 pre‑signed):
   - `POST /attachments/presign_upload` — get pre‑signed PUT URL (editor role; rate‑limited).
   - `POST /attachments/complete` — record uploaded object metadata (editor role).
   - `GET /attachments/{attachment_id}/download_url` — pre‑signed GET URL when stored in object storage.
   - Dev file uploads still supported via `POST /assessments/{id}/attachments` and `GET /attachments/{id}/download`.
- `POST /setup` — admin-only idempotent setup: creates default org (if `DEFAULT_ORG_ID` is set) and seeds clauses when empty.
   - Optional body: `{ run_migrations: true, reseed_clauses: false, verify_object_store: true }`.
  - Interactive filters: date range and severity; renders a simple trend chart.

Pagination
- All list endpoints support `limit` and `offset` and set `X-Total-Count` header with the total rows for the applied filters.

Production Notes
- Switch to Postgres by setting `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@host/db`).
- Put Uvicorn behind a reverse proxy (e.g., Nginx) and set `--workers` for concurrency.
- Logging: Structured JSON logs to stdout, include `X-Request-ID`, method, path, status, duration_ms, user-agent, and client IP.
- Harden CORS and auth (e.g., SSO/OIDC) before exposing publicly.
- Backups for the DB and evidence attachments (to be added if we manage files).
 - Prod Compose: `infrastructure/compose.prod.yaml` uses Postgres and MinIO. Copy `infrastructure/.env.example` to `.env` and configure secrets.
 - Auto-migrations: the container entrypoint runs `alembic upgrade head` on start.
 - Org scoping: in dev you can set `X-Org-ID` header; in prod derive org from JWT claims.
 - Rate limits: uploads/presign/exports are lightly rate‑limited; body size capped by `MAX_UPLOAD_MB`.
- Tracing (optional): set `OTEL_ENABLED=true` and (optionally) `OTEL_EXPORTER_OTLP_ENDPOINT` to emit traces.
 - Security headers: HSTS (prod), `X-Content-Type-Options: nosniff`, and a conservative CSP are applied.
 - Auth policy (prod): `ENV=prod` enforces JWT RS256 via JWKS; static `API_TOKEN`/HS256 accepted only outside prod.

Migrations (Alembic)
- Configure DB: `export DATABASE_URL=sqlite:////absolute/path/to/app.db` (or Postgres URL)
- Upgrade: `alembic -c backend/alembic.ini upgrade head`
- In Docker: `docker compose exec backend alembic -c backend/alembic.ini upgrade head`
- Includes initial tables and follow-up migrations for `audits`, `nonconformities`, and `management_reviews`.
 - Additional migration adds:
   - `organizations` table.
   - `org_id`, `created_by`, `updated_by`, `request_id` on business tables, plus helpful indices.

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
 - `ENV`: `dev` or `prod` (prod enforces RS256/JWKS and ignores static token).
 - `CORS_ALLOWED_ORIGINS`: Comma-separated origins (prod should not use `*`).
 - `DEFAULT_ORG_ID`: Optional default org id for dev scoping (header `X-Org-ID` overrides).
- Object store (S3/MinIO):
   - `OBJECT_STORE_ENDPOINT` (e.g., `http://minio:9000`)
   - `OBJECT_STORE_BUCKET`
   - `OBJECT_STORE_ACCESS_KEY`
   - `OBJECT_STORE_SECRET_KEY`
   - `OBJECT_STORE_USE_PATH_STYLE` (`true`/`false`)
 - Upload limits: `MAX_UPLOAD_MB` (default `25`).
  - Presign TTL: `PRESIGN_TTL_SECONDS` (default `900`).
  - MIME allow-list override: `ALLOWED_UPLOAD_MEDIA` (CSV of MIME types).
- OpenTelemetry:
  - `OTEL_ENABLED` (`true`/`false`)
  - `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g., `http://otel-collector:4318`)

Org Scoping
- Dev: provide `X-Org-ID` header; queries are filtered by `org_id` and writes store it alongside `created_by`, `updated_by`, and `request_id`.
- Prod: derive org id from JWT claims (recommended); header is for dev/testing only.

Metrics
- Prometheus `/metrics` includes:
  - `http_requests_total`
  - `http_requests_errors_total`
  - `http_request_duration_seconds{method,route,status}`

Deployment Modes
- Dev:
  - Static `API_TOKEN` allowed; HS256 accepted.
  - Org scoping via `X-Org-ID` header (or `DEFAULT_ORG_ID`).
  - CORS defaults to `*` if not set.
- Prod:
  - JWT RS256 required (JWKS-based); static token/HS256 disabled.
  - Org id is derived from JWT claims; header is ignored.
  - CORS must be explicitly configured via `CORS_ALLOWED_ORIGINS`.

Fast Wins Next
- Roll `/v1` envelope across audits, nonconformities, and reviews.
- Add strict content-type allow-list and optional SHA-256 recording for S3 objects.
- Enforce org match on update/delete operations.
- Generate Insomnia/Postman collection and add `/setup` endpoint to seed clauses idempotently.

High‑Priority TODOs (Start Here)

Foundations & Hardening
- Postgres default in prod; keep SQLite for dev.
- Enforce OIDC RS256 (JWKS) auth in prod; disable static token in prod.
- Tenant guard: inject `org_id` from JWT; filter every query; forbid header override in prod.
- Ship `organizations` migration + indices (`org_id, status, due_date`, `created_at`, etc.).
- Structured JSON logs with `request_id`, `user`, `org`, latency.
- RED metrics per route and OTEL spans when enabled.

Evidence Governance (7.5)
- Ensure bucket on startup; presign TTL 10–15 min.
- MIME allow‑list and size limits before presign.
- Store SHA‑256 and retention/hold/disposition fields on finalize (optional AV scan).

Planning Core (Clause 6)
- SAMP studio (6.2.1) with versioning and finance alignment.
- Objectives & AMPs (6.2.2–6.2.3) linked to KPIs; review cadence.
- Risks & Opportunities (6.1.2–6.1.3) with treatment & effectiveness.
- Planned Changes (6.3) with pre‑implementation risk assessment.

Operation (Clause 8)
- Lifecycle controls matrix (create→dispose) with criteria & evidence.
- Change control (8.2) incl. unintended consequence review.
- Suppliers (8.3) with scope/interfaces/responsibilities/sharing + monitoring.

Performance & Governance (Clause 9)
- KPI engine (9.1): catalog, schedules, adapters, trend charts.
- Internal audit program (9.2): templates, sampling, impartial assignment, findings workflow.
- Management review pack (9.3): one‑click report (inputs, trends, risk/opportunity profile, framework effectiveness) → PDF/Word.

Improvement (Clause 10)
- NC/CAPA: RCA templates, effectiveness verification, recurrence‑prevention flag; closure‑time KPI.
- Predictive actions (10.3): v1 interface for condition/performance/cost inputs → suggested intervention points; v2 pluggable model API.

API & Security Polish
- Move `/v1` prefix everywhere; unify list responses to `{items,total,limit,offset}`; keep `X-Total-Count`.
- Add error models/examples; stricter CORS + security headers (HSTS, nosniff, CSP).
- Tune rate limits and upload body caps.

CI/CD & Testing
- GitHub Actions: ruff + mypy + pytest + docker build + Trivy scan + push to GHCR.
- Integration tests (Testcontainers Postgres); coverage ≥ 60%.
- Ship Insomnia collection, `/setup` seed endpoint, and complete `.env.example`.

UI/UX (initial)
- OIDC login; role‑aware navigation.
- Checklist + evidence upload (presign); inline gap scoring.
- CRUD screens for SAMP/AMP, risks, KPIs, audits, NCs, reviews; exports & charts.
- Dashboard: readiness %, clause heatmap, due items, audit schedule, NC backlog, KPI trend.

Docs & Ops
- Admin & auditor guides; API reference; architecture overview.
- Backups & retention runbooks (DB PITR, object lifecycle).
- Observability playbooks (alerts, log queries, tracing how‑to).

CI
- GitHub Actions workflow at `.github/workflows/ci.yml` runs lint, type-checks, tests, and Docker build on push/PR.
 - Integration job spins up Postgres and runs an Alembic smoke test.

Tools
- Insomnia collection: see `docs/insomnia_export.json`. Import and set `baseUrl` and `token` in the environment.
- Postman collection: see `docs/postman_collection.json`. Import and set collection variables `baseUrl` and `token`.
 - Postman environment: see `docs/postman_environment.json` for local defaults.
 - OpenAPI: enriched with tags, error models, and request/response examples on v1 list routes and core write operations.
 - `OIDC_JWKS`: Inline JWKS JSON string for RS256 verification.
 - `OIDC_JWKS_PATH`: Filesystem path to JWKS JSON.
 - `OIDC_JWKS_URL`: JWKS URL (requires network access).

Next Steps
- Add authentication/authorization (OIDC/JWT).
- Add document/evidence file attachments and storage (S3 compatible).
- Expand data model: audits, management reviews, corrective actions, KPIs.
- Frontend: simple React/Vue app for assessments dashboard.
