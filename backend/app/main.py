import json
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from contextlib import asynccontextmanager
from .auth import require_write_auth, role_required
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from .logging_config import get_logger
import time
from enum import Enum
import csv
import io
from io import BytesIO
import uuid
import shutil


# --- Config ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR}/backend/app.db")
EVIDENCE_DIR = Path(os.getenv("EVIDENCE_DIR", ROOT_DIR / "backend" / "evidence")).resolve()

# SQLAlchemy engine and metadata
engine = create_engine(DEFAULT_DB_URL, future=True)
metadata = MetaData()

clauses_table = Table(
    "clauses",
    metadata,
    Column("clause_id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("summary", String, nullable=False),
)

assessments_table = Table(
    "assessments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("clause_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("evidence", String),
    Column("owner", String),
    Column("due_date", String),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

attachments_table = Table(
    "attachments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("assessment_id", Integer, nullable=False),
    Column("filename", String, nullable=False),
    Column("content_type", String),
    Column("size", Integer),
    Column("stored_path", String, nullable=False),
    Column("created_at", String, nullable=False),
)

# Audits and Nonconformities
audits_table = Table(
    "audits",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String, nullable=False),
    Column("description", String),
    Column("status", String, nullable=False),
    Column("scheduled_date", String),
    Column("completed_date", String),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

nonconformities_table = Table(
    "nonconformities",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("audit_id", Integer),
    Column("clause_id", String),
    Column("severity", String, nullable=False),
    Column("status", String, nullable=False),
    Column("description", String, nullable=False),
    Column("corrective_action", String),
    Column("owner", String),
    Column("due_date", String),
    Column("closed_date", String),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

management_reviews_table = Table(
    "management_reviews",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String, nullable=False),
    Column("period_start", String),
    Column("period_end", String),
    Column("meeting_date", String),
    Column("participants", String),
    Column("summary", String),
    Column("decisions", String),
    Column("actions", String),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)


def init_db():
    metadata.create_all(engine)


def seed_clauses_if_empty():
    with engine.begin() as conn:
        cnt = conn.execute(text("SELECT COUNT(1) FROM clauses")).scalar_one()
        if cnt and int(cnt) > 0:
            return
        seed_file = ROOT_DIR / "data" / "project_files" / "iso55001_clauses_seed.json"
        if not seed_file.exists():
            return
        clauses = json.loads(seed_file.read_text(encoding="utf-8"))
        for row in clauses:
            conn.execute(
                text(
                    "INSERT INTO clauses (clause_id, title, summary) VALUES (:clause_id, :title, :summary) ON CONFLICT(clause_id) DO NOTHING"
                    if DEFAULT_DB_URL.startswith("sqlite")
                    else "INSERT INTO clauses (clause_id, title, summary) VALUES (:clause_id, :title, :summary) ON CONFLICT DO NOTHING"
                ),
                {"clause_id": row["clause_id"], "title": row["title"], "summary": row["summary"]},
            )


# --- Schemas ---
class Clause(BaseModel):
    clause_id: str
    title: str
    summary: str

class StatusEnum(str, Enum):
    Compliant = "Compliant"
    Partial = "Partial"
    Noncompliant = "Noncompliant"
    NotApplicable = "NotApplicable"


class AssessmentCreate(BaseModel):
    clause_id: str = Field(..., description="ISO 55001 clause id, e.g., '4.1'")
    status: StatusEnum = Field(..., description="Compliant | Partial | Noncompliant | NotApplicable")
    evidence: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = Field(None, description="ISO date, e.g., 2025-03-31")


class AssessmentUpdate(BaseModel):
    status: Optional[StatusEnum] = None
    evidence: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = None


class Assessment(BaseModel):
    id: int
    clause_id: str
    status: StatusEnum
    evidence: Optional[str]
    owner: Optional[str]
    due_date: Optional[date]
    created_at: str
    updated_at: str


class Attachment(BaseModel):
    id: int
    assessment_id: int
    filename: str
    content_type: Optional[str] = None
    size: Optional[int] = None
    created_at: str


class AuditStatus(str, Enum):
    Planned = "Planned"
    InProgress = "InProgress"
    Completed = "Completed"
    Cancelled = "Cancelled"


class AuditCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: AuditStatus = AuditStatus.Planned
    scheduled_date: Optional[date] = None


class AuditUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AuditStatus] = None
    scheduled_date: Optional[date] = None
    completed_date: Optional[date] = None


class Audit(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: AuditStatus
    scheduled_date: Optional[date]
    completed_date: Optional[date]
    created_at: str
    updated_at: str


class SeverityEnum(str, Enum):
    Minor = "Minor"
    Major = "Major"
    Critical = "Critical"


class NCStatusEnum(str, Enum):
    Open = "Open"
    InProgress = "InProgress"
    Closed = "Closed"


class NonconformityCreate(BaseModel):
    description: str
    severity: SeverityEnum
    status: NCStatusEnum = NCStatusEnum.Open
    audit_id: Optional[int] = None
    clause_id: Optional[str] = None
    corrective_action: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = None


class NonconformityUpdate(BaseModel):
    description: Optional[str] = None
    severity: Optional[SeverityEnum] = None
    status: Optional[NCStatusEnum] = None
    audit_id: Optional[int] = None
    clause_id: Optional[str] = None
    corrective_action: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = None
    closed_date: Optional[date] = None


class Nonconformity(BaseModel):
    id: int
    description: str
    severity: SeverityEnum
    status: NCStatusEnum
    audit_id: Optional[int]
    clause_id: Optional[str]
    corrective_action: Optional[str]
    owner: Optional[str]
    due_date: Optional[date]
    closed_date: Optional[date]
    created_at: str
    updated_at: str


class ManagementReviewCreate(BaseModel):
    title: str
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    meeting_date: Optional[date] = None
    participants: Optional[str] = None
    summary: Optional[str] = None
    decisions: Optional[str] = None
    actions: Optional[str] = None


class ManagementReviewUpdate(BaseModel):
    title: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    meeting_date: Optional[date] = None
    participants: Optional[str] = None
    summary: Optional[str] = None
    decisions: Optional[str] = None
    actions: Optional[str] = None


class ManagementReview(BaseModel):
    id: int
    title: str
    period_start: Optional[date]
    period_end: Optional[date]
    meeting_date: Optional[date]
    participants: Optional[str]
    summary: Optional[str]
    decisions: Optional[str]
    actions: Optional[str]
    created_at: str
    updated_at: str


# --- App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_clauses_if_empty()
    yield


app = FastAPI(title="ISO 55001 Compliance API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dashboard
STATIC_DIR = ROOT_DIR / "backend" / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/dashboard")
def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not available")
    return FileResponse(str(index))

# --- Request ID + metrics middleware ---
if 'REQUEST_COUNT' not in globals():
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        labelnames=("method", "status"),
    )

# duration histogram (seconds) with per-route label
if 'REQUEST_DURATION' not in globals():
    REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        labelnames=("method", "route", "status"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )


@app.middleware("http")
async def add_request_id_and_collect_metrics(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    logger = get_logger()
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_COUNT.labels(method=request.method, status=str(500)).inc()
        route_label = getattr(request.scope.get("route"), "path", request.url.path)
        REQUEST_DURATION.labels(method=request.method, route=route_label, status="500").observe(duration_ms / 1000.0)
        logger.error(
            __import__("json").dumps(
                {
                    "event": "request_error",
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "status": 500,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": getattr(request.client, "host", None),
                    "user_agent": request.headers.get("user-agent"),
                    "error": str(exc),
                }
            )
        )
        raise
    response.headers["X-Request-ID"] = req_id
    REQUEST_COUNT.labels(method=request.method, status=str(response.status_code)).inc()
    duration_ms = (time.perf_counter() - start) * 1000.0
    route_label = getattr(request.scope.get("route"), "path", request.url.path)
    REQUEST_DURATION.labels(method=request.method, route=route_label, status=str(response.status_code)).observe(duration_ms / 1000.0)
    logger.info(
        __import__("json").dumps(
            {
                "event": "request",
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": getattr(request.client, "host", None),
                "user_agent": request.headers.get("user-agent"),
            }
        )
    )
    return response


# Auth dependency lives in app.auth


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- KPI / Reporting ---
@app.get("/kpi/overview")
def kpi_overview(
    created_from: Optional[date] = None,
    created_to: Optional[date] = None,
    as_of: Optional[date] = None,
):
    today = as_of or datetime.utcnow().date()
    def range_clause(table: str) -> tuple[str, dict]:
        clauses = []
        params: dict = {}
        if created_from:
            clauses.append(f"{table}.created_at >= :created_from")
            params["created_from"] = created_from.isoformat()
        if created_to:
            clauses.append(f"{table}.created_at <= :created_to")
            params["created_to"] = created_to.isoformat()
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    with engine.connect() as conn:
        a_where, a_params = range_clause("audits")
        audits_by_status = conn.execute(text(f"SELECT status, COUNT(*) as c FROM audits{a_where} GROUP BY status"), a_params).all()
        n_where, n_params = range_clause("nonconformities")
        ncs_by_status = conn.execute(text(f"SELECT status, COUNT(*) as c FROM nonconformities{n_where} GROUP BY status"), n_params).all()
        ncs_by_severity = conn.execute(text(f"SELECT severity, COUNT(*) as c FROM nonconformities{n_where} GROUP BY severity"), n_params).all()
        am_where, am_params = range_clause("assessments")
        assessments_by_status = conn.execute(text(f"SELECT status, COUNT(*) as c FROM assessments{am_where} GROUP BY status"), am_params).all()
        overdue = conn.execute(
            text("SELECT COUNT(*) FROM nonconformities WHERE status != 'Closed' AND due_date IS NOT NULL AND due_date < :today"),
            {"today": today.isoformat()},
        ).scalar_one()
        total_audits = conn.execute(text(f"SELECT COUNT(*) FROM audits{a_where}"), a_params).scalar_one()
        total_ncs = conn.execute(text(f"SELECT COUNT(*) FROM nonconformities{n_where}"), n_params).scalar_one()
    return {
        "audits": {
            "total": int(total_audits or 0),
            "by_status": {row[0]: int(row[1]) for row in audits_by_status},
        },
        "nonconformities": {
            "total": int(total_ncs or 0),
            "by_status": {row[0]: int(row[1]) for row in ncs_by_status},
            "by_severity": {row[0]: int(row[1]) for row in ncs_by_severity},
            "overdue": int(overdue or 0),
        },
        "assessments": {
            "by_status": {row[0]: int(row[1]) for row in assessments_by_status},
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/kpi/overdue_nonconformities")
def kpi_overdue_nonconformities(days: int = 0):
    """List open nonconformities overdue by at least `days`."""
    if days < 0:
        raise HTTPException(status_code=400, detail="days must be >= 0")
    today = datetime.utcnow().date()
    cutoff = today if days == 0 else (today - timedelta(days=days))
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, description, severity, status, due_date FROM nonconformities WHERE status != 'Closed' AND due_date IS NOT NULL AND due_date < :cutoff ORDER BY due_date ASC"
            ),
            {"cutoff": cutoff.isoformat()},
        ).mappings().all()
    items = []
    for r in rows:
        dd = r.get("due_date")
        try:
            d = date.fromisoformat(dd) if dd else None
        except Exception:
            d = None
        days_overdue = None
        if d is not None:
            delta = today - d
            if delta.days >= days:
                days_overdue = delta.days
            else:
                continue
        items.append({
            "id": r["id"],
            "description": r["description"],
            "severity": r["severity"],
            "status": r["status"],
            "due_date": dd,
            "days_overdue": days_overdue,
        })
    return {"count": len(items), "items": items}


@app.get("/kpi/nc_trends")
def kpi_nc_trends(from_date: Optional[date] = None, to_date: Optional[date] = None, severity: Optional[SeverityEnum] = None):
    filters_open = []
    params_open: dict = {}
    if from_date:
        filters_open.append("created_at >= :from_date")
        params_open["from_date"] = from_date.isoformat()
    if to_date:
        filters_open.append("created_at <= :to_date")
        params_open["to_date"] = to_date.isoformat()
    if severity:
        filters_open.append("severity = :severity")
        params_open["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    where_open = (" WHERE " + " AND ".join(filters_open)) if filters_open else ""

    filters_closed = []
    params_closed: dict = {}
    if from_date:
        filters_closed.append("closed_date >= :from_date")
        params_closed["from_date"] = from_date.isoformat()
    if to_date:
        filters_closed.append("closed_date <= :to_date")
        params_closed["to_date"] = to_date.isoformat()
    if severity:
        filters_closed.append("severity = :severity")
        params_closed["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    filters_closed.append("closed_date IS NOT NULL")
    where_closed = (" WHERE " + " AND ".join(filters_closed)) if filters_closed else " WHERE closed_date IS NOT NULL"

    with engine.connect() as conn:
        opened = conn.execute(text(f"SELECT substr(created_at,1,7) as ym, COUNT(*) as c FROM nonconformities{where_open} GROUP BY ym ORDER BY ym"), params_open).all()
        closed = conn.execute(text(f"SELECT substr(closed_date,1,7) as ym, COUNT(*) as c FROM nonconformities{where_closed} GROUP BY ym ORDER BY ym"), params_closed).all()
    opened_map = {row[0]: int(row[1]) for row in opened if row[0]}
    closed_map = {row[0]: int(row[1]) for row in closed if row[0]}
    months = sorted(set(opened_map.keys()) | set(closed_map.keys()))
    return {"months": months, "opened": opened_map, "closed": closed_map}


@app.get("/clauses", response_model=List[Clause])
def list_clauses(q: Optional[str] = None, limit: int = 50, offset: int = 0, response: Response = None):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    base = "SELECT clause_id, title, summary FROM clauses"
    params = {}
    if q:
        base += " WHERE clause_id LIKE :q OR title LIKE :q OR summary LIKE :q"
        params["q"] = f"%{q}%"
    base += " ORDER BY clause_id LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})
    with engine.connect() as conn:
        # total
        total = conn.execute(text(
            ("SELECT COUNT(*) FROM clauses WHERE clause_id LIKE :q OR title LIKE :q OR summary LIKE :q") if q else ("SELECT COUNT(*) FROM clauses")
        ), params if q else {}).scalar_one()
        rows = conn.execute(text(base), params).mappings().all()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [Clause(**dict(r)) for r in rows]


@app.get("/clauses/{clause_id}", response_model=Clause)
def get_clause(clause_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT clause_id, title, summary FROM clauses WHERE clause_id = :cid"),
            {"cid": clause_id},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Clause not found")
    return Clause(**dict(row))


@app.post("/assessments", response_model=Assessment, status_code=201)
def create_assessment(payload: AssessmentCreate, _auth=Depends(role_required("editor"))):
    # Ensure clause exists
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM clauses WHERE clause_id = :cid"), {"cid": payload.clause_id}
        ).first()
        if not exists:
            raise HTTPException(status_code=400, detail="Unknown clause_id")
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO assessments (clause_id, status, evidence, owner, due_date, created_at, updated_at)
                VALUES (:clause_id, :status, :evidence, :owner, :due_date, :created_at, :updated_at)
                RETURNING id
                """
                if not DEFAULT_DB_URL.startswith("sqlite")
                else """
                INSERT INTO assessments (clause_id, status, evidence, owner, due_date, created_at, updated_at)
                VALUES (:clause_id, :status, :evidence, :owner, :due_date, :created_at, :updated_at)
                """
            ),
            {
                "clause_id": payload.clause_id,
                "status": payload.status.value if isinstance(payload.status, StatusEnum) else payload.status,
                "evidence": payload.evidence,
                "owner": payload.owner,
                "due_date": payload.due_date.isoformat() if payload.due_date else None,
                "created_at": now,
                "updated_at": now,
            },
        )
        if DEFAULT_DB_URL.startswith("sqlite"):
            new_id = conn.execute(text("SELECT last_insert_rowid()"))
            new_id = list(new_id)[0][0]
        else:
            new_id = res.scalar_one()
        row = conn.execute(text("SELECT * FROM assessments WHERE id = :id"), {"id": new_id}).mappings().first()
    return Assessment(**dict(row))


def _row_to_assessment(row) -> Assessment:
    d = dict(row)
    if d.get("due_date"):
        try:
            d["due_date"] = date.fromisoformat(d["due_date"])  # type: ignore
        except Exception:
            d["due_date"] = None
    return Assessment(**d)


@app.get("/assessments", response_model=List[Assessment])
def list_assessments(
    clause_id: Optional[str] = None,
    status: Optional[StatusEnum] = None,
    limit: int = 50,
    offset: int = 0,
    response: Response = None,
):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    base = "SELECT * FROM assessments"
    filters = []
    params = {}
    if clause_id:
        filters.append("clause_id = :cid")
        params["cid"] = clause_id
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, StatusEnum) else status
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})
    with engine.connect() as conn:
        count_sql = "SELECT COUNT(*) FROM assessments" + (" WHERE " + " AND ".join(filters) if filters else "")
        total = conn.execute(text(count_sql), params).scalar_one()
        rows = conn.execute(text(base), params).mappings().all()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [_row_to_assessment(r) for r in rows]


@app.get("/assessments/{assessment_id}", response_model=Assessment)
def get_assessment(assessment_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM assessments WHERE id = :id"), {"id": assessment_id}
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _row_to_assessment(row)


# --- Evidence attachments ---
def _ensure_evidence_dir():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/assessments/{assessment_id}/attachments", response_model=Attachment, status_code=201)
def upload_attachment(assessment_id: int, file: UploadFile = File(...), _auth=Depends(role_required("editor"))):
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM assessments WHERE id = :id"), {"id": assessment_id}).first()
        if not exists:
            raise HTTPException(status_code=404, detail="Assessment not found")
    _ensure_evidence_dir()
    subdir = EVIDENCE_DIR / f"assessment_{assessment_id}"
    subdir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ""
    uid = uuid.uuid4().hex
    stored_name = f"{uid}{ext}"
    dest_path = subdir / stored_name
    with dest_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    size = dest_path.stat().st_size
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO attachments (assessment_id, filename, content_type, size, stored_path, created_at)
                VALUES (:aid, :filename, :content_type, :size, :stored_path, :created_at)
                RETURNING id
                """
                if not DEFAULT_DB_URL.startswith("sqlite")
                else """
                INSERT INTO attachments (assessment_id, filename, content_type, size, stored_path, created_at)
                VALUES (:aid, :filename, :content_type, :size, :stored_path, :created_at)
                """
            ),
            {
                "aid": assessment_id,
                "filename": file.filename or stored_name,
                "content_type": file.content_type,
                "size": int(size),
                "stored_path": str(dest_path),
                "created_at": now,
            },
        )
        if DEFAULT_DB_URL.startswith("sqlite"):
            att_id = conn.execute(text("SELECT last_insert_rowid()")).scalar_one()
        else:
            att_id = res.scalar_one()
        row = conn.execute(text("SELECT id, assessment_id, filename, content_type, size, created_at FROM attachments WHERE id = :id"), {"id": att_id}).mappings().first()
    return Attachment(**dict(row))


@app.get("/assessments/{assessment_id}/attachments", response_model=List[Attachment])
def list_attachments(assessment_id: int, response: Response = None):
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM assessments WHERE id = :id"), {"id": assessment_id}).first()
        if not exists:
            raise HTTPException(status_code=404, detail="Assessment not found")
        rows = conn.execute(
            text("SELECT id, assessment_id, filename, content_type, size, created_at FROM attachments WHERE assessment_id = :id ORDER BY created_at DESC, id DESC"),
            {"id": assessment_id},
        ).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM attachments WHERE assessment_id = :id"), {"id": assessment_id}).scalar_one()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [Attachment(**dict(r)) for r in rows]


@app.get("/attachments/{attachment_id}/download")
def download_attachment(attachment_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT filename, content_type, stored_path FROM attachments WHERE id = :id"),
            {"id": attachment_id},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Attachment not found")
    path = Path(row["stored_path"]).resolve()
    if not path.exists():
        raise HTTPException(status_code=410, detail="File no longer available")
    return FileResponse(str(path), media_type=row.get("content_type") or "application/octet-stream", filename=row.get("filename") or path.name)


@app.delete("/attachments/{attachment_id}", status_code=204)
def delete_attachment(attachment_id: int, _auth=Depends(role_required("admin"))):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT stored_path FROM attachments WHERE id = :id"), {"id": attachment_id}
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Attachment not found")
        conn.execute(text("DELETE FROM attachments WHERE id = :id"), {"id": attachment_id})
    try:
        Path(row["stored_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    return {}


@app.patch("/assessments/{assessment_id}", response_model=Assessment)
def update_assessment(assessment_id: int, payload: AssessmentUpdate, _auth=Depends(role_required("editor"))):
    updates = {}
    if payload.status is not None:
        updates["status"] = payload.status.value if hasattr(payload.status, "value") else payload.status
    if payload.evidence is not None:
        updates["evidence"] = payload.evidence
    if payload.owner is not None:
        updates["owner"] = payload.owner
    if payload.due_date is not None:
        updates["due_date"] = payload.due_date.isoformat() if payload.due_date else None
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates["id"] = assessment_id
    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys() if k != "id"])
    with engine.begin() as conn:
        res = conn.execute(text(f"UPDATE assessments SET {set_clause} WHERE id = :id"), updates)
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assessment not found")
        row = conn.execute(
            text("SELECT * FROM assessments WHERE id = :id"), {"id": assessment_id}
        ).mappings().first()
    return _row_to_assessment(row)


# --- Audits ---
def _row_to_audit(row) -> Audit:
    d = dict(row)
    for k in ("scheduled_date", "completed_date"):
        if d.get(k):
            try:
                d[k] = date.fromisoformat(d[k])  # type: ignore
            except Exception:
                d[k] = None
    return Audit(**d)


@app.post("/audits", response_model=Audit, status_code=201)
def create_audit(payload: AuditCreate, _auth=Depends(role_required("editor"))):
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO audits (title, description, status, scheduled_date, created_at, updated_at)
                VALUES (:title, :description, :status, :scheduled_date, :created_at, :updated_at)
                RETURNING id
                """
                if not DEFAULT_DB_URL.startswith("sqlite")
                else """
                INSERT INTO audits (title, description, status, scheduled_date, created_at, updated_at)
                VALUES (:title, :description, :status, :scheduled_date, :created_at, :updated_at)
                """
            ),
            {
                "title": payload.title,
                "description": payload.description,
                "status": payload.status.value if isinstance(payload.status, AuditStatus) else payload.status,
                "scheduled_date": payload.scheduled_date.isoformat() if payload.scheduled_date else None,
                "created_at": now,
                "updated_at": now,
            },
        )
        if DEFAULT_DB_URL.startswith("sqlite"):
            new_id = conn.execute(text("SELECT last_insert_rowid()")).scalar_one()
        else:
            new_id = res.scalar_one()
        row = conn.execute(text("SELECT * FROM audits WHERE id = :id"), {"id": new_id}).mappings().first()
    return _row_to_audit(row)


@app.get("/audits", response_model=List[Audit])
def list_audits(status: Optional[AuditStatus] = None, limit: int = 50, offset: int = 0, response: Response = None):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    base = "SELECT * FROM audits"
    params = {}
    if status:
        base += " WHERE status = :status"
        params["status"] = status.value if isinstance(status, AuditStatus) else status
    base += " ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})
    with engine.connect() as conn:
        count_sql = "SELECT COUNT(*) FROM audits" + (" WHERE status = :status" if status else "")
        total = conn.execute(text(count_sql), params).scalar_one()
        rows = conn.execute(text(base), params).mappings().all()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [_row_to_audit(r) for r in rows]


@app.get("/audits/{audit_id}", response_model=Audit)
def get_audit(audit_id: int):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM audits WHERE id = :id"), {"id": audit_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Audit not found")
    return _row_to_audit(row)


@app.patch("/audits/{audit_id}", response_model=Audit)
def update_audit(audit_id: int, payload: AuditUpdate, _auth=Depends(role_required("editor"))):
    updates = {}
    if payload.title is not None:
        updates["title"] = payload.title
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.status is not None:
        updates["status"] = payload.status.value if isinstance(payload.status, AuditStatus) else payload.status
    if payload.scheduled_date is not None:
        updates["scheduled_date"] = payload.scheduled_date.isoformat() if payload.scheduled_date else None
    if payload.completed_date is not None:
        updates["completed_date"] = payload.completed_date.isoformat() if payload.completed_date else None
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates["id"] = audit_id
    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys() if k != "id"])
    with engine.begin() as conn:
        res = conn.execute(text(f"UPDATE audits SET {set_clause} WHERE id = :id"), updates)
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Audit not found")
        row = conn.execute(text("SELECT * FROM audits WHERE id = :id"), {"id": audit_id}).mappings().first()
    return _row_to_audit(row)


# --- Nonconformities ---
def _row_to_nc(row) -> Nonconformity:
    d = dict(row)
    for k in ("due_date", "closed_date"):
        if d.get(k):
            try:
                d[k] = date.fromisoformat(d[k])  # type: ignore
            except Exception:
                d[k] = None
    return Nonconformity(**d)


@app.post("/nonconformities", response_model=Nonconformity, status_code=201)
def create_nonconformity(payload: NonconformityCreate, _auth=Depends(role_required("editor"))):
    # Validate foreign keys (audit_id, clause_id) if provided
    with engine.connect() as conn:
        if payload.audit_id is not None:
            exists = conn.execute(text("SELECT 1 FROM audits WHERE id = :id"), {"id": payload.audit_id}).first()
            if not exists:
                raise HTTPException(status_code=400, detail="Unknown audit_id")
        if payload.clause_id is not None:
            exists = conn.execute(text("SELECT 1 FROM clauses WHERE clause_id = :cid"), {"cid": payload.clause_id}).first()
            if not exists:
                raise HTTPException(status_code=400, detail="Unknown clause_id")
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO nonconformities (audit_id, clause_id, severity, status, description, corrective_action, owner, due_date, created_at, updated_at)
                VALUES (:audit_id, :clause_id, :severity, :status, :description, :corrective_action, :owner, :due_date, :created_at, :updated_at)
                RETURNING id
                """
                if not DEFAULT_DB_URL.startswith("sqlite")
                else """
                INSERT INTO nonconformities (audit_id, clause_id, severity, status, description, corrective_action, owner, due_date, created_at, updated_at)
                VALUES (:audit_id, :clause_id, :severity, :status, :description, :corrective_action, :owner, :due_date, :created_at, :updated_at)
                """
            ),
            {
                "audit_id": payload.audit_id,
                "clause_id": payload.clause_id,
                "severity": payload.severity.value if isinstance(payload.severity, SeverityEnum) else payload.severity,
                "status": payload.status.value if isinstance(payload.status, NCStatusEnum) else payload.status,
                "description": payload.description,
                "corrective_action": payload.corrective_action,
                "owner": payload.owner,
                "due_date": payload.due_date.isoformat() if payload.due_date else None,
                "created_at": now,
                "updated_at": now,
            },
        )
        if DEFAULT_DB_URL.startswith("sqlite"):
            new_id = conn.execute(text("SELECT last_insert_rowid()")).scalar_one()
        else:
            new_id = res.scalar_one()
        row = conn.execute(text("SELECT * FROM nonconformities WHERE id = :id"), {"id": new_id}).mappings().first()
    return _row_to_nc(row)


@app.get("/nonconformities", response_model=List[Nonconformity])
def list_nonconformities(
    status: Optional[NCStatusEnum] = None,
    severity: Optional[SeverityEnum] = None,
    clause_id: Optional[str] = None,
    audit_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    response: Response = None,
):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    base = "SELECT * FROM nonconformities"
    filters = []
    params = {}
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, NCStatusEnum) else status
    if severity:
        filters.append("severity = :severity")
        params["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    if clause_id:
        filters.append("clause_id = :cid")
        params["cid"] = clause_id
    if audit_id is not None:
        filters.append("audit_id = :aid")
        params["aid"] = audit_id
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})
    with engine.connect() as conn:
        count_sql = "SELECT COUNT(*) FROM nonconformities" + (" WHERE " + " AND ".join(filters) if filters else "")
        total = conn.execute(text(count_sql), params).scalar_one()
        rows = conn.execute(text(base), params).mappings().all()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [_row_to_nc(r) for r in rows]


@app.get("/nonconformities/{nc_id}", response_model=Nonconformity)
def get_nonconformity(nc_id: int):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM nonconformities WHERE id = :id"), {"id": nc_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Nonconformity not found")
    return _row_to_nc(row)


@app.patch("/nonconformities/{nc_id}", response_model=Nonconformity)
def update_nonconformity(nc_id: int, payload: NonconformityUpdate, _auth=Depends(role_required("editor"))):
    updates = {}
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.severity is not None:
        updates["severity"] = payload.severity.value if isinstance(payload.severity, SeverityEnum) else payload.severity
    if payload.status is not None:
        updates["status"] = payload.status.value if isinstance(payload.status, NCStatusEnum) else payload.status
    if payload.audit_id is not None:
        # Validate audit exists
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM audits WHERE id = :id"), {"id": payload.audit_id}).first()
            if not exists:
                raise HTTPException(status_code=400, detail="Unknown audit_id")
        updates["audit_id"] = payload.audit_id
    if payload.clause_id is not None:
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM clauses WHERE clause_id = :cid"), {"cid": payload.clause_id}).first()
            if not exists:
                raise HTTPException(status_code=400, detail="Unknown clause_id")
        updates["clause_id"] = payload.clause_id
    if payload.corrective_action is not None:
        updates["corrective_action"] = payload.corrective_action
    if payload.owner is not None:
        updates["owner"] = payload.owner
    if payload.due_date is not None:
        updates["due_date"] = payload.due_date.isoformat() if payload.due_date else None
    if payload.closed_date is not None:
        updates["closed_date"] = payload.closed_date.isoformat() if payload.closed_date else None
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates["id"] = nc_id
    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys() if k != "id"])
    with engine.begin() as conn:
        res = conn.execute(text(f"UPDATE nonconformities SET {set_clause} WHERE id = :id"), updates)
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Nonconformity not found")
        row = conn.execute(text("SELECT * FROM nonconformities WHERE id = :id"), {"id": nc_id}).mappings().first()
    return _row_to_nc(row)


# --- Management Reviews ---
def _row_to_mr(row) -> ManagementReview:
    d = dict(row)
    for k in ("period_start", "period_end", "meeting_date"):
        if d.get(k):
            try:
                d[k] = date.fromisoformat(d[k])  # type: ignore
            except Exception:
                d[k] = None
    return ManagementReview(**d)


@app.post("/management-reviews", response_model=ManagementReview, status_code=201)
def create_management_review(payload: ManagementReviewCreate, _auth=Depends(role_required("editor"))):
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO management_reviews (title, period_start, period_end, meeting_date, participants, summary, decisions, actions, created_at, updated_at)
                VALUES (:title, :period_start, :period_end, :meeting_date, :participants, :summary, :decisions, :actions, :created_at, :updated_at)
                RETURNING id
                """
                if not DEFAULT_DB_URL.startswith("sqlite")
                else """
                INSERT INTO management_reviews (title, period_start, period_end, meeting_date, participants, summary, decisions, actions, created_at, updated_at)
                VALUES (:title, :period_start, :period_end, :meeting_date, :participants, :summary, :decisions, :actions, :created_at, :updated_at)
                """
            ),
            {
                "title": payload.title,
                "period_start": payload.period_start.isoformat() if payload.period_start else None,
                "period_end": payload.period_end.isoformat() if payload.period_end else None,
                "meeting_date": payload.meeting_date.isoformat() if payload.meeting_date else None,
                "participants": payload.participants,
                "summary": payload.summary,
                "decisions": payload.decisions,
                "actions": payload.actions,
                "created_at": now,
                "updated_at": now,
            },
        )
        if DEFAULT_DB_URL.startswith("sqlite"):
            new_id = conn.execute(text("SELECT last_insert_rowid()")).scalar_one()
        else:
            new_id = res.scalar_one()
        row = conn.execute(text("SELECT * FROM management_reviews WHERE id = :id"), {"id": new_id}).mappings().first()
    return _row_to_mr(row)


@app.get("/management-reviews", response_model=List[ManagementReview])
def list_management_reviews(limit: int = 50, offset: int = 0, response: Response = None):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    base = "SELECT * FROM management_reviews ORDER BY meeting_date DESC NULLS LAST, id DESC LIMIT :limit OFFSET :offset" if not DEFAULT_DB_URL.startswith("sqlite") else "SELECT * FROM management_reviews ORDER BY meeting_date DESC, id DESC LIMIT :limit OFFSET :offset"
    with engine.connect() as conn:
        rows = conn.execute(text(base), {"limit": limit, "offset": offset}).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM management_reviews")).scalar_one()
    if response is not None:
        response.headers["X-Total-Count"] = str(int(total or 0))
    return [_row_to_mr(r) for r in rows]


@app.get("/management-reviews/{mr_id}", response_model=ManagementReview)
def get_management_review(mr_id: int):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM management_reviews WHERE id = :id"), {"id": mr_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Management review not found")
    return _row_to_mr(row)


@app.patch("/management-reviews/{mr_id}", response_model=ManagementReview)
def update_management_review(mr_id: int, payload: ManagementReviewUpdate, _auth=Depends(role_required("editor"))):
    updates = {}
    if payload.title is not None:
        updates["title"] = payload.title
    if payload.period_start is not None:
        updates["period_start"] = payload.period_start.isoformat() if payload.period_start else None
    if payload.period_end is not None:
        updates["period_end"] = payload.period_end.isoformat() if payload.period_end else None
    if payload.meeting_date is not None:
        updates["meeting_date"] = payload.meeting_date.isoformat() if payload.meeting_date else None
    if payload.participants is not None:
        updates["participants"] = payload.participants
    if payload.summary is not None:
        updates["summary"] = payload.summary
    if payload.decisions is not None:
        updates["decisions"] = payload.decisions
    if payload.actions is not None:
        updates["actions"] = payload.actions
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates["id"] = mr_id
    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys() if k != "id"])
    with engine.begin() as conn:
        res = conn.execute(text(f"UPDATE management_reviews SET {set_clause} WHERE id = :id"), updates)
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Management review not found")
        row = conn.execute(text("SELECT * FROM management_reviews WHERE id = :id"), {"id": mr_id}).mappings().first()
    return _row_to_mr(row)


# --- Exports ---
def _rows_to_csv(rows, headers):
    sio = io.StringIO()
    writer = csv.writer(sio)
    writer.writerow(headers)
    for r in rows:
        writer.writerow([r.get(h) for h in headers])
    return sio.getvalue()


@app.get("/export/audits.csv")
def export_audits_csv(status: Optional[AuditStatus] = None, scheduled_from: Optional[date] = None, scheduled_to: Optional[date] = None, completed_from: Optional[date] = None, completed_to: Optional[date] = None):
    base = "SELECT id, title, description, status, scheduled_date, completed_date, created_at, updated_at FROM audits"
    filters = []
    params: dict = {}
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, AuditStatus) else status
    if scheduled_from:
        filters.append("scheduled_date >= :sfrom")
        params["sfrom"] = scheduled_from.isoformat()
    if scheduled_to:
        filters.append("scheduled_date <= :sto")
        params["sto"] = scheduled_to.isoformat()
    if completed_from:
        filters.append("completed_date >= :cfrom")
        params["cfrom"] = completed_from.isoformat()
    if completed_to:
        filters.append("completed_date <= :cto")
        params["cto"] = completed_to.isoformat()
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
        data = [dict(r) for r in rows]
    csv_text = _rows_to_csv(data, ["id", "title", "description", "status", "scheduled_date", "completed_date", "created_at", "updated_at"])
    return Response(content=csv_text, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audits.csv"})


@app.get("/export/nonconformities.csv")
def export_ncs_csv(status: Optional[NCStatusEnum] = None, severity: Optional[SeverityEnum] = None, clause_id: Optional[str] = None, audit_id: Optional[int] = None, created_from: Optional[date] = None, created_to: Optional[date] = None, due_from: Optional[date] = None, due_to: Optional[date] = None):
    base = "SELECT id, audit_id, clause_id, severity, status, description, corrective_action, owner, due_date, closed_date, created_at, updated_at FROM nonconformities"
    filters = []
    params: dict = {}
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, NCStatusEnum) else status
    if severity:
        filters.append("severity = :severity")
        params["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    if clause_id:
        filters.append("clause_id = :cid")
        params["cid"] = clause_id
    if audit_id is not None:
        filters.append("audit_id = :aid")
        params["aid"] = audit_id
    if created_from:
        filters.append("created_at >= :cfrom")
        params["cfrom"] = created_from.isoformat()
    if created_to:
        filters.append("created_at <= :cto")
        params["cto"] = created_to.isoformat()
    if due_from:
        filters.append("due_date >= :dfrom")
        params["dfrom"] = due_from.isoformat()
    if due_to:
        filters.append("due_date <= :dto")
        params["dto"] = due_to.isoformat()
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
        data = [dict(r) for r in rows]
    csv_text = _rows_to_csv(data, ["id", "audit_id", "clause_id", "severity", "status", "description", "corrective_action", "owner", "due_date", "closed_date", "created_at", "updated_at"])
    return Response(content=csv_text, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=nonconformities.csv"})


@app.get("/export/audits.xlsx")
def export_audits_xlsx(status: Optional[AuditStatus] = None):
    from openpyxl import Workbook

    base = "SELECT id, title, description, status, scheduled_date, completed_date, created_at, updated_at FROM audits"
    params: dict = {}
    if status:
        base += " WHERE status = :status"
        params["status"] = status.value if isinstance(status, AuditStatus) else status
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
    wb = Workbook()
    ws = wb.active
    ws.title = "audits"
    headers = ["id", "title", "description", "status", "scheduled_date", "completed_date", "created_at", "updated_at"]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return Response(content=bio.read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=audits.xlsx"})


@app.get("/export/nonconformities.xlsx")
def export_ncs_xlsx(status: Optional[NCStatusEnum] = None, severity: Optional[SeverityEnum] = None):
    from openpyxl import Workbook

    base = "SELECT id, audit_id, clause_id, severity, status, description, corrective_action, owner, due_date, closed_date, created_at, updated_at FROM nonconformities"
    filters = []
    params: dict = {}
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, NCStatusEnum) else status
    if severity:
        filters.append("severity = :severity")
        params["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
    wb = Workbook()
    ws = wb.active
    ws.title = "nonconformities"
    headers = ["id", "audit_id", "clause_id", "severity", "status", "description", "corrective_action", "owner", "due_date", "closed_date", "created_at", "updated_at"]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return Response(content=bio.read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=nonconformities.xlsx"})


@app.get("/export/assessments.csv")
def export_assessments_csv(clause_id: Optional[str] = None, status: Optional[StatusEnum] = None, owner: Optional[str] = None, created_from: Optional[date] = None, created_to: Optional[date] = None, due_from: Optional[date] = None, due_to: Optional[date] = None):
    base = "SELECT id, clause_id, status, evidence, owner, due_date, created_at, updated_at FROM assessments"
    filters = []
    params: dict = {}
    if clause_id:
        filters.append("clause_id = :cid")
        params["cid"] = clause_id
    if status:
        filters.append("status = :status")
        params["status"] = status.value if isinstance(status, StatusEnum) else status
    if owner:
        filters.append("owner = :owner")
        params["owner"] = owner
    if created_from:
        filters.append("created_at >= :cfrom")
        params["cfrom"] = created_from.isoformat()
    if created_to:
        filters.append("created_at <= :cto")
        params["cto"] = created_to.isoformat()
    if due_from:
        filters.append("due_date >= :dfrom")
        params["dfrom"] = due_from.isoformat()
    if due_to:
        filters.append("due_date <= :dto")
        params["dto"] = due_to.isoformat()
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
    csv_text = _rows_to_csv(rows, ["id", "clause_id", "status", "evidence", "owner", "due_date", "created_at", "updated_at"])
    return Response(content=csv_text, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=assessments.csv"})


@app.get("/export/assessments.xlsx")
def export_assessments_xlsx(status: Optional[StatusEnum] = None):
    from openpyxl import Workbook
    base = "SELECT id, clause_id, status, evidence, owner, due_date, created_at, updated_at FROM assessments"
    params: dict = {}
    if status:
        base += " WHERE status = :status"
        params["status"] = status.value if isinstance(status, StatusEnum) else status
    base += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
    wb = Workbook()
    ws = wb.active
    ws.title = "assessments"
    headers = ["id", "clause_id", "status", "evidence", "owner", "due_date", "created_at", "updated_at"]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return Response(content=bio.read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=assessments.xlsx"})


@app.get("/export/audit_nonconformities.csv")
def export_audit_nc_csv(status: Optional[NCStatusEnum] = None, severity: Optional[SeverityEnum] = None, audit_id: Optional[int] = None):
    base = """
    SELECT a.id as audit_id, a.title as audit_title, n.id as nc_id, n.severity, n.status, n.description, n.owner, n.due_date, n.closed_date, n.created_at
    FROM audits a LEFT JOIN nonconformities n ON n.audit_id = a.id
    """
    filters = []
    params: dict = {}
    if audit_id is not None:
        filters.append("a.id = :aid")
        params["aid"] = audit_id
    if status:
        filters.append("n.status = :status")
        params["status"] = status.value if isinstance(status, NCStatusEnum) else status
    if severity:
        filters.append("n.severity = :severity")
        params["severity"] = severity.value if isinstance(severity, SeverityEnum) else severity
    if filters:
        base += " WHERE " + " AND ".join(filters)
    base += " ORDER BY a.id, n.id"
    with engine.connect() as conn:
        rows = conn.execute(text(base), params).mappings().all()
    headers = ["audit_id", "audit_title", "nc_id", "severity", "status", "description", "owner", "due_date", "closed_date", "created_at"]
    csv_text = _rows_to_csv(rows, headers)
    return Response(content=csv_text, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit_nonconformities.csv"})


@app.get("/export/audit_nonconformities.xlsx")
def export_audit_nc_xlsx():
    from openpyxl import Workbook
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT a.id as audit_id, a.title as audit_title, n.id as nc_id, n.severity, n.status, n.description, n.owner, n.due_date, n.closed_date, n.created_at FROM audits a LEFT JOIN nonconformities n ON n.audit_id = a.id ORDER BY a.id, n.id"
        )).mappings().all()
    headers = ["audit_id", "audit_title", "nc_id", "severity", "status", "description", "owner", "due_date", "closed_date", "created_at"]
    wb = Workbook()
    ws = wb.active
    ws.title = "audit_ncs"
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return Response(content=bio.read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=audit_nonconformities.xlsx"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
