import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Use a temp SQLite DB per test module; set before import
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        # For write-protection behavior tests, we toggle API_TOKEN per test
        os.environ.pop("API_TOKEN", None)
        # Ensure repo root is importable
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from backend.app.main import app  # import after env is set

        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    # Request ID should be present
    assert r.headers.get("x-request-id")

    # If provided, server should echo same ID
    r2 = client.get("/health", headers={"X-Request-ID": "fixed-id-123"})
    assert r2.headers.get("x-request-id") == "fixed-id-123"


def test_clauses_seeded_and_search(client):
    r = client.get("/clauses")
    assert r.status_code == 200
    clauses = r.json()
    assert isinstance(clauses, list)
    assert len(clauses) > 0
    # Search should return subset
    r2 = client.get("/clauses", params={"q": "policy"})
    assert r2.status_code == 200
    assert all(
        ("policy" in (c["title"] + c["summary"]).lower()) or ("policy" in c["clause_id"].lower())
        for c in r2.json()
    )


def test_assessment_create_and_get_without_token(client):
    # No API_TOKEN set -> dev mode allows writes
    payload = {
        "clause_id": "4.1",
        "status": "Partial",
        "evidence": "Initial review",
        "owner": "QA",
        "due_date": "2025-12-31",
    }
    r = client.post("/assessments", json=payload)
    assert r.status_code == 201, r.text
    created = r.json()
    aid = created["id"]
    r2 = client.get(f"/assessments/{aid}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "Partial"


def test_validation_errors(client):
    # Invalid status -> 422 from Pydantic Enum
    r = client.post(
        "/assessments",
        json={"clause_id": "4.1", "status": "BadStatus"},
    )
    assert r.status_code == 422
    # Invalid date -> 422
    r2 = client.post(
        "/assessments",
        json={"clause_id": "4.1", "status": "Compliant", "due_date": "not-a-date"},
    )
    assert r2.status_code == 422


def test_assessment_auth_enforced_when_token_set():
    # New app instance with token set; new DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test2.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["API_TOKEN"] = "test-token"
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main

        reload(main)
        from backend.app.main import app
        with TestClient(app) as c:
            # Unauthenticated write should fail
            r = c.post(
                "/assessments",
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r.status_code in (401, 403)

            # Authenticated write should pass
            r2 = c.post(
                "/assessments",
                headers={"Authorization": "Bearer test-token"},
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r2.status_code == 201, r2.text
            aid = r2.json()["id"]
            # Update with auth
            r3 = c.patch(
                f"/assessments/{aid}",
                headers={"Authorization": "Bearer test-token"},
                json={"status": "Partial"},
            )
            assert r3.status_code == 200
            assert r3.json()["status"] == "Partial"

    # Static token with viewer role must be rejected for writes
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test2b.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["API_TOKEN"] = "tok"
        os.environ["API_ROLE"] = "viewer"
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main2
        reload(main2)
        from backend.app.main import app as app2
        with TestClient(app2) as c2:
            r = c2.post(
                "/assessments",
                headers={"Authorization": "Bearer tok"},
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r.status_code == 403
            assert "Insufficient" in r.text


def test_jwt_auth_hs256():
    # Configure HS256 secret; JWT required for writes
    import jwt as pyjwt
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tesths.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["OIDC_HS256_SECRET"] = "super-secret"
        # Optional: issuer/audience
        os.environ["OIDC_ISSUER"] = "https://issuer.example"
        os.environ["OIDC_AUDIENCE"] = "iso55001-api"
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main
        reload(main)
        from backend.app.main import app
        with TestClient(app) as c:
            # Create a short-lived token
            token = pyjwt.encode({
                "iss": "https://issuer.example",
                "aud": "iso55001-api",
                "sub": "tester",
                "roles": ["editor"],
            }, "super-secret", algorithm="HS256")

            # Unauth write fails
            r_fail = c.post(
                "/assessments",
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r_fail.status_code in (401, 403)

            # Authenticated write passes
            r_ok = c.post(
                "/assessments",
                headers={"Authorization": f"Bearer {token}"},
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r_ok.status_code == 201, r_ok.text

    # Metrics endpoint available and includes histogram and route labels
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "testm.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from importlib import reload
        import backend.app.main as mainm
        reload(mainm)
        from backend.app.main import app as appm
        with TestClient(appm) as c:
            # Trigger a request to /health so metrics include a route label
            _ = c.get("/health")
            mr = c.get("/metrics")
            assert mr.status_code == 200
            assert b"http_requests_total" in mr.content
            assert b"http_request_duration_seconds" in mr.content
            assert b'route="/health"' in mr.content


def test_attachments_upload_download_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test3.db"
        evidence_dir = Path(tmpdir) / "evidence"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["API_TOKEN"] = "tok"
        os.environ["API_ROLE"] = "admin"
        os.environ["EVIDENCE_DIR"] = str(evidence_dir)
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main
        reload(main)
        from backend.app.main import app
        with TestClient(app) as c:
            # Create assessment
            r = c.post(
                "/assessments",
                headers={"Authorization": "Bearer tok"},
                json={"clause_id": "4.1", "status": "Compliant"},
            )
            assert r.status_code == 201, r.text
            aid = r.json()["id"]

            # Upload a file
            files = {"file": ("note.txt", b"hello evidence", "text/plain")}
            ur = c.post(
                f"/assessments/{aid}/attachments",
                headers={"Authorization": "Bearer tok"},
                files=files,
            )
            assert ur.status_code == 201, ur.text
            att = ur.json()

            # List attachments
            lr = c.get(f"/assessments/{aid}/attachments")
            assert lr.status_code == 200
            assert len(lr.json()) == 1

            # Download
            dr = c.get(f"/attachments/{att['id']}/download")
            assert dr.status_code == 200
            assert dr.content == b"hello evidence"

            # Delete
            delr = c.delete(
                f"/attachments/{att['id']}", headers={"Authorization": "Bearer tok"}
            )
            assert delr.status_code == 204
            # Now list should be empty
            lr2 = c.get(f"/assessments/{aid}/attachments")
            assert lr2.status_code == 200
            assert lr2.json() == []


def test_audits_and_nonconformities_flow():
    # Editor token
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "audit.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["API_TOKEN"] = "tok"
        os.environ["API_ROLE"] = "editor"
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main
        reload(main)
        from backend.app.main import app
        with TestClient(app) as c:
            # Create an audit
            ar = c.post(
                "/audits",
                headers={"Authorization": "Bearer tok"},
                json={"title": "Internal audit Q1", "scheduled_date": "2025-01-15"},
            )
            assert ar.status_code == 201, ar.text
            audit = ar.json()
            aid = audit["id"]

            # Update audit status
            ur = c.patch(
                f"/audits/{aid}",
                headers={"Authorization": "Bearer tok"},
                json={"status": "InProgress"},
            )
            assert ur.status_code == 200
            assert ur.json()["status"] == "InProgress"

            # Create a nonconformity tied to audit and clause 4.1
            ncr = c.post(
                "/nonconformities",
                headers={"Authorization": "Bearer tok"},
                json={
                    "description": "Policy not documented",
                    "severity": "Major",
                    "audit_id": aid,
                    "clause_id": "4.1",
                    "owner": "QA",
                    "due_date": "2025-02-28",
                },
            )
            assert ncr.status_code == 201, ncr.text
            nc = ncr.json()
            ncid = nc["id"]
            assert nc["status"] == "Open"

            # List and filter
            lr = c.get("/nonconformities", params={"audit_id": aid, "severity": "Major"})
            assert lr.status_code == 200
            assert any(item["id"] == ncid for item in lr.json())

            # Close the nonconformity
            cr = c.patch(
                f"/nonconformities/{ncid}",
                headers={"Authorization": "Bearer tok"},
                json={"status": "Closed", "closed_date": "2025-03-01"},
            )
            assert cr.status_code == 200
            assert cr.json()["status"] == "Closed"

            # View-only token should not be able to update
            os.environ["API_ROLE"] = "viewer"
            # Reload module to capture env change in auth
            reload(main)
            from backend.app.main import app as app2
            with TestClient(app2) as c2:
                rr = c2.patch(
                    f"/audits/{aid}",
                    headers={"Authorization": "Bearer tok"},
                    json={"status": "Completed"},
                )
            assert rr.status_code == 403


def test_kpi_and_export_and_management_review():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kpi.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["API_TOKEN"] = "tok"
        os.environ["API_ROLE"] = "editor"
        from importlib import reload
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        import backend.app.main as main
        reload(main)
        from backend.app.main import app
        with TestClient(app) as c:
            # Create some data
            c.post(
                "/audits",
                headers={"Authorization": "Bearer tok"},
                json={"title": "Audit A", "status": "Planned"},
            )
            ncr = c.post(
                "/nonconformities",
                headers={"Authorization": "Bearer tok"},
                json={
                    "description": "Old NC",
                    "severity": "Minor",
                    "status": "Open",
                    "due_date": "2000-01-01",
                },
            )
            assert ncr.status_code == 201

            # KPI overview
            kpi = c.get("/kpi/overview")
            assert kpi.status_code == 200
            body = kpi.json()
            assert "audits" in body and "nonconformities" in body
            assert body["nonconformities"]["overdue"] >= 1

            # Export CSV (filtered)
            ec = c.get("/export/nonconformities.csv?severity=Minor")
            assert ec.status_code == 200
            assert b"id,audit_id,clause_id,severity,status,description" in ec.content

            # Export XLSX
            ex = c.get("/export/audits.xlsx")
            assert ex.status_code == 200
            # XLSX starts with PK (zip signature)
            assert ex.content[:2] == b"PK"

            # KPI as_of in past should show zero overdue
            kpi_past = c.get("/kpi/overview", params={"as_of": "1999-01-01"})
            assert kpi_past.status_code == 200
            assert kpi_past.json()["nonconformities"]["overdue"] == 0

            # Dashboard served
            d = c.get("/dashboard")
            assert d.status_code == 200
            assert b"ISO 55001 Dashboard" in d.content

            # Management review create/list/update
            mr = c.post(
                "/management-reviews",
                headers={"Authorization": "Bearer tok"},
                json={
                    "title": "Q1 Review",
                    "period_start": "2025-01-01",
                    "period_end": "2025-03-31",
                    "meeting_date": "2025-04-10",
                    "participants": "CEO, COO",
                    "summary": "Good progress",
                },
            )
            assert mr.status_code == 201
            mrid = mr.json()["id"]
            ml = c.get("/management-reviews")
            assert ml.status_code == 200
            mu = c.patch(
                f"/management-reviews/{mrid}",
                headers={"Authorization": "Bearer tok"},
                json={"decisions": "Allocate budget"},
            )
            assert mu.status_code == 200
