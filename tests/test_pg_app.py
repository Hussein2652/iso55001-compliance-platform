import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").startswith("postgresql"),
    reason="Postgres DATABASE_URL not set; skipping integration test",
)


def test_pg_app_happy_path():
    # Ensure DATABASE_URL points to Postgres (provided by CI job) and no API_TOKEN to allow dev writes
    os.environ.pop("API_TOKEN", None)
    # Import the app after env is set
    from backend.app.main import app

    with TestClient(app) as client:
        # Run setup to migrate and seed
        r = client.post("/setup", json={"run_migrations": True, "reseed_clauses": True, "verify_object_store": False})
        assert r.status_code in (200, 201)

        # Clauses present
        rc = client.get("/clauses?limit=5&offset=0")
        assert rc.status_code == 200
        clauses = rc.json()
        assert isinstance(clauses, list) and len(clauses) > 0

        # Create assessment (dev mode allows writes without token)
        ra = client.post("/assessments", json={"clause_id": "4.1", "status": "Compliant"})
        assert ra.status_code == 201, ra.text
        aid = ra.json()["id"]

        # Envelope list
        rv1 = client.get("/v1/assessments?limit=10&offset=0")
        assert rv1.status_code == 200
        body = rv1.json()
        assert "items" in body and body["total"] >= 1

        # Fetch detail
        rd = client.get(f"/assessments/{aid}")
        assert rd.status_code == 200
        assert rd.json()["status"] == "Compliant"

