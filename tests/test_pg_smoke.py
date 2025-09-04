import os
import subprocess
from pathlib import Path

def test_alembic_upgrade_smoke():
    # Skip if DATABASE_URL not pointing to postgres
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url.startswith("postgresql"):
        return
    # Run alembic upgrade head
    subprocess.run([
        "alembic", "-c", str(Path("backend/alembic.ini")), "upgrade", "head"
    ], check=True)

