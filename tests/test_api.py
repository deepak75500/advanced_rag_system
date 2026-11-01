import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from localpdf.api import create_app  # noqa: E402


def test_workspace_lifecycle_and_validation(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    ws = c.post("/workspaces").json()["id"]
    assert c.get(f"/workspaces/{ws}/documents").json() == []
    r = c.post(f"/workspaces/{ws}/documents", files={"file": ("evil.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert c.get("/workspaces/..%2Fetc/documents").status_code in (400, 404)
    assert c.get("/system").status_code == 200
    assert c.delete(f"/workspaces/{ws}").status_code == 200
    assert c.get(f"/workspaces/{ws}/documents").status_code == 404
