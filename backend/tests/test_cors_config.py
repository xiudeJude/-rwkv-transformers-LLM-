import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import app
from app.core.config import Settings


def test_cors_default_allowed_origins():
    print("[1] Testing CORS with default allowed origins...", flush=True)
    client = TestClient(app)

    # 1. Allowed origin: localhost:5173
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173", (
        f"Unexpected allow origin: {resp.headers.get('access-control-allow-origin')}"
    )
    assert resp.headers.get("access-control-allow-credentials") == "true", (
        f"Unexpected allow credentials: {resp.headers.get('access-control-allow-credentials')}"
    )
    print("  -> Origin http://localhost:5173 permitted with credentials", flush=True)

    # 2. Allowed origin: 127.0.0.1:5173
    resp = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    print("  -> Origin http://127.0.0.1:5173 permitted", flush=True)

    # 3. Preflight OPTIONS request from allowed origin
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    print("  -> Preflight OPTIONS allowed for http://localhost:5173", flush=True)


def test_cors_disallowed_origin():
    print("[2] Testing CORS with unauthorized origin...", flush=True)
    client = TestClient(app)

    # Disallowed origin should NOT have Access-Control-Allow-Origin header
    resp = client.get("/health", headers={"Origin": "http://malicious-attacker.com"})
    assert "access-control-allow-origin" not in resp.headers, (
        f"Security issue: unauthorized origin was allowed! Header: {resp.headers.get('access-control-allow-origin')}"
    )
    print("  -> Unauthorized origin http://malicious-attacker.com properly blocked (no CORS headers)", flush=True)


def test_cors_settings_custom_env():
    print("[3] Testing CORS_ORIGINS environment variable parsing and middleware enforcement...", flush=True)
    orig_env = os.environ.get("CORS_ORIGINS")
    try:
        os.environ["CORS_ORIGINS"] = "https://custom.visualizer.internal, http://app.local:8080"
        custom_settings = Settings()
        origins = custom_settings.CORS_ORIGINS
        assert origins == ["https://custom.visualizer.internal", "http://app.local:8080"], f"Got: {origins}"
        print(f"  -> Successfully parsed custom origins: {origins}", flush=True)

        # Create a dynamic test app configured with these custom origins
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=custom_settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @test_app.get("/test")
        def dummy():
            return {"ok": True}

        test_client = TestClient(test_app)

        # Custom allowed origin
        resp_allowed = test_client.get("/test", headers={"Origin": "https://custom.visualizer.internal"})
        assert resp_allowed.status_code == 200
        assert resp_allowed.headers.get("access-control-allow-origin") == "https://custom.visualizer.internal"

        # Disallowed origin (even former default localhost:5173 is blocked when customized)
        resp_blocked = test_client.get("/test", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in resp_blocked.headers
        print("  -> Custom middleware correctly allows configured origins and rejects others", flush=True)

    finally:
        if orig_env is None:
            os.environ.pop("CORS_ORIGINS", None)
        else:
            os.environ["CORS_ORIGINS"] = orig_env


if __name__ == "__main__":
    test_cors_default_allowed_origins()
    test_cors_disallowed_origin()
    test_cors_settings_custom_env()
    print("\nALL CORS TESTS PASSED!")
