"""Small security switch for public, resource-constrained research demos.

This is an operational safeguard, not a substitute for authentication, audit,
encryption, persistence, or bank-grade authorization.
"""
from __future__ import annotations
import os
from fastapi import HTTPException

def is_public_demo() -> bool:
    return os.getenv("PUBLIC_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

def require_local_write_mode() -> None:
    if is_public_demo():
        raise HTTPException(
            status_code=403,
            detail="Action désactivée sur la démonstration publique. Utiliser un environnement local protégé.",
        )
