from typing import Optional
from fastapi import HTTPException, Header, Query, Depends, Request


def verify_token(
    request: Request,
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    settings = request.app.state.settings
    supplied = token
    if not supplied and authorization:
        parts = authorization.split()
        if len(parts) >= 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
    if settings.SECRET_TOKEN and supplied != settings.SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True
