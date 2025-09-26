# trustviz/server/middleware_policy.py
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import Request

DENY = re.compile(
    r"(ransomware|keylogger|mimikatz|shellcode|c2\s*server|"
    r"privilege\s*escalation|payload|cve-\d{4}-\d+)", re.I
)

class PolicyGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = (await request.body()).decode(errors="ignore")
        if DENY.search(body):
            return JSONResponse(
                {"error": "Blocked: harmful or dual-use request. We can cover defensive topics instead."},
                status_code=400
            )
        return await call_next(request)
