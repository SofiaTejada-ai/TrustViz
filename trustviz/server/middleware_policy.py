# middleware_policy.py
import re
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

DENY = re.compile(r"(bypass\s+mfa|real\s+bank\s+site|ransomware|keylogger|shellcode|c2\s*server|payload|cve-\d{4}-\d+)", re.I)

class PolicyGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = (await request.body()).decode(errors="ignore")
        query = request.url.query or ""
        text = f"{body} {query}"
        # mark risky; do not block
        request.state.risky = bool(DENY.search(text))
        return await call_next(request)
