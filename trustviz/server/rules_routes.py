# trustviz/server/rules_routes.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse


from trustviz.rules.core import SequenceRule, Step, lint_rule
from trustviz.rules.compile import to_sigma, to_kql
from trustviz.lab.sim import simulate_logs
from trustviz.lab.grade import grade

router = APIRouter(prefix="/rules", tags=["rules"])

def _parse_rule(rule_json: dict) -> SequenceRule:
    try:
        return SequenceRule(
            name=rule_json["name"],
            window_s=int(rule_json["window_s"]),
            on=list(rule_json["on"]),
            steps=[
                Step(id=s["id"], event=s["event"], where=s.get("where", {}), cite=s.get("cite", []))
                for s in rule_json["steps"]
            ],
            tags=rule_json.get("tags", []),
        )
    except Exception as e:
        raise HTTPException(400, f"bad rule json: {e}")

@router.post("/compile/sigma")
def compile_sigma(rule_json: dict = Body(...)):
    rule = _parse_rule(rule_json)
    errs = lint_rule(rule)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=400)
    return {"ok": True, "sigma": to_sigma(rule)}

@router.post("/compile/kql")
def compile_kql(rule_json: dict = Body(...)):
    rule = _parse_rule(rule_json)
    errs = lint_rule(rule)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=400)
    return {"ok": True, "kql": to_kql(rule)}

@router.post("/lab/simulate")
def lab_simulate(rule_json: dict = Body(...)):
    rule = _parse_rule(rule_json)
    errs = lint_rule(rule)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=400)
    logs = simulate_logs(rule)
    return {"ok": True, "logs": logs}

@router.post("/lab/grade")
def lab_grade(payload: dict = Body(...)):
    rule_json = payload.get("rule") or {}
    logs = payload.get("logs") or []
    rule = _parse_rule(rule_json)
    errs = lint_rule(rule)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=400)
    result = grade(rule, logs)
    return {"ok": True, "grade": result}
