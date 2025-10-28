# trustviz/notebook/runner.py
# Minimal notebook runner for TrustViz — uvloop-friendly (no nest_asyncio)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json, time, contextlib

from jupyter_client import KernelManager

# ---- Whitelisted modules available to cells ----
SAFE_IMPORTS = {
    "json", "math", "statistics", "itertools", "collections", "random",
    "networkx", "sympy", "numpy", "sklearn"
}

# ---- Prelude executed in the kernel before any user cells ----
# Removes dangerous builtins and restricts imports. Provides ARTIFACTS dict.
PRELUDE = r"""
# --- TrustViz Notebook Prelude (restricted) ---
import builtins as _bi, sys as _sys

# Remove dangerous builtins
for _name in ["__import__", "open", "compile", "eval", "exec", "input"]:
    if hasattr(_bi, _name):
        setattr(_bi, _name, None)

# Strict import hook: allow only vetted libs
_ALLOWED = {""" + ",".join([f"'{m}'" for m in SAFE_IMPORTS]) + r"""}
_real_import = _bi.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root not in _ALLOWED:
        raise ImportError(f"Import of '{name}' is blocked")
    return _real_import(name, globals, locals, fromlist, level)
_bi.__import__ = _guarded_import

# Provide ARTIFACTS dict for cells to fill
ARTIFACTS = {}
"""

@dataclass
class CellResult:
    ok: bool
    exec_ms: int
    stdout: str
    stderr: str

@dataclass
class RunResult:
    ok: bool
    cells: List[CellResult]
    artifacts: Dict[str, Any]
    error: Optional[str] = None


class NotebookRunner:
    """
    Starts a Python kernel, runs a small set of cells with timeouts & import guard,
    captures ARTIFACTS produced by the cells, then shuts the kernel down.
    """

    def __init__(self, timeout_s: int = 8):
        self.timeout_s = timeout_s
        self.km: Optional[KernelManager] = None
        self.kc = None

    # ---------------- Lifecycle ----------------

    def start(self):
        km = KernelManager()
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=self.timeout_s)
        self.km, self.kc = km, kc
        # Run the prelude to lock down builtins/imports and create ARTIFACTS
        self._exec(PRELUDE, self.timeout_s)

    def stop(self):
        if self.kc:
            with contextlib.suppress(Exception):
                self.kc.stop_channels()
        if self.km:
            with contextlib.suppress(Exception):
                self.km.shutdown_kernel(now=True)

    # ---------------- Execution ----------------

    def _exec(self, code: str, timeout: int) -> CellResult:
        t0 = time.time()
        msg_id = self.kc.execute(code)  # type: ignore[attr-defined]
        stdout, stderr = [], []
        ok = True
        try:
            while True:
                msg = self.kc.get_iopub_msg(timeout=timeout)  # type: ignore[attr-defined]
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                mtype = msg.get("msg_type")
                cont = msg.get("content", {})
                if mtype == "stream":
                    if cont.get("name") == "stdout":
                        stdout.append(cont.get("text", ""))
                    elif cont.get("name") == "stderr":
                        stderr.append(cont.get("text", ""))
                elif mtype == "error":
                    ok = False
                    tb = cont.get("traceback") or []
                    stderr.append("\n".join(tb) if tb else cont.get("evalue", "error"))
                elif mtype == "status" and cont.get("execution_state") == "idle":
                    break
        except Exception as e:
            ok = False
            stderr.append(str(e))
        ms = int((time.time() - t0) * 1000)
        return CellResult(ok=ok, exec_ms=ms, stdout="".join(stdout), stderr="".join(stderr))

    def run(self, cells: List[str]) -> RunResult:
        if not self.kc:
            self.start()
        results: List[CellResult] = []
        ok_all = True

        for code in cells:
            res = self._exec(code, self.timeout_s)
            results.append(res)
            ok_all = ok_all and res.ok
            if not res.ok:
                break

        # Pull ARTIFACTS from the kernel
        pull = self._exec("import json; print(json.dumps(ARTIFACTS))", self.timeout_s)
        artifacts: Dict[str, Any] = {}
        if pull.ok:
            try:
                artifacts = json.loads((pull.stdout or "").strip() or "{}")
            except Exception:
                artifacts = {}

        ok_all = ok_all and pull.ok
        return RunResult(ok=ok_all, cells=results, artifacts=artifacts, error=None if ok_all else "cell_error")
