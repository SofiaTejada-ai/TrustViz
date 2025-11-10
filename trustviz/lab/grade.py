# trustviz/lab/grade.py
from typing import List, Dict, Tuple
from trustviz.rules.core import SequenceRule

def grade(rule: SequenceRule, logs: List[Dict]) -> Dict:
    # find positive identities (by simulator convention: v0_)
    gold: Dict[Tuple, List[Dict]] = {}
    for e in logs:
        ids = tuple(e.get(k, "") for k in rule.on)
        if any(str(v).startswith("v0_") for v in ids):
            gold.setdefault(ids, []).append(e)

    ok_keys: List[Tuple] = []
    for key, evs in gold.items():
        evs_sorted = sorted(evs, key=lambda x: x["ts"])
        pos = 0
        t0 = None
        for s in rule.steps:
            found = next(
                (ev for ev in evs_sorted
                 if ev["event"] == s.event
                 and all(ev.get(k) == v for k, v in (s.where or {}).items())),
                None
            )
            if not found:
                break
            if pos == 0:
                t0 = found["ts"]
            pos += 1
        if pos == len(rule.steps) and t0 is not None and (evs_sorted[-1]["ts"] - t0) <= rule.window_s:
            ok_keys.append(key)

    coverage = len(ok_keys) > 0
    return {
        "pass": coverage,
        "covered_identities": [list(k) for k in ok_keys],
        "notes": [] if coverage else ["Sequence not fully matched within window or join keys mismatched"]
    }
