# trustviz/lab/sim.py
import random, time
from typing import List, Dict
from trustviz.rules.core import SequenceRule

def simulate_logs(rule: SequenceRule, seed: int = 1337, positives: int = 1, negatives: int = 30) -> List[Dict]:
    rnd = random.Random(seed)
    logs: List[Dict] = []

    base0 = int(time.time()) - 3600  # ~1h ago

    # Positive chains (that satisfy the rule)
    for p in range(positives):
        base = base0 + p * 100
        ident = {k: f"v{p}_{k}" for k in rule.on}
        t = 0
        step_gap = max(2, rule.window_s // max(2, len(rule.steps)))
        for s in rule.steps:
            ev = {"ts": base + t, "event": s.event, **ident}
            for k, v in (s.where or {}).items():
                ev[k] = v
            logs.append(ev)
            t += rnd.randint(1, step_gap)

    # Negatives (near-misses)
    for i in range(negatives):
        base = base0 + 500 + i * 7
        ident = {k: f"neg{i}_{k}" for k in rule.on}
        choose = rnd.randint(1, len(rule.steps))
        for j in range(choose):
            s = rule.steps[j]
            ev = {"ts": base + j * rnd.randint(30, 90), "event": s.event, **ident}
            # break a join or a where occasionally
            if rnd.random() < 0.5:
                badk = rnd.choice(rule.on)
                ev[badk] = ev[badk] + "_off"
            if rnd.random() < 0.4 and s.where:
                k = rnd.choice(list(s.where.keys()))
                ev[k] = str(ev.get(k, "")) + "_mismatch"
            logs.append(ev)

    logs.sort(key=lambda x: x["ts"])
    return logs
