# trustviz/llm/diagram_templates.py
# Deterministic, topic-specific Mermaid storyboards (3 panels each).

from __future__ import annotations
import re

# --- Heuristic topic router (Coursera units + common asks) ---
def pick_topic(prompt: str) -> str:
    s = (prompt or "").lower()
    if any(k in s for k in ["asset inventory", "inventory", "cmdb"]): return "asset_risk"
    if "risk triage" in s or "risk" in s and "triage" in s: return "asset_risk"
    if any(k in s for k in ["cia triad", "confidentiality", "integrity", "availability"]): return "cia"
    if any(k in s for k in ["network", "segmentation", "firewall", "ids", "ips"]): return "network"
    if "phish" in s: return "phishing"
    if any(k in s for k in ["cloud", "s3", "bucket", "iam", "public access"]): return "cloud"
    if any(k in s for k in ["detection", "soc", "triage", "siem", "edr"]): return "detect_response"
    if any(k in s for k in ["zero trust", "zerotrust"]): return "zerotrust"
    if any(k in s for k in ["linux", "sudo", "sql", "db", "database"]): return "linux_sql"
    return "breach"  # safe default

def _caption(prompt: str, max_len=60) -> str:
    s = re.sub(r"\s+", " ", (prompt or "").strip())
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

# --- Panels return a tuple of three Mermaid strings: (process, controls, improvement) ---
def storyboard(topic: str, prompt: str) -> tuple[str, str, str]:
    cap = _caption(prompt)

    if topic == "asset_risk":
        panel1 = f"""flowchart LR
  subgraph Context: {cap}
  end
  A[Asset Inventory] --> B[Classification (Criticality)]
  B --> C[Threat/Vuln Mapping]
  C --> D[Risk Triage]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class A,B,C,D blk;
"""
        panel2 = """flowchart LR
  A[Asset Inventory] --> B[Classification]
  B --> C[Threat/Vuln Mapping]
  C --> D[Risk Triage]
  subgraph Controls
    K1[Config Baselines]:::def
    K2[Patch SLAs]:::def
    K3[Network Segmentation]:::def
    K4[Backups/DR]:::def
  end
  B -.-> K1
  C -.-> K2
  C -.-> K3
  D -.-> K4
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class A,B,C,D blk;
"""
        panel3 = """flowchart LR
  M1[Measure exposure] --> M2[Prioritize backlog]
  M2 --> M3[Implement control]
  M3 --> M4[Verify (scan/audit)]
  M4 --> M1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class M1,M2,M3,M4 cyc;
"""
        return panel1, panel2, panel3

    if topic == "cia":
        p1 = """flowchart LR
  CI[Confidentiality] --- IN[Integrity] --- AV[Availability]
  U[Use Cases] --> CI
  U --> IN
  U --> AV
  classDef blk fill:#fef3c7,stroke:#92400e,color:#111;
  class CI,IN,AV,U blk;
"""
        p2 = """flowchart LR
  subgraph Threat Flow
    A[Recon] --> B[Initial Access] --> C[Privilege Escalation] --> D[Data Access]
  end
  subgraph Controls by CIA
    C1[Encrypt/Mask]:::def
    C2[WORM/Checksums]:::def
    C3[HA/Backups]:::def
  end
  D -.-> C1
  C -.-> C2
  B -.-> C3
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  L1[Assess CIA gaps] --> L2[Select control]
  L2 --> L3[Test/Validate]
  L3 --> L4[Monitor & Improve]
  L4 --> L1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class L1,L2,L3,L4 cyc;
"""
        return p1, p2, p3

    if topic == "network":
        p1 = """flowchart LR
  H1[User Net] --> FW[Firewall]
  FW --> DMZ[DMZ Services]
  DMZ --> APP[App Tier]
  APP --> DB[DB Tier]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class H1,FW,DMZ,APP,DB blk;
"""
        p2 = """flowchart LR
  H1[User Net] --> FW[Firewall]
  FW --> DMZ[DMZ Services]
  DMZ --> APP[App Tier]
  APP --> DB[DB Tier]
  subgraph Network Controls
    S1[Segmentation/ACLs]:::def
    S2[WAF/IDS/IPS]:::def
    S3[EDR]:::def
    S4[DLP/Egress]:::def
  end
  FW -.-> S1
  DMZ -.-> S2
  APP -.-> S3
  DB -.-> S4
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  M1[Collect Net Telemetry] --> M2[Detect & Triage]
  M2 --> M3[Contain (ACL/Quarantine)]
  M3 --> M4[Tune rules/Playbooks]
  M4 --> M1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class M1,M2,M3,M4 cyc;
"""
        return p1, p2, p3

    if topic == "phishing":
        p1 = """flowchart LR
  T1[Recon/Targeting] --> T2[Phish Email/SMS] --> T3[User Click] --> T4[Token/Cred Steal] --> T5[Initial Access]
  classDef atk fill:#fdecea,stroke:#e53935,color:#111;
  class T1,T2,T3,T4,T5 atk;
"""
        p2 = """flowchart LR
  T2[Phish] --> T3[User Click] --> T4[Token Steal]
  subgraph Controls
    C1[Email Filter + DMARC]:::def
    C2[MFA (phish-resistant)]:::def
    C3[SSO/Short TTL]:::def
    C4[Awareness/Sim]:::def
  end
  T2 -.-> C1
  T3 -.-> C4
  T4 -.-> C2
  T4 -.-> C3
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  L1[Simulate] --> L2[Measure Click/Report]
  L2 --> L3[Targeted Coaching]
  L3 --> L4[Re-test]
  L4 --> L1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class L1,L2,L3,L4 cyc;
"""
        return p1, p2, p3

    if topic == "cloud":
        p1 = """flowchart LR
  C1[Public Bucket/Misconfig] --> C2[Unauth Access] --> C3[Data Enumeration] --> C4[Bulk Download]
  classDef atk fill:#fdecea,stroke:#e53935,color:#111;
  class C1,C2,C3,C4 atk;
"""
        p2 = """flowchart LR
  C1[Public Bucket/Misconfig] --> C2[Unauth Access] --> C3[Data Enumeration] --> C4[Bulk Download]
  subgraph Controls
    D1[Block Public Access]:::def
    D2[IAM Least Privilege]:::def
    D3[DLP/Object Lock]:::def
  end
  C1 -.-> D1
  C2 -.-> D2
  C4 -.-> D3
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  M1[Config Scan] --> M2[Fix IaC/Policies]
  M2 --> M3[Drift Detection]
  M3 --> M1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class M1,M2,M3 cyc;
"""
        return p1, p2, p3

    if topic == "detect_response":
        p1 = """flowchart LR
  D1[Signal (SIEM/EDR)] --> D2[Triage] --> D3[Contain] --> D4[Eradicate] --> D5[Recover]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class D1,D2,D3,D4,D5 blk;
"""
        p2 = """flowchart LR
  D2[Triage] --> D3[Contain]
  subgraph SOC Controls
    C1[Use Cases/Rules]:::def
    C2[Playbooks/SOAR]:::def
    C3[Threat Intel]:::def
    C4[EDR Isolation]:::def
  end
  D2 -.-> C1
  D2 -.-> C3
  D3 -.-> C2
  D3 -.-> C4
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  P1[Post-Incident Review] --> P2[Control Gaps]
  P2 --> P3[Tune Detections]
  P3 --> P1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class P1,P2,P3 cyc;
"""
        return p1, p2, p3

    if topic == "zerotrust":
        p1 = """flowchart LR
  Z1[Request] --> Z2[Policy (Id + Device + Context)] --> Z3[Grant Least Priv] --> Z4[Continuous Verify]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class Z1,Z2,Z3,Z4 blk;
"""
        p2 = """flowchart LR
  subgraph Policy Inputs
    A1[MFA/FIDO2]:::def
    A2[Device Posture]:::def
    A3[Network Microseg]:::def
    A4[Risk Signals]:::def
  end
  Z2[Policy Engine] <.-> A1
  Z2 <.-> A2
  Z2 <.-> A3
  Z2 <.-> A4
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  L1[Measure policy impact] --> L2[Refine rules]
  L2 --> L3[Re-evaluate access]
  L3 --> L1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class L1,L2,L3 cyc;
"""
        return p1, p2, p3

    if topic == "linux_sql":
        p1 = """flowchart LR
  U1[User] --> SU[Sudo Misuse]
  SU --> LM[Lateral Movement (ssh)]
  LM --> DB[SQL Data Access]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class U1,SU,LM,DB blk;
"""
        p2 = """flowchart LR
  SU[Sudo] --> LM[SSH]
  LM --> DB[SQL]
  subgraph Controls
    C1[Least Privilege/No-Password Sudo]:::def
    C2[EDR + Auditd]:::def
    C3[DB Roles & Query Audit]:::def
  end
  SU -.-> C1
  LM -.-> C2
  DB -.-> C3
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
        p3 = """flowchart LR
  R1[Collect audit logs] --> R2[Alert on misuse]
  R2 --> R3[Remediate & tighten roles]
  R3 --> R1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class R1,R2,R3 cyc;
"""
        return p1, p2, p3

    # --- Default breach sequence ---
    p1 = """flowchart LR
  A[Recon] --> B[Initial Access] --> C[Priv Esc/Lateral] --> D[Data Discovery] --> E[Exfiltration]
  classDef blk fill:#f3f4f6,stroke:#111,color:#111;
  class A,B,C,D,E blk;
"""
    p2 = """flowchart LR
  B[Initial Access] --> C[Priv Esc/Lateral] --> D[Data Discovery] --> E[Exfiltration]
  subgraph Controls
    M1[MFA]:::def
    M2[EDR]:::def
    M3[DLP]:::def
    M4[Egress Monitor]:::def
  end
  B -.-> M1
  C -.-> M2
  D -.-> M3
  E -.-> M4
  classDef def fill:#e8f5e9,stroke:#2e7d32,color:#111,stroke-width:2px;
"""
    p3 = """flowchart LR
  I1[Instrument/Monitor] --> I2[Detect] --> I3[Respond] --> I4[Learn]
  I4 --> I1
  classDef cyc fill:#eef2ff,stroke:#3730a3,stroke-width:2px,color:#111;
  class I1,I2,I3,I4 cyc;
"""
    return p1, p2, p3
