# trustviz/sources/papers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import datetime as dt
import re, httpx

# 1) OpenAlex: no auth, robust filtering
OPENALEX = "https://api.openalex.org/works"

# 2) Semantic Scholar: optional if you add a key; also works w/out for low volume
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

# 3) arXiv: for cs.CR, cs.CV adversarial, cs.LG security, etc.
import arxiv

CYBER_KEYWORDS = [
    "cybersecurity", "adversarial", "intrusion detection", "threat intelligence",
    "malware", "ransomware", "phishing", "network security", "cryptography",
    "vulnerability", "zero-day", "IDS", "SIEM", "model inversion", "data poisoning",
    "LLM jailbreaking", "prompt injection"
]
ARXIV_CATS = ["cs.CR", "cs.LG", "cs.CV", "cs.NE"]

@dataclass
class Paper:
    title: str
    year: int
    venue: Optional[str]
    url: str                  # landing page
    pdf_url: Optional[str]    # direct pdf if available
    abstract: Optional[str]
    source: str               # "openalex" | "semanticscholar" | "arxiv"
    figures_hint: Optional[str] = None  # maybe first few fig captions

def _looks_cyber(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CYBER_KEYWORDS)

async def fetch_openalex(query: str, since_year: int = 2022, limit: int = 8) -> List[Paper]:
    # Filter: recent, open access preferred, computer science/security topics if tagged
    params = {
        "search": query,
        "from_publication_date": f"{since_year}-01-01",
        "per_page": limit,
        "sort": "publication_date:desc",
    }
    async with httpx.AsyncClient(timeout=30) as cx:
        r = await cx.get(OPENALEX, params=params)
        r.raise_for_status()
        data = r.json()
    out: List[Paper] = []
    for w in data.get("results", []):
        title = w.get("title") or ""
        abstr = w.get("abstract_inverted_index")
        abstract = None
        if isinstance(abstr, dict):
            # reconstruct quick abstract
            words = []
            for k, poss in sorted(abstr.items(), key=lambda kv: kv[1][0]):
                for _ in poss:
                    words.append(k)
            abstract = " ".join(words)
        venue = (w.get("host_venue") or {}).get("display_name")
        year = (w.get("publication_year") or 0)
        url = w.get("id")
        pdf_url = None
        oa = w.get("open_access") or {}
        if oa.get("is_oa") and oa.get("oa_url"):
            pdf_url = oa.get("oa_url")
        # small topical gate
        if _looks_cyber((title or "") + " " + (abstract or "")):
            out.append(Paper(title, year, venue, url, pdf_url, abstract, "openalex"))
    return out

async def fetch_semantic_scholar(query: str, limit: int = 6) -> List[Paper]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,venue,abstract,url,openAccessPdf"
    }
    async with httpx.AsyncClient(timeout=30) as cx:
        r = await cx.get(S2_BASE, params=params)
        r.raise_for_status()
    data = r.json()
    out: List[Paper] = []
    for p in data.get("data", []):
        title = p.get("title","")
        abstract = p.get("abstract")
        if not _looks_cyber(title + " " + (abstract or "")):
            continue
        pdf_url = (p.get("openAccessPdf") or {}).get("url")
        out.append(Paper(
            title=title, year=p.get("year") or 0, venue=p.get("venue"),
            url=p.get("url"), pdf_url=pdf_url, abstract=abstract, source="semanticscholar"
        ))
    return out

async def fetch_arxiv(query: str, max_results: int = 6) -> List[Paper]:
    q = f"({query}) AND ({' OR '.join('cat:'+c for c in ARXIV_CATS)})"
    search = arxiv.Search(query=q, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = list(search.results())
    out: List[Paper] = []
    for r in results:
        title = r.title or ""
        abstract = r.summary or ""
        if _looks_cyber(title + " " + abstract):
            out.append(Paper(
                title=title, year=r.published.year if r.published else 0,
                venue="arXiv", url=r.entry_id, pdf_url=r.pdf_url,
                abstract=abstract, source="arxiv"
            ))
    return out

async def fetch_cyber_papers(query: str, since_year: int = 2022, max_total: int = 10) -> List[Paper]:
    # fan-out and then de-dup by title
    seen = set()
    out: List[Paper] = []
    for batch in [
        await fetch_openalex(query, since_year, limit=max_total),
        await fetch_semantic_scholar(query, limit=max_total),
        await fetch_arxiv(query, max_results=max_total),
    ]:
        for p in batch:
            key = re.sub(r"\W+", "", (p.title or "").lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= max_total:
                return out
    return out
