from pydantic import BaseModel, Field, conlist
from typing import Literal, List, Dict, Set
import networkx as nx

class KGNode(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    type: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=120)

class KGEdge(BaseModel):
    src: str
    dst: str
    rel: str = Field(min_length=1, max_length=40)

class KnowledgeGraphSpec(BaseModel):
    title: str = "Knowledge Graph"
    alt_text: str = "Graph of entities and relations."
    palette: Literal["cbf"] = "cbf"
    nodes: conlist(KGNode, min_length=1)
    edges: conlist(KGEdge, min_length=0)

def verify_kg(spec: KnowledgeGraphSpec) -> dict:
    ids: Set[str] = set(n.id for n in spec.nodes)
    # endpoint validity
    for e in spec.edges:
        if e.src not in ids or e.dst not in ids:
            raise ValueError(f"edge {e.src}->{e.dst} references unknown node")
    G = nx.DiGraph()
    for n in spec.nodes: G.add_node(n.id, **n.model_dump())
    for e in spec.edges: G.add_edge(e.src, e.dst, rel=e.rel)

    # degree caps
    deg_max = max((d for _, d in G.degree()), default=0)
    if deg_max > 200:
        raise ValueError(f"graph too dense for classroom view (max degree {deg_max})")
    comps = list(nx.weakly_connected_components(G))
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "graph_ok": True
    }
