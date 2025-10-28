def kg_cyto_payload(spec: dict) -> dict:
    # spec is KnowledgeGraphSpec.model_dump()
    elements = []
    for n in spec["nodes"]:
        elements.append({"data": {"id": n["id"], "label": n["label"], "type": n["type"]}})
    for e in spec["edges"]:
        elements.append({"data": {"source": e["src"], "target": e["dst"], "rel": e["rel"]}})
    return {"elements": elements, "title": spec["title"]}
