from .pie_spec import PieSpec
def verify_pie(raw: dict) -> PieSpec:
    return PieSpec(**raw)