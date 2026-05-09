from .legal import LEGAL_TERMS, LEGAL_CONTRADICTIONS
from .general import GENERAL_TERMS, GENERAL_CONTRADICTIONS
from .medical import MEDICAL_TERMS, MEDICAL_CONTRADICTIONS

def get_preset(domain: str):
    if domain == "legal":
        return {"terms": LEGAL_TERMS, "contradiction_rules": LEGAL_CONTRADICTIONS}
    elif domain == "medical":
        return {"terms": MEDICAL_TERMS, "contradiction_rules": MEDICAL_CONTRADICTIONS}
    else:
        return {"terms": GENERAL_TERMS, "contradiction_rules": GENERAL_CONTRADICTIONS}
