from dataclasses import dataclass
from typing import Sequence, Optional

@dataclass(frozen=True)
class EvidenceDecision:
    decision: str  # "ok" | "abstain" | "conflict"
    reason: str
    top_score: float
    node_count: int
    strong_files: tuple[str, ...]


def _score(node) -> float:
    return float(getattr(node, "score", 0) or 0)


def _get_metadata(node) -> Optional[dict]:
    """Support both shapes: node.node.metadata and node.metadata."""
    # Try nested shape first
    nested = getattr(node, "node", None)
    if nested is not None and getattr(nested, "metadata", None) is not None:
        return nested.metadata
    # Try top-level metadata
    md = getattr(node, "metadata", None)
    if md is not None:
        return md
    return None


def evaluate_evidence(nodes: Sequence, min_score: float, min_count: int, conflict_margin: float) -> EvidenceDecision:
    if not nodes:
        return EvidenceDecision("abstain", "no_nodes", 0.0, 0, ())

    # Ensure nodes are sorted by score descending for robust behavior
    sorted_nodes = sorted(nodes, key=_score, reverse=True)

    top_score = _score(sorted_nodes[0])
    strong = [n for n in sorted_nodes if _score(n) >= min_score]
    if top_score < min_score or len(strong) < min_count:
        return EvidenceDecision("abstain", "below_threshold", top_score, len(strong), ())

    decision = "ok"
    strong_files = []
    for n in strong:
        md = _get_metadata(n)
        f = md.get("file_name") if isinstance(md, dict) else None
        if not f:
            f = "Unknown"
        if f not in strong_files:
            strong_files.append(f)

    if len(strong) >= 2:
        first, second = strong[0], strong[1]
        f1 = (_get_metadata(first) or {}).get("file_name") or "Unknown"
        f2 = (_get_metadata(second) or {}).get("file_name") or "Unknown"
        if f1 != f2 and abs(_score(first) - _score(second)) <= conflict_margin:
            decision = "conflict"

    return EvidenceDecision(decision, "threshold_pass", top_score, len(strong), tuple(strong_files))
