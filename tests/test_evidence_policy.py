from dataclasses import dataclass
from retrievers.evidence_policy import evaluate_evidence

@dataclass
class FakeNode:
    metadata: dict

@dataclass
class FakeNodeWithScore:
    node: FakeNode
    score: float

@dataclass
class FakeTopLevel:
    metadata: dict
    score: float


def test_evidence_threshold_behavior():
    decision = evaluate_evidence([], min_score=0.6, min_count=1, conflict_margin=0.05)
    assert decision.decision == "abstain"


def test_abstain_on_no_nodes():
    decision = evaluate_evidence([], min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "abstain"


def test_abstain_on_low_score():
    nodes = [FakeNodeWithScore(FakeNode({"file_name": "a.pdf"}), 0.2)]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "abstain"


def test_ok_on_strong_evidence():
    nodes = [FakeNodeWithScore(FakeNode({"file_name": "a.pdf"}), 0.8)]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "ok"
    assert decision.top_score == 0.8
    assert decision.node_count == 1
    assert isinstance(decision.strong_files, tuple)
    assert decision.strong_files == ("a.pdf",)


def test_conflict_on_close_scores_different_files():
    nodes = [
        FakeNodeWithScore(FakeNode({"file_name": "a.pdf"}), 0.85),
        FakeNodeWithScore(FakeNode({"file_name": "b.pdf"}), 0.82),
    ]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "conflict"
    assert decision.strong_files == ("a.pdf", "b.pdf")


def test_unsorted_input_is_handled_and_sorted():
    # highest score is second element; function should sort and detect conflict
    nodes = [
        FakeNodeWithScore(FakeNode({"file_name": "b.pdf"}), 0.82),
        FakeNodeWithScore(FakeNode({"file_name": "a.pdf"}), 0.85),
    ]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=2, conflict_margin=0.05)
    # after sorting top_score should be 0.85 and two strong nodes -> conflict because diff 0.03 <= 0.05 and files differ
    assert decision.top_score == 0.85
    assert decision.node_count == 2
    assert decision.strong_files == ("a.pdf", "b.pdf")
    assert decision.decision == "conflict"


def test_top_level_metadata_supported():
    nodes = [FakeTopLevel({"file_name": "top.pdf"}, 0.9)]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "ok"
    assert decision.top_score == 0.9
    assert decision.node_count == 1
    assert isinstance(decision.strong_files, tuple)
    assert decision.strong_files == ("top.pdf",)
