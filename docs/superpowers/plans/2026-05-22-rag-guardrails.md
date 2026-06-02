# RAG Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce hallucinations by enforcing evidence gating and strict citation grounding without adding API calls.

**Architecture:** Add an evidence policy that scores reranked nodes and decides OK/ABSTAIN/CONFLICT, then gate responses in the API. Tighten the generation prompt to enforce citations per statement and no extrapolation. Expose evidence thresholds via env vars and log decisions.

**Tech Stack:** FastAPI, llama_index, Gemini (llm_generator), Qdrant, pytest

---

## File Structure

- Create: `retrievers/evidence_policy.py` — evidence decision logic (thresholds, conflict heuristic).
- Modify: `retrievers/hybrid_search.py` — compute evidence decision and log it.
- Modify: `api.py` — apply evidence gating before generating responses.
- Modify: `generators/llm_generator.py` — stricter prompt + optional evidence status.
- Create: `tests/test_evidence_policy.py` — unit tests for evidence gating.

---

### Task 1: Add evidence policy helper + tests

**Files:**
- Create: `retrievers/evidence_policy.py`
- Create: `tests/test_evidence_policy.py`

- [ ] **Step 1: Write the failing tests**

```python
from dataclasses import dataclass
from retrievers.evidence_policy import evaluate_evidence

@dataclass
class FakeNode:
    metadata: dict

@dataclass
class FakeNodeWithScore:
    node: FakeNode
    score: float

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

def test_conflict_on_close_scores_different_files():
    nodes = [
        FakeNodeWithScore(FakeNode({"file_name": "a.pdf"}), 0.85),
        FakeNodeWithScore(FakeNode({"file_name": "b.pdf"}), 0.82),
    ]
    decision = evaluate_evidence(nodes, min_score=0.5, min_count=1, conflict_margin=0.05)
    assert decision.decision == "conflict"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evidence_policy.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'retrievers.evidence_policy'`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class EvidenceDecision:
    decision: str  # "ok" | "abstain" | "conflict"
    reason: str
    top_score: float
    node_count: int
    strong_files: list[str]

def _score(node) -> float:
    return float(getattr(node, "score", 0) or 0)

def evaluate_evidence(nodes: Sequence, min_score: float, min_count: int, conflict_margin: float) -> EvidenceDecision:
    if not nodes:
        return EvidenceDecision("abstain", "no_nodes", 0.0, 0, [])

    top_score = _score(nodes[0])
    strong = [n for n in nodes if _score(n) >= min_score]
    if top_score < min_score or len(strong) < min_count:
        return EvidenceDecision("abstain", "below_threshold", top_score, len(strong), [])

    decision = "ok"
    strong_files = []
    for n in strong:
        f = (n.node.metadata.get("file_name") if getattr(n, "node", None) else None) or "Unknown"
        if f not in strong_files:
            strong_files.append(f)

    if len(strong) >= 2:
        first, second = strong[0], strong[1]
        f1 = (first.node.metadata.get("file_name") if getattr(first, "node", None) else None) or "Unknown"
        f2 = (second.node.metadata.get("file_name") if getattr(second, "node", None) else None) or "Unknown"
        if f1 != f2 and abs(_score(first) - _score(second)) <= conflict_margin:
            decision = "conflict"

    return EvidenceDecision(decision, "threshold_pass", top_score, len(strong), strong_files)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evidence_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrievers/evidence_policy.py tests/test_evidence_policy.py
git commit -m "feat: add evidence gating helper"
```

---

### Task 2: Integrate evidence gating into hybrid retrieval

**Files:**
- Modify: `retrievers/hybrid_search.py`

- [ ] **Step 1: Write the failing test (lightweight integration check)**

```python
from retrievers.evidence_policy import evaluate_evidence

def test_evidence_threshold_behavior():
    decision = evaluate_evidence([], min_score=0.6, min_count=1, conflict_margin=0.05)
    assert decision.decision == "abstain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evidence_policy.py::test_evidence_threshold_behavior -v`  
Expected: FAIL with `test not found` (until added to test file)

- [ ] **Step 3: Implement retrieval integration**

```python
from retrievers.evidence_policy import evaluate_evidence

class AdvancedHybridRetriever:
    def __init__(...):
        ...
        self.evidence_min_score = float(os.getenv("EVIDENCE_MIN_SCORE", "0.55"))
        self.evidence_min_count = int(os.getenv("EVIDENCE_MIN_COUNT", "1"))
        self.evidence_conflict_margin = float(os.getenv("EVIDENCE_CONFLICT_MARGIN", "0.05"))

    def retrieve_and_rerank(...):
        ...
        reranked_nodes = self.reranker.postprocess_nodes(...)
        balanced_nodes = self._select_balanced_nodes(reranked_nodes, self.final_top_n)

        evidence = evaluate_evidence(
            reranked_nodes,
            min_score=self.evidence_min_score,
            min_count=self.evidence_min_count,
            conflict_margin=self.evidence_conflict_margin,
        )

        print(
            f"📊 [Evidence] decision={evidence.decision} "
            f"top_score={evidence.top_score:.3f} strong_count={evidence.node_count} "
            f"files={evidence.strong_files}"
        )

        if evidence.decision == "abstain":
            return [], combined_nodes, evidence

        return balanced_nodes, combined_nodes, evidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evidence_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrievers/hybrid_search.py tests/test_evidence_policy.py
git commit -m "feat: gate retrieval by evidence"
```

---

### Task 3: Enforce response policies in API

**Files:**
- Modify: `api.py`

- [ ] **Step 1: Write the failing test (API policy behavior)**

```python
# tests/test_api_policy.py
from fastapi.testclient import TestClient
import api

def test_abstain_returns_safe_message(monkeypatch):
    client = TestClient(api.app)

    class FakeRetriever:
        def retrieve_and_rerank(self, *args, **kwargs):
            class Evidence:
                decision = "abstain"
            return [], [], Evidence()

    api.state.retriever = FakeRetriever()
    response = client.post("/api/v1/chat", json={"query": "x", "selected_files": None})
    assert response.status_code == 200
    assert "Không tìm thấy" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_policy.py -v`  
Expected: FAIL with `No module named 'tests.test_api_policy'` (until test file is created)

- [ ] **Step 3: Implement API gating**

```python
best_nodes, raw_nodes, evidence = state.retriever.retrieve_and_rerank(req.query, req.selected_files)

if evidence.decision == "abstain":
    def empty_response():
        yield "Không tìm thấy thông tin phù hợp trong tài liệu hiện có."
    return StreamingResponse(empty_response(), media_type="text/plain")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api_policy.py
git commit -m "feat: apply abstain policy in chat"
```

---

### Task 4: Tighten prompt grounding and pass evidence status

**Files:**
- Modify: `generators/llm_generator.py`
- Modify: `api.py` (pass evidence decision)

- [ ] **Step 1: Write the failing test (prompt contains strict rules)**

```python
# tests/test_prompt_rules.py
from generators.llm_generator import ResponseGenerator

def test_prompt_contains_grounding_rules():
    gen = ResponseGenerator()
    prompt = gen.qa_prompt_tmpl.template
    assert "không được suy luận" in prompt.lower()
    assert "[file:" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_rules.py -v`  
Expected: FAIL until prompt updated

- [ ] **Step 3: Update prompt + evidence flag**

```python
class ResponseGenerator:
    def __init__(...):
        self.qa_prompt_tmpl = PromptTemplate(
            "Bạn là một trợ lý AI phân tích tài liệu chuyên nghiệp.\n"
            "Dưới đây là các thông tin được trích xuất từ tài liệu của người dùng:\n"
            "--- NGỮ CẢNH TÀI LIỆU ---\n"
            "{context_str}\n"
            "-------------------------\n"
            "TÌNH TRẠNG BẰNG CHỨNG: {evidence_status}\n"
            "NHIỆM VỤ CỦA BẠN:\n"
            "1. Trả lời chính xác dựa trên ngữ cảnh.\n"
            "2. Mỗi ý quan trọng phải có trích dẫn [File: Tên_File].\n"
            "3. Không được suy luận ngoài ngữ cảnh.\n"
            "4. Nếu mâu thuẫn, nêu rõ và trích dẫn cả hai.\n"
            "5. Nếu thiếu thông tin, nói rõ không tìm thấy trong tài liệu.\n"
            "Câu hỏi: {query_str}\n"
            "Trả lời chi tiết bằng Tiếng Việt:"
        )

    def generate_answer_stream(self, query_str, retrieved_nodes, evidence_status="OK"):
        ...
        fmt_prompt = self.qa_prompt_tmpl.format(
            context_str=context_str,
            query_str=query_str,
            evidence_status=evidence_status,
        )
```

```python
# api.py
evidence_status = evidence.decision.upper()
for chunk in state.generator.generate_answer_stream(req.query, best_nodes, evidence_status=evidence_status):
    yield chunk
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt_rules.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add generators/llm_generator.py api.py tests/test_prompt_rules.py
git commit -m "feat: enforce grounding prompt rules"
```

---

## Spec Self-Review Checklist
- Evidence gating, prompt discipline, response policies, config/logs are covered in Tasks 1–4.
- No placeholders remain.
- Function names and signatures are consistent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-22-rag-guardrails.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
