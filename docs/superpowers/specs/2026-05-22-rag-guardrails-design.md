# RAG Guardrails Design (No Extra API Calls)

## Goal
Reduce hallucinations while keeping Gemini + current reranker and **not increasing API call count** or noticeable latency.

## Non-Goals
- Switching LLM providers or adding external verification calls.
- Rebuilding ingestion pipeline or changing embedding model.

## Current Context
- FastAPI backend (`api.py`) streams answers from `generators/llm_generator.py`.
- Retrieval is hybrid vector + BM25 with rerank (`retrievers/hybrid_search.py`).
- Gemini used for query rewrite and answer generation.

## Approach Summary
Tighten evidence selection and grounding rules so responses are limited to strong retrieved evidence, and enforce strict citation discipline in prompt. Add configurable evidence thresholds and decision logging.

## Design

### 1) Retrieval & Evidence Gating
- Normalize and apply `file_name` filters consistently across vector + BM25.
- Prefer **top evidence** by rerank score over balanced spread when a dominant source exists.
- Introduce minimum evidence thresholds:
  - `EVIDENCE_MIN_SCORE`: top rerank score must meet a minimum.
  - `EVIDENCE_MIN_COUNT`: require at least N nodes above threshold.
  - `EVIDENCE_STRONG_MARGIN`: optional margin between top-1 and top-2 to allow focus.
- If thresholds are not met → **ABSTAIN** (no LLM call; return a safe fallback).

### 2) Prompt Grounding & Citation Discipline
- Update prompt to require:
  - Every important statement includes `[File: ...]`.
  - No extrapolation beyond provided context.
  - If evidence conflicts, acknowledge conflict with citations to both.
  - If evidence is insufficient, explicitly say so.

### 3) Response Policies & Fallbacks
Define response modes:
- **OK**: evidence passes thresholds → normal answer.
- **ABSTAIN**: insufficient evidence → short, safe response (“Không tìm thấy trong tài liệu hiện có.”).
- **CONFLICT**: conflicting evidence → highlight conflict with citations.

### 4) Configuration & Observability
Add env-driven settings for evidence thresholds and log:
- Top scores, node counts, selected files.
- Decision: OK / ABSTAIN / CONFLICT.

## Data Flow
1. Retrieve + rerank nodes.
2. Apply evidence gating thresholds.
3. If ABSTAIN → return fallback immediately.
4. If OK/CONFLICT → generate response with stricter prompt.

## Error Handling
- Preserve existing API error handling.
- Ensure ABSTAIN is not treated as an error.

## Testing Strategy
- Unit tests for evidence gating (threshold behavior).
- Integration tests for chat endpoint:
  - No evidence → ABSTAIN.
  - Strong evidence → OK with citations.
  - Conflicting evidence → CONFLICT format.

## Rollout Notes
- Default thresholds conservative to avoid hallucination.
- Adjust via env vars after observing logs.
