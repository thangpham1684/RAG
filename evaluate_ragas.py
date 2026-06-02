import argparse
import ast
import json
import os
import time
from typing import Any

from dotenv import load_dotenv

from evaluate_rag_benchmark import build_or_load_retriever, load_cases
from generators.llm_generator import ResponseGenerator
from logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG quality with RAGAS + Gemini.")
    parser.add_argument(
        "--benchmark-file",
        default=os.path.join("tests", "rag_benchmark_vn_en.json"),
        help="Path to benchmark JSON.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only first N cases.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index before evaluation.")
    parser.add_argument(
        "--generator-model",
        default="models/gemini-flash-latest",
        help="Gemini model used to generate system answers.",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-flash-latest",
        help="Gemini model used by RAGAS evaluator LLM.",
    )
    parser.add_argument(
        "--judge-embedding-model",
        default="models/embedding-001",
        help="Embedding model for RAGAS metrics.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=75,
        help="Delay between metric calls to reduce minute-level quota bursts.",
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help="Resume from existing output-json if available.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metrics to run (e.g. context_precision faithfulness).",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save report JSON.")
    return parser.parse_args()


def resolve_google_api_key() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def build_ragas_record(case: dict[str, Any], answer: str, contexts: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question": case["query"],
        "answer": answer,
        "contexts": contexts,
        "user_input": case["query"],
        "response": answer,
        "retrieved_contexts": contexts,
    }
    ground_truth = case.get("ground_truth") or case.get("reference")
    if ground_truth:
        record["ground_truth"] = ground_truth
        record["reference"] = ground_truth
    return record


def choose_metrics(has_ground_truth: bool) -> list[str]:
    base = ["faithfulness", "answer_relevancy"]
    if has_ground_truth:
        base.extend(["context_precision", "context_recall"])
    return base


def filter_metrics(available_metrics: list[str], requested_metrics: list[str] | None) -> list[str]:
    if not requested_metrics:
        return available_metrics

    alias_map = {
        "context_precision": "context_precision"
        if "context_precision" in available_metrics
        else "llm_context_precision_without_reference"
    }
    normalized = [alias_map.get(metric, metric) for metric in requested_metrics]

    known_metrics = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "llm_context_precision_without_reference",
    }
    unknown = [metric for metric in normalized if metric not in known_metrics]
    if unknown:
        raise ValueError(f"Unknown metrics requested: {unknown}")

    unavailable = [metric for metric in normalized if metric not in available_metrics]
    if unavailable:
        raise ValueError(
            f"Requested metrics unavailable for current data: {unavailable}. "
            f"Available metrics: {available_metrics}"
        )
    return normalized


def ensure_metric_object(metric_candidate):
    if isinstance(metric_candidate, type):
        return metric_candidate()
    return metric_candidate


def _load_metric_objects(metric_names: list[str]):
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        _LLMContextPrecisionWithoutReference,
    )

    metric_map = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "llm_context_precision_without_reference": _LLMContextPrecisionWithoutReference,
    }
    return [ensure_metric_object(metric_map[name]) for name in metric_names]


def _build_judges(judge_model: str, embedding_model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(model=judge_model, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    return llm, embeddings


def _evaluation_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    try:
        return dict(result)
    except Exception:
        return {"result": str(result)}


def extract_metric_score(result_dict: dict[str, Any], metric_name: str) -> float | None:
    direct = result_dict.get(metric_name)
    if isinstance(direct, (int, float)):
        return float(direct)

    raw = result_dict.get("result")
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            val = parsed.get(metric_name)
            if isinstance(val, (int, float)):
                return float(val)
        except Exception:
            return None
    return None


def build_summary_from_case_scores(case_scores: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for metric in metric_names:
        vals = [
            float(case.get("scores", {}).get(metric))
            for case in case_scores
            if isinstance(case.get("scores", {}).get(metric), (int, float))
        ]
        summary[metric] = round(sum(vals) / len(vals), 4) if vals else None
    return summary


def _load_existing_case_scores(output_json: str | None) -> list[dict[str, Any]]:
    if not output_json or not os.path.exists(output_json):
        return []
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload.get("case_scores"), list):
            return payload["case_scores"]
    except Exception:
        return []
    return []


def _save_progress(
    output_json: str,
    summary: dict[str, Any],
    metric_names: list[str],
    has_ground_truth: bool,
    records: list[dict[str, Any]],
    case_scores: list[dict[str, Any]],
):
    payload = {
        "summary": summary,
        "metrics_used": metric_names,
        "ground_truth_used": has_ground_truth,
        "total_cases": len(records),
        "samples": records,
        "case_scores": case_scores,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    load_dotenv(".env")
    resolve_google_api_key()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for Gemini-based RAGAS evaluation.")

    cases = load_cases(args.benchmark_file, args.max_cases)
    retriever = build_or_load_retriever(rebuild_index=args.rebuild_index)
    generator = ResponseGenerator(model_name=args.generator_model)

    records: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        query = case["query"]
        selected_files = case.get("selected_files")
        best_nodes, _, _ = retriever.retrieve_and_rerank(query, selected_files)

        answer_chunks = []
        for chunk in generator.generate_answer_stream(query, best_nodes):
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)
        contexts = [n.node.text for n in best_nodes]
        records.append(build_ragas_record(case, answer, contexts))
        logger.info(f"[{idx}/{len(cases)}] prepared sample for query: {query[:80]}")

    has_ground_truth = all(bool(r.get("ground_truth")) for r in records)
    available_metrics = choose_metrics(has_ground_truth)
    if args.metrics and not has_ground_truth:
        available_metrics.append("llm_context_precision_without_reference")
    metric_names = filter_metrics(available_metrics, args.metrics)
    metric_objects = _load_metric_objects(metric_names)

    from ragas import evaluate

    judge_llm, judge_embeddings = _build_judges(args.judge_model, args.judge_embedding_model)
    from datasets import Dataset

    existing_scores = _load_existing_case_scores(args.output_json) if args.resume_output else []
    score_index = {}
    case_scores: list[dict[str, Any]] = []
    for item in existing_scores:
        idx = item.get("case_index")
        if isinstance(idx, int):
            score_index[idx] = item
            case_scores.append(item)

    for case_index, record in enumerate(records):
        if case_index not in score_index:
            score_index[case_index] = {
                "case_index": case_index,
                "question": record.get("question"),
                "scores": {},
                "errors": {},
            }
            case_scores.append(score_index[case_index])

    total_calls = len(records) * len(metric_names)
    done_calls = 0
    for case_index, record in enumerate(records):
        for metric_name, metric_obj in zip(metric_names, metric_objects):
            case_state = score_index[case_index]
            if metric_name in case_state["scores"]:
                done_calls += 1
                continue
            ds = Dataset.from_list([record])
            try:
                result = evaluate(ds, metrics=[metric_obj], llm=judge_llm, embeddings=judge_embeddings)
                result_dict = _evaluation_result_to_dict(result)
                score = extract_metric_score(result_dict, metric_name)
                case_state["scores"][metric_name] = score
                if metric_name in case_state["errors"]:
                    case_state["errors"].pop(metric_name, None)
            except Exception as exc:
                case_state["scores"][metric_name] = None
                case_state["errors"][metric_name] = str(exc)

            done_calls += 1
            summary_progress = build_summary_from_case_scores(case_scores, metric_names)
            logger.info("[progress] %d/%d case=%d/%d metric=%s score=%s",
                done_calls, total_calls, case_index + 1, len(records),
                metric_name, case_state['scores'][metric_name])
            if args.output_json:
                _save_progress(
                    args.output_json,
                    summary_progress,
                    metric_names,
                    has_ground_truth,
                    records,
                    case_scores,
                )
            if done_calls < total_calls and args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    result_dict = build_summary_from_case_scores(case_scores, metric_names)

    logger.info("\n=== RAGAS SUMMARY ===")
    logger.info(result_dict)
    logger.info("metrics_used: %s", metric_names)
    logger.info("ground_truth_used: %s", has_ground_truth)

    if args.output_json:
        _save_progress(
            args.output_json,
            result_dict,
            metric_names,
            has_ground_truth,
            records,
            case_scores,
        )
        logger.info("saved_report: %s", args.output_json)


if __name__ == "__main__":
    main()
