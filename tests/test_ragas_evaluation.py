import unittest
from unittest.mock import patch

from evaluate_ragas import (
    _load_metric_objects,
    build_ragas_record,
    build_summary_from_case_scores,
    choose_metrics,
    filter_metrics,
    ensure_metric_object,
    extract_metric_score,
    parse_args,
)


class TestRagasEvaluationHelpers(unittest.TestCase):
    def test_build_ragas_record_with_ground_truth(self):
        case = {"query": "q1", "ground_truth": "gt"}
        answer = "a1"
        contexts = ["c1", "c2"]

        record = build_ragas_record(case, answer, contexts)

        self.assertEqual(record["question"], "q1")
        self.assertEqual(record["answer"], "a1")
        self.assertEqual(record["contexts"], ["c1", "c2"])
        self.assertEqual(record["ground_truth"], "gt")
        self.assertEqual(record["user_input"], "q1")
        self.assertEqual(record["response"], "a1")
        self.assertEqual(record["retrieved_contexts"], ["c1", "c2"])
        self.assertEqual(record["reference"], "gt")

    def test_build_ragas_record_without_ground_truth(self):
        case = {"query": "q2"}
        record = build_ragas_record(case, "a2", ["c1"])

        self.assertEqual(record["question"], "q2")
        self.assertNotIn("ground_truth", record)

    def test_choose_metrics_without_ground_truth(self):
        metrics = choose_metrics(has_ground_truth=False)
        self.assertIn("faithfulness", metrics)
        self.assertIn("answer_relevancy", metrics)
        self.assertNotIn("context_recall", metrics)

    def test_choose_metrics_with_ground_truth(self):
        metrics = choose_metrics(has_ground_truth=True)
        self.assertIn("faithfulness", metrics)
        self.assertIn("answer_relevancy", metrics)
        self.assertIn("context_precision", metrics)
        self.assertIn("context_recall", metrics)

    def test_filter_metrics_supports_specific_metric(self):
        all_metrics = choose_metrics(has_ground_truth=True)
        filtered = filter_metrics(all_metrics, ["context_precision"])
        self.assertEqual(filtered, ["context_precision"])

    def test_filter_metrics_raises_for_unknown_metric(self):
        with self.assertRaises(ValueError):
            filter_metrics(["faithfulness", "answer_relevancy"], ["not_a_metric"])

    def test_filter_metrics_raises_for_unavailable_metric(self):
        with self.assertRaises(ValueError):
            filter_metrics(["faithfulness", "answer_relevancy"], ["context_precision"])

    def test_filter_metrics_maps_context_precision_to_no_reference_variant(self):
        filtered = filter_metrics(
            ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"],
            ["context_precision"],
        )
        self.assertEqual(filtered, ["llm_context_precision_without_reference"])

    def test_default_judge_model_is_gemini_flash_latest(self):
        with patch("sys.argv", ["evaluate_ragas.py"]):
            args = parse_args()
        self.assertEqual(args.judge_model, "gemini-flash-latest")
        self.assertEqual(args.delay_seconds, 75)

    def test_parse_args_metrics_single(self):
        with patch("sys.argv", ["evaluate_ragas.py", "--metrics", "context_precision"]):
            args = parse_args()
        self.assertEqual(args.metrics, ["context_precision"])

    def test_ensure_metric_object_instantiates_classes(self):
        class DummyMetric:
            pass

        metric = ensure_metric_object(DummyMetric)
        self.assertIsInstance(metric, DummyMetric)

    def test_ensure_metric_object_keeps_instances(self):
        class DummyMetric:
            pass

        metric_obj = DummyMetric()
        self.assertIs(ensure_metric_object(metric_obj), metric_obj)

    def test_load_metric_objects_returns_metric_instances(self):
        metrics = _load_metric_objects(["faithfulness", "answer_relevancy"])
        self.assertEqual(len(metrics), 2)
        for metric in metrics:
            self.assertNotEqual(metric.__class__.__name__, "module")

    def test_extract_metric_score_from_direct_key(self):
        score = extract_metric_score({"faithfulness": 0.91}, "faithfulness")
        self.assertEqual(score, 0.91)

    def test_extract_metric_score_from_result_string(self):
        score = extract_metric_score({"result": "{'faithfulness': 0.75}"}, "faithfulness")
        self.assertEqual(score, 0.75)

    def test_build_summary_from_case_scores_ignores_none(self):
        case_scores = [
            {"scores": {"faithfulness": 0.8, "answer_relevancy": None}},
            {"scores": {"faithfulness": 0.6, "answer_relevancy": 0.5}},
        ]
        summary = build_summary_from_case_scores(case_scores, ["faithfulness", "answer_relevancy"])
        self.assertEqual(summary["faithfulness"], 0.7)
        self.assertEqual(summary["answer_relevancy"], 0.5)


if __name__ == "__main__":
    unittest.main()
