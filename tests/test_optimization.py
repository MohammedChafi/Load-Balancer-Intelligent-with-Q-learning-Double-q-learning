import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from app.optimization import (Optimizer, parameters, decode_point, latin_points,
                              cholesky, solve_cholesky, bayesian_point, hyperband_brackets)


class SearchDesign(unittest.TestCase):
    def test_validation_rejects_nonfinite_and_invalid_ranges(self):
        for change in ({"gamma_max": 1}, {"eta_min": 0}, {"episodes": 50.5},
                       {"episodes": True}, {"eta_max": float("nan")},
                       {"gamma_min": .9, "gamma_max": .8}, {"method": "invented"},
                       {"method": "bayesian", "trials": 4}, {"method": "hyperband", "episodes": 50}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parameters(change)

    def test_log_eta_and_linear_gamma(self):
        p = parameters({"eta_min": .001, "eta_max": 1, "gamma_min": 0, "gamma_max": .98})
        pair = decode_point((.5, .5), p)
        self.assertAlmostEqual(pair["eta"], math.sqrt(.001))
        self.assertAlmostEqual(pair["gamma"], .49)

    def test_latin_hypercube_stratifies_both_axes(self):
        points = latin_points(12, random.Random(8))
        for axis in range(2):
            self.assertEqual(sorted(int(p[axis]*12) for p in points), list(range(12)))
        self.assertEqual(points, latin_points(12, random.Random(8)))

    def test_gp_linear_algebra(self):
        matrix = [[4., 2.], [2., 3.]]
        result = solve_cholesky(cholesky(matrix), [6., 5.])
        self.assertAlmostEqual(result[0], 1.)
        self.assertAlmostEqual(result[1], 1.)

    def test_bayesian_acquisition_uses_observed_scores(self):
        positions = [(0, 0), (.25, .3), (.5, .5), (.7, .8), (1, 1)]
        first = [(p, -(p[0]**2+p[1]**2), .001) for p in positions]
        second = [(p, -((1-p[0])**2+(1-p[1])**2), .001) for p in positions]
        a = bayesian_point(first, random.Random(12))
        b = bayesian_point(second, random.Random(12))
        self.assertNotEqual(a, b)
        self.assertLess(sum(a), sum(b))
        self.assertTrue(all(0 <= x <= 1 for x in (*a, *b)))

    def test_hyperband_has_multiple_brackets_and_full_budget(self):
        brackets = hyperband_brackets(500, 13)
        self.assertEqual(sum(n for n, _ in brackets), 13)
        self.assertEqual(len(brackets), 3)
        self.assertEqual([len(b) for _, b in brackets], [3, 2, 1])
        for _, budgets in brackets:
            self.assertEqual(budgets[-1], 500)
            self.assertEqual(sorted(set(budgets)), budgets)


class OptimizationIntegration(unittest.TestCase):
    def test_resume_and_validation_preserve_training_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            optimizer = Optimizer(folder)
            p = parameters({"episodes": 150, "repeats": 2, "validation_seeds": 2})
            optimizer.state.update(work_done=0, work_total=100000, current={})
            for name in ("qlearning", "double_qlearning"):
                resumed = optimizer._new_candidate(name, (.4, .8), p)
                full = optimizer._new_candidate(name, (.4, .8), p)
                optimizer._train(resumed, 50, p)
                tables = [copy.deepcopy(a.to_dict()) for a in resumed["agents"]]
                rngs = [a.rng.getstate() for a in resumed["agents"]]
                v1 = optimizer._evaluate(resumed, p)
                v2 = optimizer._evaluate(resumed, p)
                self.assertEqual(v1, v2)
                self.assertEqual(tables, [a.to_dict() for a in resumed["agents"]])
                self.assertEqual(rngs, [a.rng.getstate() for a in resumed["agents"]])
                optimizer._train(resumed, 150, p)
                optimizer._train(full, 150, p)
                self.assertEqual([a.to_dict() for a in resumed["agents"]], [a.to_dict() for a in full["agents"]])

    def test_all_five_methods_train_select_test_and_persist(self):
        for method in ("grid", "random", "bayesian", "lhs", "hyperband"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                optimizer = Optimizer(folder)
                optimizer.start({"method": method, "episodes": 150 if method=="hyperband" else 50,
                                 "algorithm": "qlearning", "trials": 6, "grid_size": 2,
                                 "repeats": 1, "validation_seeds": 2, "test_seeds": 5})
                optimizer.thread.join(60)
                state = optimizer.snapshot()
                self.assertFalse(state["running"])
                self.assertEqual(state["status"], "completed", state["message"])
                self.assertAlmostEqual(state["work_done"], state["work_total"])
                result = state["results"]["qlearning"]
                self.assertGreaterEqual(result["validation"]["score"], result["baseline_validation"]["score"])
                self.assertEqual(result["test"]["seeds_per_scenario"], 5)
                self.assertEqual(result["test"]["scenarios"].keys(), {"faible", "moyen", "élevé"})
                self.assertTrue(result["learning_curve"])
                report = optimizer.report()
                json.dumps(report, allow_nan=False)
                self.assertIn("qlearning", report["models"])
                self.assertEqual(Optimizer(folder).snapshot()["id"], state["id"])
                if method == "hyperband":
                    self.assertTrue(any(r["status"]=="pruned" for r in state["trials"]))

    def test_cancellation_and_single_search_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            optimizer = Optimizer(folder)
            optimizer.start({"episodes": 5000})
            with self.assertRaises(RuntimeError):
                optimizer.start({})
            optimizer.stop()
            optimizer.thread.join(10)
            self.assertEqual(optimizer.snapshot()["status"], "stopped")
            self.assertFalse(optimizer.snapshot()["running"])

    def test_final_test_does_not_reselect_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            optimizer = Optimizer(folder)
            original_evaluate = optimizer._evaluate

            def evaluate(candidate, p, split="validation"):
                metrics = original_evaluate(candidate, p, split)
                # Force validation to prefer candidates and test to prefer default.
                metrics["score"] = (-2 if candidate["label"]=="baseline" else -1) if split=="validation" else (-1 if candidate["label"]=="baseline" else -3)
                return metrics

            with patch.object(optimizer, "_evaluate", side_effect=evaluate):
                optimizer.start({"episodes": 50, "grid_size": 2, "algorithm": "double_qlearning",
                                 "repeats": 1, "validation_seeds": 2, "test_seeds": 5})
                optimizer.thread.join(30)
            result = optimizer.snapshot()["results"]["double_qlearning"]
            self.assertFalse(result["is_default"])
            self.assertGreater(result["validation_gain_pct"], 0)
            self.assertLess(result["test_gain_pct"], 0)


class OptimizationAPI(unittest.TestCase):
    def test_routes_validation_and_existing_app_still_load(self):
        from app.api import create_app
        app = create_app()
        with tempfile.TemporaryDirectory() as folder:
            app.extensions["optimizer"].directory = folder
            client = app.test_client()
            with client.get("/") as response:
                self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/api/config").status_code, 200)
            self.assertEqual(client.get("/api/training/status").status_code, 200)
            with client.get("/optimization-assets/optimization.js") as response:
                self.assertEqual(response.status_code, 200)
            for payload in ([], {"eta_min": 0}, {"gamma_max": 1}):
                self.assertEqual(client.post("/api/optimization/start", json=payload).status_code, 400)
            response = client.post("/api/optimization/start", json={"episodes": 5000})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(client.post("/api/optimization/start", json={}).status_code, 409)
            self.assertEqual(client.get("/api/optimization/report").status_code, 409)
            client.post("/api/optimization/stop", json={})
            app.extensions["optimizer"].thread.join(10)
            download = client.get("/api/optimization/report")
            self.assertEqual(download.status_code, 200)
            self.assertIn("attachment", download.headers["Content-Disposition"])


if __name__ == "__main__":
    unittest.main()
