"""Isolated, reproducible hyperparameter searches using the original learning rules.

No dependency beyond the application's standard library and Flask. Search,
training, validation and final-test RNG streams are deliberately separated.
"""
import copy
from datetime import datetime, timezone
import json
import math
import os
import random
import statistics
import threading
import time
import uuid

from . import config as C
from .agents import QLearning, DoubleQLearning
from .episode import run_episode
from .simulator import LoadBalancerEnv

METHODS = {
    "grid": "Grid search",
    "random": "Random search",
    "bayesian": "Optimisation bayésienne",
    "lhs": "Latin hypercube",
    "hyperband": "Hyperband",
}
AGENTS = {"qlearning": QLearning, "double_qlearning": DoubleQLearning}
METRICS = ("score", "latency_ms", "rejection_rate")


def parameters(data):
    if not isinstance(data, dict):
        raise ValueError("Les paramètres doivent être un objet JSON.")
    p = {"method": data.get("method", "grid"), "algorithm": data.get("algorithm", "both")}
    if p["method"] not in METHODS or p["algorithm"] not in (*AGENTS, "both"):
        raise ValueError("Méthode ou agent inconnu.")
    specs = {"episodes": (500, 50, 10000), "trials": (12, 4, 64),
             "grid_size": (4, 2, 8), "repeats": (2, 1, 5),
             "validation_seeds": (5, 2, 20), "test_seeds": (10, 5, 50),
             "seed": (42, 0, 1000000)}
    for key, (default, lo, hi) in specs.items():
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not lo <= value <= hi:
            raise ValueError(f"{key} doit être un entier entre {lo} et {hi}.")
        p[key] = int(value)
    for key, default in (("eta_min", .01), ("eta_max", .5), ("gamma_min", 0.), ("gamma_max", .99)):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} doit être un nombre fini.")
        p[key] = float(value)
    if not 0 < p["eta_min"] < p["eta_max"] <= 1:
        raise ValueError("Choisir 0 < η min < η max ≤ 1.")
    if not 0 <= p["gamma_min"] < p["gamma_max"] < 1:
        raise ValueError("Choisir 0 ≤ γ min < γ max < 1.")
    if p["method"] == "bayesian" and p["trials"] < 6:
        raise ValueError("La recherche bayésienne demande au moins 6 paires, dont 5 essais initiaux.")
    if p["method"] == "hyperband" and p["episodes"] < 150:
        raise ValueError("Hyperband demande au moins 150 épisodes pour comparer plusieurs budgets.")
    return p


def decode_point(point, p):
    """Logarithmic learning-rate axis, linear discount axis."""
    u, v = point
    return {"eta": math.exp(math.log(p["eta_min"]) + u * math.log(p["eta_max"] / p["eta_min"])),
            "gamma": p["gamma_min"] + v * (p["gamma_max"] - p["gamma_min"])}


def latin_points(n, rng):
    axes = []
    for _ in range(2):
        axis = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(axis)
        axes.append(axis)
    return list(zip(*axes))


def cholesky(matrix):
    n = len(matrix)
    out = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j] - sum(out[i][k] * out[j][k] for k in range(j))
            out[i][j] = math.sqrt(max(s, 1e-12)) if i == j else s / out[j][j]
    return out


def forward(lower, b):
    x = []
    for i, row in enumerate(lower):
        x.append((b[i] - sum(row[j] * x[j] for j in range(i))) / row[i])
    return x


def solve_cholesky(lower, b):
    y = forward(lower, b)
    x = [0.] * len(y)
    for i in reversed(range(len(y))):
        x[i] = (y[i] - sum(lower[j][i] * x[j] for j in range(i + 1, len(y)))) / lower[i][i]
    return x


def bayesian_point(observations, rng):
    """Gaussian-process surrogate + expected improvement, noisy observations.

    Inputs are in [0,1]^2; outputs and observation variances are standardized.
    RBF length scale .25, unit prior variance, EI exploration offset .01.
    A finite random acquisition pool keeps this two-dimensional search light.
    """
    if len(observations) < 5:
        return (rng.random(), rng.random())
    values = [x[1] for x in observations]
    mean = statistics.mean(values)
    scale = max(statistics.pstdev(values), .01)
    targets = [(x - mean) / scale for x in values]
    kernel = lambda a, b: math.exp(-sum((x-y)**2 for x, y in zip(a, b)) / (2 * .25**2))
    matrix = [[kernel(a[0], b[0]) for b in observations] for a in observations]
    for i, (_, _, variance) in enumerate(observations):
        matrix[i][i] += max(variance / scale**2, .0025)
    lower = cholesky(matrix)
    weights = solve_cholesky(lower, targets)
    incumbent = max(targets)

    def acquisition(point):
        k = [kernel(point, obs[0]) for obs in observations]
        mu = sum(a*b for a, b in zip(k, weights))
        v = forward(lower, k)
        sigma = math.sqrt(max(1. - sum(x*x for x in v), 1e-12))
        improvement = mu - incumbent - .01
        z = improvement / sigma
        return improvement * (.5 * (1 + math.erf(z / math.sqrt(2)))) + sigma * math.exp(-.5*z*z) / math.sqrt(2*math.pi)

    pool = [(rng.random(), rng.random()) for _ in range(384)]
    pool = [x for x in pool if all(sum((a-b)**2 for a, b in zip(x, obs[0])) > 1e-8 for obs in observations)]
    return max(pool, key=acquisition)


def hyperband_brackets(max_episodes, candidates, factor=3):
    """Finite-budget Hyperband: each bracket is successive halving.

    Limit to three brackets, with minimum initial budget about max/9.
    Scale canonical Hyperband candidate weights 3^s/(s+1) to the user's
    finite candidate cap, reserving at least one candidate per bracket.
    """
    smax = min(2, int(math.log(max(1, max_episodes // 50), factor)), int(math.log(candidates, factor)))
    brackets = []
    remaining = candidates
    for s in reversed(range(smax + 1)):
        weight = factor**s / (s+1)
        n = remaining if s == 0 else max(1, round(candidates * weight / sum(factor**j/(j+1) for j in range(smax + 1))))
        n = min(n, remaining - s)
        remaining -= n
        budgets = sorted(set([max(1, round(max_episodes / factor**j)) for j in reversed(range(s+1))]))
        brackets.append((n, budgets))
    return brackets


class Cancelled(Exception):
    pass


class Optimizer:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.state = {"status": "idle", "running": False, "trials": [], "results": {}, "history": [], "progress": 0.}
        self.started = None
        self._artifacts = {}
        self._load()

    def _load(self):
        try:
            with open(os.path.join(self.directory, "last.json"), encoding="utf-8") as f:
                report = json.load(f)
            self.state = report["search"]
            self._artifacts = report.get("models", {})
            self.state["running"] = False
            if self.state["status"] == "running":
                self.state["status"] = "interrupted"
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        if state["running"] and self.started is not None:
            state["elapsed_s"] = round(time.monotonic() - self.started, 1)
        return state

    def start(self, data):
        p = parameters(data)
        with self.lock:
            if self.state["running"]:
                raise RuntimeError("Une optimisation est déjà en cours.")
            self.cancel.clear()
            self._artifacts = {}
            self.started = time.monotonic()
            self.state = {"id": uuid.uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
                          "status": "running", "running": True, "params": p, "trials": [], "history": [],
                          "results": {}, "progress": 0., "work_done": 0, "work_total": self._work_total(p),
                          "elapsed_s": 0., "current": {}, "message": "Préparation des entraînements…"}
            self.thread = threading.Thread(target=self._run, args=(p,), daemon=True)
            self.thread.start()

    @staticmethod
    def _work_total(p):
        ev = 3 * p["validation_seeds"] * p["repeats"]
        base = p["episodes"] * p["repeats"] + ev
        if p["method"] == "hyperband":
            search = 0
            for n, budgets in hyperband_brackets(p["episodes"], p["trials"]):
                previous = 0
                for b in budgets:
                    search += n * ((b-previous)*p["repeats"] + ev)
                    previous = b
                    n = max(1, n // 3)
        else:
            count = p["grid_size"]**2 if p["method"] == "grid" else p["trials"]
            search = count * base
        final = 2 * 3 * p["test_seeds"] * p["repeats"]
        return (base + search + final) * (2 if p["algorithm"] == "both" else 1)

    def stop(self):
        with self.lock:
            if self.state["running"]:
                self.cancel.set()
                self.state["message"] = "Arrêt demandé : conservation des essais terminés…"

    def _check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def _advance(self, **current):
        self._check()
        with self.lock:
            self.state["work_done"] += 1
            self.state["progress"] = min(.999, self.state["work_done"] / self.state["work_total"])
            self.state["current"].update(current)

    def _new_candidate(self, algorithm, point, p, label="candidate", bracket=None):
        pair = {"eta": C.LR, "gamma": C.GAMMA} if label == "baseline" else decode_point(point, p)
        agents = [AGENTS[algorithm](lr=pair["eta"], gamma=pair["gamma"], seed=p["seed"] * 1000 + r * 17 + 11)
                  for r in range(p["repeats"])]
        return {"algorithm": algorithm, "point": point, **pair, "agents": agents, "episodes": 0,
                "label": label, "bracket": bracket, "history": {}, "record": None}

    def _train(self, candidate, target, p):
        old = candidate["episodes"]
        for repeat, agent in enumerate(candidate["agents"]):
            # Independent repeats, but identical exogenous scenarios across all pairs.
            for ep in range(old, target):
                self._check()
                stream = p["seed"] * 10000000 + 100000 + repeat * 20000 + ep
                rng = random.Random(stream + 100000000)
                load = rng.uniform(*C.TRAIN_LOAD_RANGE)
                # Always use the FULL budget in epsilon decay, including in Hyperband.
                frac = min(1., ep / (C.EPS_DECAY_FRACTION * p["episodes"]))
                eps = C.EPS_START - (C.EPS_START-C.EPS_END) * frac
                result = run_episode(LoadBalancerEnv(load, seed=stream), agent, eps=eps, learn=True)
                key = ep // 25
                total, count = candidate["history"].get(key, (0., 0))
                candidate["history"][key] = (total + result["reward"]/C.N_REQUESTS, count+1)
                self._advance(phase="training", algorithm=candidate["algorithm"], eta=candidate["eta"],
                              gamma=candidate["gamma"], episode=ep+1, target=target, repeat=repeat+1,
                              label=candidate["label"], epsilon=eps)
        candidate["episodes"] = target

    def _evaluate(self, candidate, p, split="validation"):
        self._check()
        count = p["validation_seeds"] if split == "validation" else p["test_seeds"]
        # All pairs and both algorithms get common validation/final traffic streams.
        # Final streams are disjoint from validation and never used by the search.
        offset = 1000000 if split == "validation" else 2000000
        runs = []
        per_scenario = {name: [] for name in C.TEST_SCENARIOS}
        for repeat, source in enumerate(candidate["agents"]):
            scores, latencies, failures = [], [], []
            for name, load in C.TEST_SCENARIOS.items():
                for seed in range(count):
                    self._check()
                    agent = AGENTS[candidate["algorithm"]](lr=candidate["eta"], gamma=candidate["gamma"],
                                                             seed=p["seed"]*1000 + repeat*17 + seed + offset)
                    agent.load_dict(copy.deepcopy(source.to_dict()))
                    res = run_episode(LoadBalancerEnv(load, seed=p["seed"]*10000000 + offset + seed), agent, eps=0., learn=False)
                    row = {"score": res["reward"]/C.N_REQUESTS, "latency_ms": res["lat_mean"], "rejection_rate": res["fail_rate"]}
                    per_scenario[name].append(row)
                    scores.append(row["score"])
                    if math.isfinite(row["latency_ms"]):
                        latencies.append(row["latency_ms"])
                    failures.append(row["rejection_rate"])
                    self._advance(phase=split, algorithm=candidate["algorithm"], eta=candidate["eta"], gamma=candidate["gamma"])
            runs.append({"score": statistics.mean(scores), "latency_ms": statistics.mean(latencies) if latencies else None,
                         "rejection_rate": statistics.mean(failures)})

        def aggregate(rows):
            summary = {}
            for key in METRICS:
                values = [row[key] for row in rows if row[key] is not None and math.isfinite(row[key])]
                summary[key] = statistics.mean(values) if values else None
                summary[key+"_std"] = statistics.stdev(values) if len(values) > 1 else 0.
            return summary
        return {**aggregate(runs), "repeats": p["repeats"], "seeds_per_scenario": count,
                "scenarios": {name: aggregate(rows) for name, rows in per_scenario.items()}}

    def _assess(self, candidate, target, p):
        self._train(candidate, target, p)
        metrics = self._evaluate(candidate, p)
        with self.lock:
            record = candidate["record"]
            if record is None:
                record = {"id": len(self.state["trials"])+1, "algorithm": candidate["algorithm"],
                          "eta": candidate["eta"], "gamma": candidate["gamma"], "label": candidate["label"],
                          "bracket": candidate["bracket"], "checkpoints": []}
                candidate["record"] = record
                self.state["trials"].append(record)
            record.update(episodes=target, metrics=metrics,
                          status="complete" if target == p["episodes"] else "checkpoint")
            record["checkpoints"].append({"episodes": target, "score": metrics["score"]})
            self.state["history"].append({"trial": record["id"], "algorithm": candidate["algorithm"],
                                          "episodes": target, "score": metrics["score"]})
        return metrics

    def _search(self, algorithm, p, rng):
        completed = []
        if p["method"] == "hyperband":
            for bracket, (n, budgets) in enumerate(hyperband_brackets(p["episodes"], p["trials"])):
                candidates = [self._new_candidate(algorithm, (rng.random(), rng.random()), p, bracket=bracket+1) for _ in range(n)]
                for target in budgets:
                    for candidate in candidates:
                        self._assess(candidate, target, p)
                    if target == p["episodes"]:
                        completed.extend(candidates)
                    else:
                        candidates.sort(key=lambda c: c["record"]["metrics"]["score"], reverse=True)
                        keep = max(1, len(candidates)//3)
                        with self.lock:
                            for candidate in candidates[keep:]:
                                candidate["record"]["status"] = "pruned"
                        candidates = candidates[:keep]
            return completed

        n = p["grid_size"]**2 if p["method"] == "grid" else p["trials"]
        if p["method"] == "grid":
            axis = [i/(p["grid_size"]-1) for i in range(p["grid_size"])]
            points = [(u, v) for u in axis for v in axis]
        elif p["method"] == "lhs":
            points = latin_points(n, rng)
        else:
            points = [(rng.random(), rng.random()) for _ in range(n)]
        observations = []
        for index in range(n):
            self._check()
            point = bayesian_point(observations, rng) if p["method"] == "bayesian" else points[index]
            candidate = self._new_candidate(algorithm, point, p)
            metrics = self._assess(candidate, p["episodes"], p)
            observations.append((point, metrics["score"], metrics["score_std"]**2 / p["repeats"]))
            completed.append(candidate)
        return completed

    def _run(self, p):
        try:
            names = list(AGENTS) if p["algorithm"] == "both" else [p["algorithm"]]
            for algorithm in names:
                self._check()
                baseline = self._new_candidate(algorithm, None, p, "baseline")
                self._assess(baseline, p["episodes"], p)
                # Same initial search design across algorithms; Bayesian adapts independently.
                candidates = self._search(algorithm, p, random.Random(p["seed"]))
                winner = max([baseline, *candidates], key=lambda c: c["record"]["metrics"]["score"])
                with self.lock:
                    self.state["message"] = "Confirmation sur de nouveaux scénarios de test…"
                best_test = self._evaluate(winner, p, "test")
                base_test = best_test if winner is baseline else self._evaluate(baseline, p, "test")
                # Credit the cached test when the baseline wins, for an accurate progress bar.
                if winner is baseline:
                    with self.lock:
                        self.state["work_done"] += 3*p["test_seeds"]*p["repeats"]
                base_score = baseline["record"]["metrics"]["score"]
                result = {"eta": winner["eta"], "gamma": winner["gamma"], "trial_id": winner["record"]["id"],
                          "is_default": winner is baseline, "validation": winner["record"]["metrics"],
                          "baseline_validation": baseline["record"]["metrics"], "test": best_test,
                          "baseline_test": base_test,
                          "validation_gain_pct": 100*(winner["record"]["metrics"]["score"]-base_score)/max(abs(base_score), 1e-12),
                          "test_gain_pct": 100*(best_test["score"]-base_test["score"])/max(abs(base_test["score"]), 1e-12),
                          "learning_curve": self._curve(winner), "baseline_curve": self._curve(baseline)}
                with self.lock:
                    self.state["results"][algorithm] = result
                    self._artifacts[algorithm] = {"eta": winner["eta"], "gamma": winner["gamma"],
                                                  "episodes": p["episodes"],
                                                  "replicates": [copy.deepcopy(a.to_dict()) for a in winner["agents"]]}
            with self.lock:
                self.state.update(status="completed", progress=1., message="Recherche terminée. Paramètres choisis sur validation, puis testés séparément.")
        except Cancelled:
            with self.lock:
                self.state.update(status="stopped", message="Recherche arrêtée. Les essais évalués restent consultables ; les résultats incomplets ne sont pas confirmés.")
        except Exception as error:
            with self.lock:
                self.state.update(status="error", message=f"Échec de la recherche : {error}")
        finally:
            with self.lock:
                self.state.update(elapsed_s=round(time.monotonic()-self.started, 1), current={})
                try:
                    # Persist before releasing the single-search guard.
                    self._save()
                except OSError as error:
                    self.state["message"] += f" Sauvegarde indisponible : {error}"
                self.state["running"] = False

    @staticmethod
    def _curve(candidate):
        return [{"episode": min((block+1)*25, candidate["episodes"]), "score": total/count}
                for block, (total, count) in sorted(candidate["history"].items())]

    def report(self):
        with self.lock:
            state = copy.deepcopy(self.state)
            state["running"] = False
            return {"schema_version": 1, "search": state,
                    "protocol": {"objective": "mean undiscounted reward per request; maximize",
                                 "scenario_weights": {k: 1/3 for k in C.TEST_SCENARIOS},
                                 "selection": "validation only; default retained if no candidate improves it",
                                 "variation": "sample standard deviation across independent training replicates",
                                 "eta_sampling": "logarithmic", "gamma_sampling": "linear",
                                 "training": "original QLearning/DoubleQLearning update; 200 requests per episode",
                                 "epsilon": "1.0 to 0.05 over 80% of full episode budget; unchanged across Hyperband rungs",
                                 "test": "fresh exogenous seeds; epsilon=0; no learning; no reselection after final test",
                                 "bayesian": "Gaussian process / RBF(.25) / expected improvement; 5 startup trials; 384 acquisition candidates"},
                    "models": copy.deepcopy(self._artifacts)}

    def _save(self):
        os.makedirs(self.directory, exist_ok=True)
        report = self.report()
        for name in (self.state["id"]+".json", "last.json"):
            path = os.path.join(self.directory, name)
            with open(path+".tmp", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, allow_nan=False)
            os.replace(path+".tmp", path)
