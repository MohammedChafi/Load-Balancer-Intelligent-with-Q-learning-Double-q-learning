"""
Phase de test (section 7) : mêmes scénarios de trafic et mêmes graines pour Round Robin,
Q-Learning et Double Q-Learning, sans exploration (ε = 0) et sans apprentissage.
"""
import math
from . import config as C
from .agents import RoundRobin
from .episode import run_episode
from .simulator import LoadBalancerEnv

METRICS = ["lat_mean", "lat_p95", "throughput", "fail_rate", "load_gap", "reward"]


def _mean_std(v):
    v = [x for x in v if not math.isnan(x)]
    m = sum(v) / len(v)
    return {"mean": m, "std": math.sqrt(sum((x - m) ** 2 for x in v) / len(v))}


def evaluate_all(trained, n_seeds=C.N_TEST_SEEDS):
    """trained : dict {'qlearning': agent, 'double_qlearning': agent}."""
    agents = {"round_robin": RoundRobin(), **trained}
    out = {"n_seeds": n_seeds, "n_requests": C.N_REQUESTS, "scenarios": {}}
    for name, lf in C.TEST_SCENARIOS.items():
        res = {}
        for key, ag in agents.items():
            runs = [run_episode(LoadBalancerEnv(lf, seed=C.TEST_SEED_BASE + s), ag, eps=0.0, learn=False)
                    for s in range(n_seeds)]
            res[key] = {"label": ag.label, **{m: _mean_std([r[m] for r in runs]) for m in METRICS},
                        "share": [sum(r["share"][k] for r in runs) / n_seeds for k in range(C.N_ACTIONS)]}
        out["scenarios"][name] = {"load": lf, "results": res}
    return out
