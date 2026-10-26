"""Boucle d'un épisode (utilisée par l'entraînement ET par l'évaluation)."""
from . import config as C


def percentile(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[int(round(p / 100.0 * (len(v) - 1)))]


def run_episode(env, agent, eps=0.0, learn=False, on_update=None, episode_no=0):
    """Joue un épisode complet. Si learn=True l'agent met à jour ses tables à chaque requête."""
    s = env.reset()
    agent.reset()
    total, lats, fails, gaps = 0.0, [], 0, []
    share = [0] * C.N_ACTIONS
    step = 0
    while True:
        valid = env.valid_actions()
        a, explored = agent.act(s, valid, eps)
        cpu = env.cpu_pct()
        gaps.append(max(cpu) - min(cpu))                       # écart de charge entre serveurs
        s2, r, done, info = env.step(a)
        if learn:
            log = agent.update(s, a, r, s2, done, [] if done else env.valid_actions())
            if on_update:
                on_update(episode_no, step, s, a, r, s2, explored, info, log)
        total += r
        share[a] += 1
        if info["failed"]:
            fails += 1
        else:
            lats.append(info["latency"])
        s = s2
        step += 1
        if done:
            break
    n = env.n_requests
    return {"reward": total,
            "lat_mean": sum(lats) / len(lats) if lats else float("nan"),
            "lat_p95": percentile(lats, 95),
            "throughput": (n - fails) / (env.t / 1000.0),
            "fail_rate": fails / n,
            "load_gap": sum(gaps) / len(gaps),
            "share": [x / n for x in share]}
