"""
Phase d'entraînement (section 3 et 4) : Q-Learning et Double Q-Learning apprennent en parallèle
sur exactement le même trafic simulé. L'entraînement tourne dans un thread ; tout ce qui se passe
(courbes, tables Q, détail de chaque mise à jour) est conservé pour être exposé par l'API.
"""
import json
import os
import threading
import time
from collections import deque

from . import config as C
from .agents import QLearning, DoubleQLearning
from .episode import run_episode
from .simulator import LoadBalancerEnv
import random

NAMES = ["qlearning", "double_qlearning"]


class Trainer:
    def __init__(self, models_path):
        self.models_path = models_path
        self.lock = threading.RLock()
        self.thread = None
        self.stop_flag = False
        self.running = False
        self.finished = False
        self.agents = None
        self.params = None
        self.eval_result = None
        self._reset()

    def _reset(self):
        self.episode = 0
        self.total = 0
        self.epsilon = C.EPS_START
        self.started_at = None
        self.elapsed = 0.0
        self.hist = {"epsilon": [], "load": [],
                     **{n: {"reward": [], "lat_mean": [], "fail_rate": []} for n in NAMES}}
        self.updates = {n: deque(maxlen=500) for n in NAMES}
        self.update_count = {n: 0 for n in NAMES}
        self.snapshots = []

    # ------------------------------------------------------------------ démarrage / arrêt
    def start(self, p):
        with self.lock:
            if self.running:
                raise RuntimeError("Un entraînement est déjà en cours.")
            self.params = {
                "episodes": int(p.get("episodes", C.N_EPISODES)),
                "lr": float(p.get("lr", C.LR)),
                "gamma": float(p.get("gamma", C.GAMMA)),
                "eps_start": float(p.get("eps_start", C.EPS_START)),
                "eps_end": float(p.get("eps_end", C.EPS_END)),
                "seed": int(p.get("seed", 0)),
                "delay_ms": float(p.get("delay_ms", 0)),     # ralentit l'entraînement pour le regarder
            }
            q = self.params
            if not (50 <= q["episodes"] <= 100000 and 0 < q["lr"] <= 1 and 0 <= q["gamma"] < 1
                    and 0 <= q["eps_end"] <= q["eps_start"] <= 1):
                raise ValueError("Paramètres invalides.")
            self._reset()
            self.total = q["episodes"]
            self.epsilon = q["eps_start"]
            s = q["seed"]
            self.agents = {"qlearning": QLearning(q["lr"], q["gamma"], s + 11),
                           "double_qlearning": DoubleQLearning(q["lr"], q["gamma"], s + 12)}
            self.eval_result = None
            self.stop_flag, self.running, self.finished = False, True, False
            self._snapshot(0)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_flag = True

    # ------------------------------------------------------------------ boucle d'entraînement
    def _callback(self, name):
        def cb(ep, step, s, a, r, s2, explored, info, log):
            self.update_count[name] += 1
            self.updates[name].append((self.update_count[name], ep, step, s, a, r, s2, explored,
                                       self.epsilon, info, log))
        return cb

    def _run(self):
        p = self.params
        rng = random.Random(p["seed"])
        n = p["episodes"]
        checkpoints = {round(k * n / 20) for k in range(1, 21)}
        cbs = {name: self._callback(name) for name in NAMES}
        self.started_at = time.time()
        try:
            for ep in range(n):
                if self.stop_flag:
                    break
                frac = min(1.0, ep / (C.EPS_DECAY_FRACTION * n))
                eps = p["eps_start"] - (p["eps_start"] - p["eps_end"]) * frac
                load = rng.uniform(*C.TRAIN_LOAD_RANGE)           # niveau de trafic de l'épisode
                with self.lock:
                    self.epsilon = eps
                    for name in NAMES:
                        env = LoadBalancerEnv(load, seed=10_000 + ep)   # même trafic pour les deux agents
                        res = run_episode(env, self.agents[name], eps=eps, learn=True,
                                          on_update=cbs[name], episode_no=ep + 1)
                        h = self.hist[name]
                        h["reward"].append(res["reward"])
                        h["lat_mean"].append(res["lat_mean"])
                        h["fail_rate"].append(res["fail_rate"])
                    self.hist["epsilon"].append(eps)
                    self.hist["load"].append(load)
                    self.episode = ep + 1
                    if self.episode in checkpoints:
                        self._snapshot(self.episode)
                self.elapsed = time.time() - self.started_at
                if p["delay_ms"] > 0:
                    time.sleep(p["delay_ms"] / 1000.0)
        finally:
            self.elapsed = time.time() - self.started_at
            self.running = False
            self.finished = self.episode > 0
            if self.finished:
                self.save()

    # ------------------------------------------------------------------ consultation (API)
    def _snapshot(self, episode):
        self.snapshots.append({"episode": episode, "epsilon": self.epsilon,
                               **{name: self._copy_tables(name) for name in NAMES}})

    def _copy_tables(self, name):
        return json.loads(json.dumps(self.agents[name].tables()))

    @property
    def model_available(self):
        return self.agents is not None and self.episode > 0

    def tables(self):
        with self.lock:
            return {"episode": self.episode, **{name: self._copy_tables(name) for name in NAMES}}

    def get_updates(self, name, since=0, limit=50):
        with self.lock:
            rows = [u for u in self.updates[name] if u[0] > since][-limit:]
        return [{"id": u[0], "episode": u[1], "step": u[2], "state": u[3], "action": u[4], "reward": u[5],
                 "next_state": u[6], "explored": u[7], "epsilon": u[8], "failed": u[9]["failed"],
                 "latency": u[9]["latency"], "terms": u[9]["terms"], "update": u[10]} for u in rows]

    def history(self, since=0):
        with self.lock:
            return {"since": since, "episode": self.episode, "epsilon": self.hist["epsilon"][since:],
                    "load": self.hist["load"][since:],
                    **{n: {k: v[since:] for k, v in self.hist[n].items()} for n in NAMES}}

    def status(self):
        with self.lock:
            def recent(name):
                h = self.hist[name]
                last = lambda v: (sum(v[-100:]) / len(v[-100:])) if v else None
                return {"reward_avg100": last(h["reward"]), "lat_mean_avg100": last(h["lat_mean"]),
                        "fail_rate_avg100": last(h["fail_rate"]), "updates": self.update_count[name]}
            return {"running": self.running, "finished": self.finished, "model_available": self.model_available,
                    "episode": self.episode, "total": self.total, "epsilon": self.epsilon,
                    "progress": (self.episode / self.total) if self.total else 0.0,
                    "elapsed_s": round(self.elapsed, 1), "params": self.params,
                    "steps_per_agent": self.episode * C.N_REQUESTS,
                    "agents": {n: recent(n) for n in NAMES}, "snapshots": [s["episode"] for s in self.snapshots]}

    # ------------------------------------------------------------------ sauvegarde
    def save(self):
        with open(self.models_path, "w") as f:
            json.dump({"params": self.params, "episode": self.episode,
                       **{n: self.agents[n].to_dict() for n in NAMES}}, f)

    def load_model(self):
        if not os.path.exists(self.models_path):
            return False
        with open(self.models_path) as f:
            d = json.load(f)
        p = d.get("params") or {}
        self.params = p
        self.agents = {"qlearning": QLearning(p.get("lr", C.LR), p.get("gamma", C.GAMMA)),
                       "double_qlearning": DoubleQLearning(p.get("lr", C.LR), p.get("gamma", C.GAMMA))}
        for n in NAMES:
            self.agents[n].load_dict(d[n])
        self.episode = self.total = d.get("episode", 0)
        self.finished = True
        return True
