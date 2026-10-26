"""
Simulation en direct : le front demande "fais avancer le temps de dt ms" et c'est CE programme Python
qui reçoit chaque requête, lit l'état, choisit le serveur (Round Robin, Q-Learning ou Double Q-Learning,
sans exploration) puis applique la décision dans le simulateur.
"""
import uuid
from collections import deque

from . import config as C
from . import mdp
from .simulator import LoadBalancerEnv
from .episode import percentile


class LiveSession:
    def __init__(self, agent, load, seed=None):
        self.id = uuid.uuid4().hex[:8]
        self.agent = agent
        self.agent.reset()
        self.env = LoadBalancerEnv(load, seed=seed if seed is not None else int(uuid.uuid4().int % 10**6),
                                   n_requests=None)
        self.clock = 0.0
        self.n = self.failed = 0
        self.reward_sum = 0.0
        self.lats = deque(maxlen=500)
        self.gaps = deque(maxlen=500)

    def tick(self, dt_ms):
        dt = max(0.0, min(float(dt_ms), 3000.0))
        target = self.clock + dt
        events = []
        while self.env.next_arrival_time() <= target and len(events) < 400:
            s = self.env.observe()                                    # 1. observer l'état
            valid = self.env.valid_actions()
            cpu_before = [round(c, 1) for c in self.env.cpu_pct()]
            a, _ = self.agent.act(s, valid, 0.0)                      # 2. décider (Python)
            qv = self.agent.q_values(s)
            info = self.env.apply(a)                                  # 3. appliquer, 4. récompense
            self.n += 1
            self.reward_sum += info["reward"]
            self.failed += int(info["failed"])
            if not info["failed"]:
                self.lats.append(info["latency"])
            self.gaps.append(max(cpu_before) - min(cpu_before))
            events.append({"id": self.n, "t": round(info["t"], 1), "demand_ms": round(info["demand"], 1),
                           "state": s, "levels": mdp.decode_state(s), "cpu_before": cpu_before, "valid": valid,
                           "q_values": [round(x, 3) for x in qv] if qv else None, "action": a,
                           "server": C.SERVER_NAMES[a], "wait_ms": round(info["wait"], 1),
                           "latency_ms": round(info["latency"], 1), "failed": info["failed"],
                           "reason": info["reason"], "cpu_after": round(info["cpu_after"], 1),
                           "reward": round(info["reward"], 3),
                           "terms": {k: round(v, 3) for k, v in info["terms"].items()}})
        self.clock = events[-1]["t"] if len(events) >= 400 else target
        return {"session": self.id, "algorithm": self.agent.name, "clock_ms": round(self.clock, 1),
                "events": events, "servers": self.env.snapshot(self.clock), "metrics": self.metrics(),
                "load": self.env.load_factor}

    def metrics(self):
        lat = list(self.lats)
        return {"requests": self.n, "failed": self.failed,
                "fail_rate": self.failed / self.n if self.n else 0.0,
                "lat_mean": sum(lat) / len(lat) if lat else None, "lat_p95": percentile(lat, 95) if lat else None,
                "throughput": (self.n - self.failed) / (self.clock / 1000.0) if self.clock else 0.0,
                "load_gap": sum(self.gaps) / len(self.gaps) if self.gaps else 0.0,
                "reward_sum": self.reward_sum}
