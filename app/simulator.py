"""
Environnement : simulateur de Load Balancer (sections 2.3 et 6 du document).

Modèle de files d'attente simplifié :
  - chaque serveur a une capacité de traitement et un travail en attente (backlog, en ms) ;
  - une requête affectée à un serveur augmente son backlog ;
  - le temps qui passe fait diminuer le backlog (capacité x durée) ;
  - latence = attente + traitement ; CPU (%) = remplissage de la file ;
  - un serveur saturé (attente > MAX_WAIT) ou en panne rejette la requête.

Le trafic et les pannes utilisent deux générateurs aléatoires distincts, indépendants des actions :
une même graine produit donc exactement le même scénario pour tous les algorithmes.
"""
import math
import random
from . import config as C
from . import mdp


class LoadBalancerEnv:
    def __init__(self, load_factor, seed=0, n_requests=C.N_REQUESTS):
        self.cap = list(C.CAPACITIES)
        self.seed = seed
        self.n_requests = n_requests          # None = flux infini (simulation en direct)
        self.set_load(load_factor)
        self.reset()

    def set_load(self, load_factor):
        self.load_factor = load_factor
        self.mean_ia = C.DEMAND_MEAN / (load_factor * sum(self.cap))   # intervalle moyen (ms)

    # ------------------------------------------------------------------ cycle de vie
    def reset(self):
        self.traffic_rng = random.Random(self.seed * 2 + 1)
        self.fault_rng = random.Random(self.seed * 2 + 2)
        self.backlog = [0.0] * C.N_SERVERS
        self.down = [0] * C.N_SERVERS          # requêtes restantes avant la fin de la panne
        self.t = 0.0                           # date (ms) de la dernière requête observée
        self.i = 0                             # indice de la requête en attente
        self.burst_left = 0
        self.handled = [0] * C.N_SERVERS
        self.failed = [0] * C.N_SERVERS
        self.done = False
        self.pending = None
        self._observed = False
        self._draw_next(0.0)
        return self.observe()

    def _draw_next(self, t_prev):
        rate = 1 + C.BURST_AMPL * math.sin(2 * math.pi * self.i / C.BURST_PERIOD)
        if self.burst_left > 0:
            rate *= C.BURST_FACTOR
            self.burst_left -= 1
        inter = self.traffic_rng.expovariate(rate / self.mean_ia)
        heavy = self.traffic_rng.random() < C.P_HEAVY
        demand = self.traffic_rng.expovariate(1.0 / (C.HEAVY_MEAN if heavy else C.LIGHT_MEAN))
        self.pending = (t_prev + inter, demand)

    # ------------------------------------------------------------------ observation
    def next_arrival_time(self):
        return self.pending[0]

    def observe(self):
        """Fait avancer le temps jusqu'à l'arrivée de la requête en attente et renvoie l'état."""
        if not self._observed:
            t_arr = self.pending[0]
            dt = t_arr - self.t
            for k in range(C.N_SERVERS):
                if self.down[k] == 0:
                    self.backlog[k] = max(0.0, self.backlog[k] - self.cap[k] * dt)
            self.t = t_arr
            for k in range(C.N_SERVERS):       # pannes aléatoires
                if self.down[k] > 0:
                    self.down[k] -= 1
                elif self.fault_rng.random() < C.P_OUTAGE:
                    self.down[k], self.backlog[k] = C.OUTAGE_LEN, 0.0
            self._observed = True
        return self.state()

    def cpu_pct(self, k=None):
        f = lambda i: 100.0 * min(1.0, self.backlog[i] / self.cap[i] / C.Q_REF)
        return f(k) if k is not None else [f(i) for i in range(C.N_SERVERS)]

    def state(self):
        return mdp.encode_state([mdp.cpu_level(c) for c in self.cpu_pct()])

    def valid_actions(self):
        up = [k for k in range(C.N_SERVERS) if self.down[k] == 0]
        return up or list(range(C.N_SERVERS))      # si tout est en panne, aucune action n'est vraiment valide

    # ------------------------------------------------------------------ transition
    def apply(self, a):
        """Traite la requête en attente avec l'action a. Renvoie un dictionnaire d'informations."""
        demand = self.pending[1]
        wait = self.backlog[a] / self.cap[a]
        unavailable = self.down[a] > 0
        saturated = wait > C.MAX_WAIT
        failed = unavailable or saturated
        if failed:
            latency = C.FAIL_LATENCY
        else:
            latency = wait + demand / self.cap[a]
            self.backlog[a] += demand
        cpu_after = self.cpu_pct(a)
        overload = (not failed) and cpu_after > C.OVERLOAD_CPU
        reward, terms = mdp.compute_reward(latency, cpu_after, failed, overload)
        self.handled[a] += 1
        self.failed[a] += int(failed)
        info = {"t": self.t, "request": self.i, "demand": demand, "wait": wait, "latency": latency,
                "failed": failed, "reason": ("panne" if unavailable else "saturation") if failed else None,
                "cpu_after": cpu_after, "overload": overload, "reward": reward, "terms": terms}
        self.i += 1
        self.done = self.n_requests is not None and self.i >= self.n_requests
        if not self.done:
            self._draw_next(self.t)
        self._observed = False
        return info

    def step(self, a):
        """Interface classique de l'apprentissage par renforcement : (s', r, done, info)."""
        info = self.apply(a)
        s2 = self.state() if self.done else self.observe()
        return s2, info["reward"], self.done, info

    # ------------------------------------------------------------------ outils (simulation en direct)
    def snapshot(self, at_time):
        """État des serveurs projeté à la date at_time (pour l'affichage entre deux arrivées)."""
        dt = max(0.0, at_time - self.t)
        out = []
        for k in range(C.N_SERVERS):
            b = self.backlog[k] if self.down[k] else max(0.0, self.backlog[k] - self.cap[k] * dt)
            cpu = 100.0 * min(1.0, b / self.cap[k] / C.Q_REF)
            out.append({"name": C.SERVER_NAMES[k], "capacity": self.cap[k], "up": self.down[k] == 0,
                        "down_left": self.down[k], "backlog_ms": round(b, 1), "wait_ms": round(b / self.cap[k], 1),
                        "cpu_pct": round(cpu, 1), "level": mdp.cpu_level(cpu),
                        "handled": self.handled[k], "failed": self.failed[k]})
        return out

    def inject_fault(self, k):
        self.down[k], self.backlog[k] = C.OUTAGE_LEN, 0.0

    def trigger_burst(self, n_requests=150):
        self.burst_left = n_requests
