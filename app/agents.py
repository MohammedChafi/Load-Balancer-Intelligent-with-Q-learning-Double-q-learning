"""
Agents : Round Robin (référence), Q-Learning et Double Q-Learning (sections 3 et 4 du document).
Tables tabulaires 27 états x 3 actions, sans réseau de neurones.
Les actions interdites (serveur en panne) sont exclues du choix et du max sur l'état suivant.
"""
import random
from . import config as C


class RoundRobin:
    name = "round_robin"
    label = "Round Robin"

    def __init__(self):
        self.k = -1

    def reset(self):
        self.k = -1

    def q_values(self, s):
        return None

    def act(self, s, valid, eps=0.0):
        for _ in range(C.N_ACTIONS):                  # serveurs à tour de rôle, en sautant ceux en panne
            self.k = (self.k + 1) % C.N_ACTIONS
            if self.k in valid:
                break
        return self.k, False


class QLearning:
    name = "qlearning"
    label = "Q-Learning"

    def __init__(self, lr=C.LR, gamma=C.GAMMA, seed=0):
        self.lr, self.gamma = lr, gamma
        self.rng = random.Random(seed)
        self.Q = [[0.0] * C.N_ACTIONS for _ in range(C.N_STATES)]
        self.N = [[0] * C.N_ACTIONS for _ in range(C.N_STATES)]      # nombre de visites de (s, a)

    def reset(self):
        pass

    def q_values(self, s):
        return list(self.Q[s])

    def _greedy(self, values, valid):
        best = max(values[k] for k in valid)
        return self.rng.choice([k for k in valid if values[k] == best])

    def act(self, s, valid, eps=0.0):
        """ε-greedy : exploration avec probabilité ε, sinon meilleure valeur connue. Renvoie (action, exploré)."""
        if self.rng.random() < eps:
            return self.rng.choice(valid), True
        return self._greedy(self.q_values(s), valid), False

    def update(self, s, a, r, s2, done, valid_next):
        """Q(s,a) <- Q(s,a) + η [ r + γ max_a' Q(s',a') - Q(s,a) ]"""
        q_old = self.Q[s][a]
        future = 0.0 if done else max(self.Q[s2][k] for k in valid_next)
        target = r + self.gamma * future
        td = target - q_old
        self.Q[s][a] = q_old + self.lr * td
        self.N[s][a] += 1
        return {"table": "Q", "q_old": q_old, "future": future, "gamma_future": self.gamma * future,
                "target": target, "td_error": td, "q_new": self.Q[s][a]}

    def tables(self):
        return {"Q": self.Q, "N": self.N}

    def to_dict(self):
        return {"Q": self.Q, "N": self.N}

    def load_dict(self, d):
        self.Q, self.N = d["Q"], d["N"]


class DoubleQLearning(QLearning):
    name = "double_qlearning"
    label = "Double Q-Learning"

    def __init__(self, lr=C.LR, gamma=C.GAMMA, seed=0):
        super().__init__(lr, gamma, seed)
        self.QA = [[0.0] * C.N_ACTIONS for _ in range(C.N_STATES)]
        self.QB = [[0.0] * C.N_ACTIONS for _ in range(C.N_STATES)]

    def q_values(self, s):
        return [(a + b) / 2 for a, b in zip(self.QA[s], self.QB[s])]    # décision : moyenne des deux tables

    def update(self, s, a, r, s2, done, valid_next):
        """On met à jour QA ou QB (50 %) : la table mise à jour CHOISIT la meilleure action suivante,
        l'autre table l'ÉVALUE :  QA(s,a) += η [ r + γ QB(s', argmax QA(s',.)) - QA(s,a) ]"""
        if self.rng.random() < 0.5:
            X, Y, name = self.QA, self.QB, "QA"
        else:
            X, Y, name = self.QB, self.QA, "QB"
        q_old = X[s][a]
        if done:
            best, future = None, 0.0
        else:
            best = self._greedy(X[s2], valid_next)          # sélection par la table mise à jour
            future = Y[s2][best]                             # évaluation par l'autre table
        target = r + self.gamma * future
        td = target - q_old
        X[s][a] = q_old + self.lr * td
        self.N[s][a] += 1
        return {"table": name, "select_action": best, "future": future, "gamma_future": self.gamma * future,
                "q_old": q_old, "target": target, "td_error": td, "q_new": X[s][a]}

    def tables(self):
        avg = [[(x + y) / 2 for x, y in zip(ra, rb)] for ra, rb in zip(self.QA, self.QB)]
        return {"Q": avg, "QA": self.QA, "QB": self.QB, "N": self.N}

    def to_dict(self):
        return {"QA": self.QA, "QB": self.QB, "N": self.N}

    def load_dict(self, d):
        self.QA, self.QB, self.N = d["QA"], d["QB"], d["N"]
