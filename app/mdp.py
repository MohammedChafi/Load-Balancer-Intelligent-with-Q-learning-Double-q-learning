"""
Le MDP : états S, actions A, récompense R, facteur gamma (section 2 du document).
Les transitions P sont dans simulator.py (on ne les calcule pas, on les simule).
"""
from . import config as C


def cpu_level(cpu_pct):
    """0 = faible (<40 %), 1 = moyen (40-70 %), 2 = élevé (>70 %)."""
    if cpu_pct < C.CPU_LOW:
        return 0
    return 1 if cpu_pct <= C.CPU_HIGH else 2


def encode_state(levels):
    """(l_A, l_B, l_C) -> entier 0..26."""
    s = 0
    for lv in levels:
        s = s * C.N_LEVELS + lv
    return s


def decode_state(s):
    return [s // 9, (s // 3) % 3, s % 3]


def compute_reward(latency_ms, cpu_after_pct, failed, overload):
    """R = -(alpha*T/Tref + beta*L/100 + delta*F + zeta*surcharge). Renvoie (R, détail des termes)."""
    terms = {
        "latence": C.ALPHA * latency_ms / C.T_REF,
        "cpu": C.BETA * cpu_after_pct / 100.0,
        "echec": C.DELTA * (1.0 if failed else 0.0),
        "surcharge": C.ZETA * (1.0 if overload else 0.0),
    }
    return -sum(terms.values()), terms


def describe():
    """Description complète du MDP, exposée par l'API."""
    return {
        "states": [{"index": s, "levels": decode_state(s),
                    "label": " ".join(f"{n}:{C.LEVEL_NAMES[l]}" for n, l in zip(C.SERVER_NAMES, decode_state(s)))}
                   for s in range(C.N_STATES)],
        "cpu_thresholds": {"faible": f"< {C.CPU_LOW:.0f} %", "moyen": f"{C.CPU_LOW:.0f} % à {C.CPU_HIGH:.0f} %",
                           "élevé": f"> {C.CPU_HIGH:.0f} %"},
        "actions": [{"index": i, "name": f"envoyer au serveur {n}", "server": n} for i, n in enumerate(C.SERVER_NAMES)],
        "action_masking": "Une action est interdite si le serveur est en panne (section 2.2).",
        "transitions": ("Non calculées à l'avance : le simulateur tire les arrivées, les durées de traitement, "
                        "les pannes et fait évoluer les files d'attente (sections 2.3 et 6.2)."),
        "reward": {
            "formula": "R = -( alpha*T/Tref + beta*L/100 + delta*F + zeta*Surcharge )",
            "T": "temps de réponse de la requête (ms) = attente + traitement ; 500 ms si rejetée",
            "L": "CPU (%) du serveur choisi après affectation",
            "F": "1 si la requête est rejetée (serveur saturé ou en panne), sinon 0",
            "Surcharge": f"1 si le CPU du serveur choisi dépasse {C.OVERLOAD_CPU:.0f} % après affectation",
            "weights": {"alpha": C.ALPHA, "beta": C.BETA, "delta": C.DELTA, "zeta": C.ZETA},
            "Tref_ms": C.T_REF,
        },
        "gamma": C.GAMMA,
        "learning": {"eta": C.LR, "eps_start": C.EPS_START, "eps_end": C.EPS_END,
                     "eps_decay_fraction": C.EPS_DECAY_FRACTION, "episodes": C.N_EPISODES,
                     "requests_per_episode": C.N_REQUESTS},
        "servers": [{"name": n, "capacity": c} for n, c in zip(C.SERVER_NAMES, C.CAPACITIES)],
        "environment": {"demand_mean_ms": C.DEMAND_MEAN, "p_heavy": C.P_HEAVY, "max_wait_ms": C.MAX_WAIT,
                        "q_ref_ms": C.Q_REF, "p_outage": C.P_OUTAGE, "outage_len_requests": C.OUTAGE_LEN},
        "test_scenarios": C.TEST_SCENARIOS,
    }
