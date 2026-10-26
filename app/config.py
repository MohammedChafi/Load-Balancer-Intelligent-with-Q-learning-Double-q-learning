"""
Paramètres du projet. Chaque valeur renvoie à une section du document de conception.
"""

# --- Serveurs (section 1 et 6.2 : chaque serveur a une capacité de traitement) -------------
SERVER_NAMES = ["A", "B", "C"]
CAPACITIES = [0.7, 1.0, 1.5]        # vitesse de traitement (1.0 = référence)
N_SERVERS = 3

# --- États S (section 2.1) : niveau de CPU de chaque serveur, 3 niveaux -> 3^3 = 27 états ---
CPU_LOW, CPU_HIGH = 40.0, 70.0      # faible < 40 %, moyen 40-70 %, élevé > 70 %
LEVEL_NAMES = ["faible", "moyen", "élevé"]
N_LEVELS = 3
N_STATES = N_LEVELS ** N_SERVERS    # 27

# --- Actions A (section 2.2) : choisir le serveur A, B ou C -----------------------------------
N_ACTIONS = N_SERVERS

# --- Récompense R (section 2.4) : R = -(a*T/Tref + b*L/100 + d*F) (+ pénalité de surcharge) ---
T_REF = 100.0                       # temps de référence en ms
ALPHA, BETA, DELTA = 1.0, 0.5, 5.0  # poids latence, CPU, échec (à justifier dans le rapport)
ZETA = 1.0                          # pénalité de surcharge (section 2.4, "À prévoir")
OVERLOAD_CPU = 90.0                 # CPU (%) au-delà duquel on parle de surcharge
FAIL_LATENCY = 5 * T_REF            # latence comptée pour une requête rejetée

# --- Transitions P (section 2.3 et 6.2) : modèle de files d'attente simplifié -----------------
DEMAND_MEAN = 50.0                  # ms de calcul moyen par requête (à vitesse 1)
P_HEAVY, HEAVY_MEAN, LIGHT_MEAN = 0.10, 150.0, 35.0   # requêtes lourdes / légères
Q_REF = 200.0                       # attente (ms) qui correspond à un CPU de 100 %
MAX_WAIT = 400.0                    # attente max : au-delà, la requête est rejetée
BURST_AMPL, BURST_PERIOD = 0.5, 70  # trafic en vagues
BURST_FACTOR = 2.5                  # multiplicateur du débit pendant un "pic de trafic"
P_OUTAGE, OUTAGE_LEN = 0.004, 40    # pannes aléatoires (durée exprimée en nombre de requêtes)

# --- Apprentissage (section 3, 4 et tableau 6.1) -----------------------------------------------
LR = 0.1                            # η, taux d'apprentissage
GAMMA = 0.9                         # γ, facteur d'actualisation
EPS_START, EPS_END = 1.0, 0.05      # ε-greedy
EPS_DECAY_FRACTION = 0.8            # ε atteint sa valeur finale après 80 % des épisodes
N_EPISODES = 5000
N_REQUESTS = 200                    # requêtes par épisode
TRAIN_LOAD_RANGE = (0.3, 1.0)       # taux d'utilisation tiré au hasard à chaque épisode

# --- Évaluation (section 7) ----------------------------------------------------------------------
TEST_SCENARIOS = {"faible": 0.4, "moyen": 0.7, "élevé": 0.9}
N_TEST_SEEDS = 20
TEST_SEED_BASE = 1_000_000          # graines de test, distinctes de celles de l'entraînement
