# Load Balancer intelligent : Q-Learning et Double Q-Learning

Projet complet en Python qui suit le document de conception : MDP, simulateur, Q-Learning,
Double Q-Learning, comparaison avec Round Robin, API REST et front de simulation.
**Toute la logique et toutes les décisions sont dans Python.** Le front n'est qu'un client de l'API.

## Lancer

    pip install -r requirements.txt
    python run.py                          # puis ouvrir http://localhost:5000
    python -m unittest discover -s tests   # tests (formules du document, graines, pannes)

## Organisation (et lien avec le document)

| Fichier | Contenu | Section du document |
|---|---|---|
| `app/config.py` | tous les paramètres (serveurs, seuils CPU, poids, γ, η, ε, épisodes, scénarios) | 2, 6.1 |
| `app/mdp.py` | états S (27), actions A (3), récompense R, description du MDP | 2.1, 2.2, 2.4, 2.5 |
| `app/simulator.py` | environnement : files d'attente, latence, CPU, rejets, pannes, trafic (transitions P) | 2.3, 6.2 |
| `app/agents.py` | Round Robin, Q-Learning, Double Q-Learning (tables 27 x 3, masque des actions interdites) | 3, 4 |
| `app/episode.py` | boucle d'un épisode et calcul des métriques | 7 |
| `app/trainer.py` | entraînement en thread, historique, snapshots des tables, journal des mises à jour | 3.3, 7 |
| `app/evaluator.py` | phase de test : mêmes scénarios et mêmes graines, ε = 0 | 7 |
| `app/live.py` | simulation en direct : Python reçoit chaque requête et choisit le serveur | 6, 8 |
| `app/api.py` | routes REST | |
| `static/index.html` | front : affiche et pilote, ne décide rien | |

## API

| Route | Rôle |
|---|---|
| `GET /api/config` | description du MDP : 27 états, actions, formule de récompense, poids, γ, η, ε |
| `POST /api/training/start` | `{episodes, lr, gamma, eps_start, eps_end, seed, delay_ms}` lance l'entraînement |
| `POST /api/training/stop` | arrête l'entraînement |
| `GET /api/training/status` | progression, ε, moyennes récentes, nombre de mises à jour |
| `GET /api/training/history?since=N` | récompense, latence, rejets par épisode pour chaque agent, ε, niveau de trafic |
| `GET /api/training/qtables` | tables actuelles : Q, QA, QB, visites N |
| `GET /api/training/updates?agent=&since=&limit=` | détail de chaque mise à jour : état, action, r, Q ancien, cible, erreur TD, Q nouveau, table QA/QB |
| `GET /api/training/snapshots` | tables à 21 moments de l'entraînement |
| `POST /api/evaluation/run` | test des 3 méthodes sur 3 scénarios (`{seeds}`) |
| `GET /api/evaluation/results` | dernier résultat de test |
| `POST /api/live/start` | `{algorithm: round_robin/qlearning/double_qlearning, load}` ouvre une session |
| `POST /api/live/<id>/tick` | `{dt_ms}` : Python génère les requêtes, décide, renvoie événements, serveurs, métriques |
| `POST /api/live/<id>/fault` · `/burst` · `/load` | injecter une panne, un pic de trafic, changer le trafic |

## D'où viennent les données d'entraînement ?

Il n'y a pas de jeu de données. Chaque épisode, `simulator.py` génère un trafic aléatoire (arrivées de Poisson
modulées en vagues, 10 % de requêtes lourdes, pannes, niveau d'utilisation entre 30 % et 100 %). L'agent choisit
(ε-greedy), le simulateur renvoie (s', r) : le tuple (s, a, r, s') met à jour une case de la table, puis il est jeté.
Les deux agents voient exactement le même trafic (même graine). Le test utilise d'autres graines.
En production, on remplacerait le simulateur par des logs réels rejoués, ou par un apprentissage en ligne prudent.

## Choix à justifier dans le rapport

- Poids de la récompense (α=1, β=0,5, δ=5, ζ=1) : choisis par moi, le document ne les fixe pas.
- Le CPU est déduit du remplissage de la file (attente / 200 ms), comme le suggère la section 6.2.
- Les actions vers un serveur en panne sont interdites pour tous les algorithmes, Round Robin compris.
- Le décalage entre simulateur et réalité limite la validité de la politique apprise.
