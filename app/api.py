"""API REST (Flask). Toute la logique est dans les autres modules ; ici on ne fait qu'exposer."""
import os
from flask import Flask, jsonify, request, send_from_directory

from . import config as C
from . import mdp
from .agents import RoundRobin
from .evaluator import evaluate_all
from .live import LiveSession
from .trainer import Trainer, NAMES
from .optimization_api import install_optimization

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app():
    app = Flask(__name__, static_folder=None)
    trainer = Trainer(os.path.join(ROOT, "models", "qtables.json"))
    trainer.load_model()
    sessions = {}
    install_optimization(app, ROOT)

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(os.path.join(ROOT, "static"), "index.html")

    # ---------------------------------------------------------------- MDP
    @app.get("/api/config")
    def config():
        return jsonify(mdp.describe())

    # ---------------------------------------------------------------- entraînement
    @app.post("/api/training/start")
    def training_start():
        try:
            trainer.start(request.get_json(silent=True) or {})
        except RuntimeError as e:
            return err(str(e), 409)
        except (ValueError, TypeError) as e:
            return err(str(e))
        return jsonify(trainer.status())

    @app.post("/api/training/stop")
    def training_stop():
        trainer.stop()
        return jsonify({"stopping": True})

    @app.get("/api/training/status")
    def training_status():
        return jsonify(trainer.status())

    @app.get("/api/training/history")
    def training_history():
        return jsonify(trainer.history(request.args.get("since", 0, type=int)))

    @app.get("/api/training/qtables")
    def training_qtables():
        if not trainer.model_available:
            return err("Aucun entraînement.", 404)
        return jsonify(trainer.tables())

    @app.get("/api/training/updates")
    def training_updates():
        agent = request.args.get("agent", "qlearning")
        if agent not in NAMES:
            return err("agent inconnu")
        if trainer.agents is None:
            return jsonify([])
        return jsonify(trainer.get_updates(agent, request.args.get("since", 0, type=int),
                                           request.args.get("limit", 50, type=int)))

    @app.get("/api/training/snapshots")
    def training_snapshots():
        with trainer.lock:
            return jsonify(trainer.snapshots)

    # ---------------------------------------------------------------- évaluation
    @app.post("/api/evaluation/run")
    def evaluation_run():
        if trainer.running:
            return err("Attendez la fin de l'entraînement.", 409)
        if not trainer.model_available:
            return err("Entraînez d'abord les agents.", 409)
        n = (request.get_json(silent=True) or {}).get("seeds", C.N_TEST_SEEDS)
        trainer.eval_result = evaluate_all(trainer.agents, n_seeds=max(1, min(int(n), 200)))
        return jsonify(trainer.eval_result)

    @app.get("/api/evaluation/results")
    def evaluation_results():
        if trainer.eval_result is None:
            return err("Pas encore d'évaluation.", 404)
        return jsonify(trainer.eval_result)

    # ---------------------------------------------------------------- simulation en direct
    @app.post("/api/live/start")
    def live_start():
        d = request.get_json(silent=True) or {}
        algo = d.get("algorithm", "round_robin")
        load = float(d.get("load", 0.7))
        if not 0.1 <= load <= 1.2:
            return err("load doit être entre 0.1 et 1.2")
        if algo == "round_robin":
            agent = RoundRobin()
        elif algo in NAMES:
            if trainer.running or not trainer.model_available:
                return err("Terminez l'entraînement avant d'utiliser cet algorithme.", 409)
            agent = trainer.agents[algo]
        else:
            return err("algorithme inconnu")
        sess = LiveSession(agent, load, d.get("seed"))
        sessions[sess.id] = sess
        while len(sessions) > 8:
            sessions.pop(next(iter(sessions)))
        return jsonify({"session": sess.id, "algorithm": algo, "load": load})

    def get_session(sid):
        return sessions.get(sid)

    @app.post("/api/live/<sid>/tick")
    def live_tick(sid):
        s = get_session(sid)
        if not s:
            return err("session inconnue", 404)
        return jsonify(s.tick((request.get_json(silent=True) or {}).get("dt_ms", 100)))

    @app.post("/api/live/<sid>/fault")
    def live_fault(sid):
        s = get_session(sid)
        if not s:
            return err("session inconnue", 404)
        k = (request.get_json(silent=True) or {}).get("server")
        k = C.SERVER_NAMES.index(k) if k in C.SERVER_NAMES else 0
        s.env.inject_fault(k)
        return jsonify({"server": C.SERVER_NAMES[k], "down_requests": C.OUTAGE_LEN})

    @app.post("/api/live/<sid>/burst")
    def live_burst(sid):
        s = get_session(sid)
        if not s:
            return err("session inconnue", 404)
        s.env.trigger_burst()
        return jsonify({"burst": True})

    @app.post("/api/live/<sid>/load")
    def live_load(sid):
        s = get_session(sid)
        if not s:
            return err("session inconnue", 404)
        s.env.set_load(max(0.1, min(float((request.get_json(silent=True) or {}).get("load", 0.7)), 1.2)))
        return jsonify({"load": s.env.load_factor})

    return app
