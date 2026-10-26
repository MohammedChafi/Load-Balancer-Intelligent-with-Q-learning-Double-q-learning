"""Routes and assets for the appended optimization section only."""
import os
from flask import Blueprint, jsonify, request
from .optimization import Optimizer, METHODS


def install_optimization(app, root):
    optimizer = Optimizer(os.path.join(root, "models", "optimization"))
    app.extensions["optimizer"] = optimizer
    routes = Blueprint("optimization", __name__, static_folder=os.path.join(root, "static"),
                       static_url_path="/optimization-assets")

    @routes.get("/api/optimization/status")
    def status():
        return jsonify(optimizer.snapshot())

    @routes.post("/api/optimization/start")
    def start():
        try:
            payload = request.get_json(silent=True)
            if payload is None:
                raise ValueError("Un objet JSON de paramètres est requis.")
            optimizer.start(payload)
        except RuntimeError as error:
            return jsonify(error=str(error)), 409
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(optimizer.snapshot()), 202

    @routes.post("/api/optimization/stop")
    def stop():
        optimizer.stop()
        return jsonify(optimizer.snapshot())

    @routes.get("/api/optimization/report")
    def report():
        state = optimizer.snapshot()
        if state["running"]:
            return jsonify(error="Attendez la fin ou arrêtez la recherche avant l’export."), 409
        if not state.get("id"):
            return jsonify(error="Aucune recherche à exporter."), 404
        response = jsonify(optimizer.report())
        response.headers["Content-Disposition"] = f'attachment; filename="optimisation-{state["id"]}.json"'
        return response

    app.register_blueprint(routes)
