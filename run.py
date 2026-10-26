"""Lance le serveur : python run.py  ->  http://localhost:5000"""
import os
from app.api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True, debug=False)
