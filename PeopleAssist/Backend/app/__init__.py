from flask import Flask
from flask_cors import CORS
from app.utils.db import db, init_db
from app.main.routes import register_routes

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../instance/config.py')
    CORS(app, resources={r"/*": {"origins": "*"}})
    init_db(app)
    register_routes(app) 
    return app
