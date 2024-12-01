
from flask import Blueprint, jsonify, request
from app.utils.db import db
from app.main.models import Solicitud

solicitud_bp = Blueprint('solicitud', __name__)

@solicitud_bp.route('/historical-data', methods=['GET'])
def get_solicitudes():
    """Obtiene la lista completa de solicitudes con todos los campos."""
    solicitudes = Solicitud.query.all()
    result = [solicitud.__dict__ for solicitud in solicitudes]
    for r in result:
        r.pop('_sa_instance_state', None)  # Quitar metadatos internos de SQLAlchemy
    return jsonify(result), 200
