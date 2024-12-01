from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from app.utils.db import db
from app.main.models import Usuario

usuario_bp = Blueprint('usuario', __name__, url_prefix='/usuarios')

@usuario_bp.route('/', methods=['GET'])
def get_usuarios():
    """Obtiene la lista completa de usuarios con todos los campos."""
    usuarios = Usuario.query.all()
    result = [usuario.__dict__ for usuario in usuarios]
    for r in result:
        r.pop('_sa_instance_state', None)  # Quitar metadatos internos de SQLAlchemy
    return jsonify(result), 200