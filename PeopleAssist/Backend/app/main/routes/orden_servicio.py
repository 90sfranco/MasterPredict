from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from app.utils.db import db
from app.main.models import OrdenServicio

orden_servicio_bp = Blueprint('ordenServicio', __name__, url_prefix='/ordenServicio')

@orden_servicio_bp.route('/', methods=['GET'])
def get_orden_servicio():
    """Obtiene la lista completa de OrdenTrabajo con todos los campos."""
    ordenes_servicio = OrdenServicio.query.all()
    result = [OrdenServicio.__dict__ for OrdenServicio in ordenes_servicio]
    for r in result:
        r.pop('_sa_instance_state', None)  
    return jsonify(result), 200