from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from app.utils.db import db
from app.main.models import OrdenTrabajo

orden_trabajo_bp = Blueprint('ordenTrabajo', __name__, url_prefix='/ordenTrabajo')
@orden_trabajo_bp.route('/', methods=['GET'])
def get_orden_trabajo():
    """Obtiene la lista completa de OrdenTrabajo con todos los campos."""
    ordenes_trabajo = OrdenTrabajo.query.all()
    result = [OrdenTrabajo.__dict__ for OrdenTrabajo in ordenes_trabajo]
    for r in result:
        r.pop('_sa_instance_state', None)  
    return jsonify(result), 200