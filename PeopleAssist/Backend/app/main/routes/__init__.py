from app.main.routes.solicitudes import solicitud_bp
from app.main.routes.usuarios import usuario_bp
from app.main.routes.orden_servicio import orden_servicio_bp
from app.main.routes.orden_trabajo import orden_trabajo_bp

def register_routes(app):
    app.register_blueprint(solicitud_bp)
    app.register_blueprint(usuario_bp)
    app.register_blueprint(orden_servicio_bp)
    app.register_blueprint(orden_trabajo_bp)