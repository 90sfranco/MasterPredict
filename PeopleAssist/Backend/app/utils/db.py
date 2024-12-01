from flask_sqlalchemy import SQLAlchemy

# Inicializar la instancia de SQLAlchemy
db = SQLAlchemy()

def init_db(app):
    """
    Inicializa la conexión a la base de datos y prepara las tablas si es necesario.
    """
    db.init_app(app)
    with app.app_context():
        # Verificar si las tablas existen; si no, se crean
        db.create_all()
        print("Conexión a la base de datos establecida y tablas verificadas.")
