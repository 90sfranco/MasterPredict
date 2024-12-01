# Archivo de configuración para Flask y la base de datos

# Cadena de conexión a SQL Server
SQLALCHEMY_DATABASE_URI = (
    "mssql+pyodbc://bdadministrator:M4yC0%2A2024%2A@192.1.2.26/helppeople_NOV?driver=ODBC+Driver+17+for+SQL+Server"
)

'''
SQLALCHEMY_DATABASE_URI = (
"mssql+pyodbc://@DESKTOP-UL33RBD\SQLEXPRESS/helppeople_NOV?driver=ODBC+Driver+17+for+SQL+Server"
)
'''

# Evita rastrear modificaciones de objetos de SQLAlchemy (mejora el rendimiento)
SQLALCHEMY_TRACK_MODIFICATIONS = False

#
SQLALCHEMY_ECHO = True

# Clave secreta para la aplicación Flask
SECRET_KEY = "team-hackaton"

