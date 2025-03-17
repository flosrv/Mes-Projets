import json, psycopg2, random
from sqlalchemy import create_engine, text
from faker import Faker

# Charger les credentials
postgres_creds_path = r"C:\Users\f.gionnane\Documents\Data Engineering\Credentials\postgresql_creds.json"

with open(postgres_creds_path, 'r', encoding='utf-8') as file:
    content = json.load(file)

# Récupération des valeurs
host = content['host']
port = content['port']
user = content['user']
password = content['password']
dbname = "MyProjects" 
schema = 'Learn_SSIS'

# Connexion à PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")

try:
    conn = engine.connect()
    print("Connexion réussie !")
    conn.close()
except Exception as e:
    print(f"Erreur de connexion : {e}")

def create_schema_and_table(conn, schema, table_name, col):
    # Vérifier si le schéma existe, sinon le créer
    result = conn.execute(text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema}'"))
    if not result.fetchone():
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS \"{schema}\""))
        print(f'Schema "{schema}" created.')
    else:
        print(f'Schema "{schema}" already exists.')

    # Vérifier si la table existe, sinon la créer
    result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname = '{schema}' AND tablename = '{table_name}')"))
    table_exists = result.fetchone()[0]

    if not table_exists:
        print(f"Table '{table_name}' does not exist. Creating...")

        # Définir un type de colonne par défaut (par exemple, 'VARCHAR')
        columns_definition = ', '.join([f'"{col_name}" VARCHAR' for col_name in col])

        # Créer la requête SQL pour créer la table avec les colonnes spécifiées
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS \"{schema}\".\"{table_name}\" (
            id SERIAL PRIMARY KEY,
            {columns_definition}
        );
        """
        conn.execute(text(create_table_query))  # Exécution directe de la requête
        conn.commit()  # S'assurer que la transaction est validée
        print(f"Table '{table_name}' created in schema '{schema}'.")

    else:
        print(f"Table '{table_name}' already exists.")



# Fonction pour créer une table sans insérer de données
def create_fake_table(conn, schema):
    # Générer un nom de table aléatoire
    table_name = f"fake_data_{random.randint(1000, 9999)}"

    # Définir les colonnes de la table
    columns = ["name", "email", "address", "phone"]
    
    # Créer la table
    create_schema_and_table(conn, schema, table_name, columns)

    print(f"Table '{table_name}' créée dans le schéma '{schema}'.")

# Assurez-vous que la connexion est bien ouverte avant d'appeler la fonction
with engine.connect() as conn:
    create_fake_table(conn=conn, schema=schema)

