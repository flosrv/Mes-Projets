from imports import *
import warnings

def display_null_counts(df):
    row_count = df.shape[0]
    null_counts = df.isnull().sum()
    
    formatted_output = "\n".join(
        f"{col:<25}{count:<4}/ {row_count}" for col, count in null_counts.items()
    )
    
    print(formatted_output)

def fetch_table_data(conn, schema, table_name, as_df=False):
    """
    Récupère toutes les données d'une table PostgreSQL et les charge dans un DataFrame Pandas ou sous forme de JSON.

    :param conn: Connexion SQLAlchemy active à la base de données PostgreSQL.
    :param schema: Nom du schéma contenant la table.
    :param table_name: Nom de la table à récupérer.
    :param as_df: Si True, retourne un DataFrame Pandas, sinon retourne les données en JSON.
    :return: DataFrame ou JSON contenant les données de la table.
    """
    query = text(f'SELECT * FROM "{schema}"."{table_name}"')
    df = pd.read_sql(query, conn)
    df = df.reset_index(drop=True)


    if as_df:
        return df
    else:
        result_dict = {}
        for idx, row in df.iterrows():
            row_dict = row.to_dict()  # Convertir chaque ligne en dictionnaire
            result_dict[idx] = row_dict  # Ajouter chaque ligne comme une entrée dans le dictionnaire
        return result_dict
      
def drop_columns_if_exist(df, columns_to_drop):
    existing_columns = []
    for col in columns_to_drop:
        if col in df.columns:
            existing_columns.append(col)
            print(f"Colonne '{col}' Supprimée")
        else: 
            print(f"Colonne '{col}' Non Trouvée")
    return df.drop(columns=existing_columns)

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
        try:
            conn.execute(text(create_table_query))  # Exécution directe de la requête
            conn.commit()  # S'assurer que la transaction est validée
            print(f"Table '{table_name}' created in schema '{schema}'.")
        except Exception as e:
            print(f"Error while creating table '{table_name}' in schema '{schema}': {e}")

    else:
        print(f"Table '{table_name}' already exists.")

def add_daytime_and_month_column(df, time_column='Datetime'):
    """
    Ajoute les colonnes 'DayTime' et 'Month' basées sur l'heure et le mois de la colonne 'Datetime' ou d'une autre colonne de type datetime.
    
    Parameters:
    df (pd.DataFrame): La DataFrame sur laquelle ajouter les colonnes.
    time_column (str): Nom de la colonne ou 'index' pour utiliser l'index datetime. Par défaut, utilise l'index.
    
    Returns:
    pd.DataFrame: DataFrame avec les nouvelles colonnes 'DayTime' et 'Month'.
    """
    # Vérifier si la colonne choisie est valide
    if time_column == 'index':
        time_data = df.index
    elif time_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_column]):
        time_data = df[time_column]
    else:
        raise ValueError("La colonne spécifiée n'est pas valide ou n'est pas de type datetime.")
    
    # Fonction pour déterminer la période de la journée
    def get_daytime(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    
    # Si on utilise l'index (DatetimeIndex)
    if isinstance(time_data, pd.DatetimeIndex):
        # Extraire l'heure et le mois de l'index datetime
        df['DayTime'] = time_data.hour.map(lambda hour: get_daytime(hour))
        df['Month'] = time_data.month
    elif isinstance(time_data, pd.Series) and pd.api.types.is_datetime64_any_dtype(time_data):
        # Si c'est une colonne datetime classique
        df['DayTime'] = time_data.dt.hour.map(lambda hour: get_daytime(hour))
        df['Month'] = time_data.dt.month
    else:
        raise ValueError("Les données temporelles ne sont pas de type datetime.")

    return df
    
    # Fonction pour déterminer la période de la journée
    def get_daytime(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    # Si on utilise l'index (DatetimeIndex)
    if isinstance(time_data, pd.DatetimeIndex):
        # Extraire l'heure et le mois de l'index datetime
        df['DayTime'] = time_data.hour.map(lambda hour: get_daytime(hour))
        df['Month'] = time_data.month
    else:
        # Si on utilise une colonne datetime classique
        df['DayTime'] = time_data.dt.hour.map(lambda hour: get_daytime(hour))
        df['Month'] = time_data.dt.month
    
    return df

def convert_to_datetime(date_value):
    try:
        # Si l'entrée est déjà un objet datetime, on le retourne directement
        if isinstance(date_value, datetime):
            return date_value
        
        # Si l'entrée est un objet pandas.Timestamp, on le convertit en datetime
        if isinstance(date_value, pd.Timestamp):
            return date_value.to_pydatetime()
        
        # Si l'entrée est une chaîne de caractères, on tente de la convertir en datetime
        if isinstance(date_value, str):
            return datetime.fromisoformat(date_value)
        
        # Si le format n'est pas reconnu, on renvoie une erreur
        print(f"Format non pris en charge pour : {date_value}")
        return None
    except ValueError:
        # En cas d'erreur, on retourne None pour éviter de casser le programme
        print(f"Erreur de conversion pour : {date_value}")
        return None

def process_and_resample(df, column_name, resample_interval='h'):
    try:
        # Vérification si la colonne existe dans le DataFrame
        if column_name not in df.columns:
            print(f"Erreur : La colonne '{column_name}' n'existe pas dans le DataFrame.")
            return df

        # Appliquer la fonction lambda qui utilise convert_to_datetime pour chaque élément
        df[column_name] = df[column_name].apply(lambda x: convert_to_datetime(x))

        # Supprimer les lignes où la conversion a échoué (valeurs None)
        df = df.dropna(subset=[column_name])

        # Filtrer pour garder uniquement les lignes où les minutes et les secondes sont 00
        df = df[df[column_name].dt.minute == 0]
        df = df[df[column_name].dt.second == 0]

        # Renommer la colonne spécifiée en 'Datetime' à la fin
        df.rename(columns={column_name: 'Datetime'}, inplace=True)

        # Retourner la DataFrame résultante
        return df
    
    except ValueError as e:
        print(f"Erreur de conversion des dates : {e}")
        return df
    except KeyError as e:
        print(f"Erreur : La colonne spécifiée n'a pas été trouvée dans le DataFrame. ({e})")
        return df
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return df

def handle_null_values(df):
    # Calculate the percentage of missing values for each column
    missing_percent = (df.isnull().sum() / len(df)) * 100

    # Lists to group columns by action type
    dropped_columns_100 = []
    dropped_columns_above_50 = []
    imputed_columns = []
    skipped_columns = []

    # Handle missing values
    for column in df.columns:
        null_percentage = missing_percent[column]  # Now it's a single value

        if isinstance(null_percentage, (int, float)):
            null_percentage = float(null_percentage)  # Ensure it's a float
            
            if null_percentage == 100:
                dropped_columns_100.append(column)
                df = df.drop(columns=[column])
            elif null_percentage > 50:
                dropped_columns_above_50.append(column)
                df = df.drop(columns=[column])
            elif null_percentage > 0:
                if df[column].dtype in ['float64', 'int64']:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    imputed_columns.append(column)
                else:
                    skipped_columns.append(column)

    # Print logs
    if dropped_columns_100:
        print(f"Dropped columns (100% missing): {', '.join(dropped_columns_100)}")
    if dropped_columns_above_50:
        print(f"Dropped columns (>50% missing): {', '.join(dropped_columns_above_50)}")
    if imputed_columns:
        print(f"Imputed columns (<50% missing, median): {', '.join(imputed_columns)}")
    if skipped_columns:
        print(f"Skipped non-numeric columns: {', '.join(skipped_columns)}")

    return df

def meteo_api_request(coordinates, mode='historical', days=92, interval='hourly'):
    """
    Fonction pour récupérer les données météo depuis l'API Open-Meteo avec cache et réessayer en cas d'erreur.
    
    Paramètres:
        coordinates (list) : liste de coordonnées [latitude, longitude]
        mode (str) : intervalle de données ('forecast' ou 'historical', par défaut 'historical')
        days (int) : nombre de jours dans le passé ou dans le futur (par défaut : 92 pour historique)
        interval (str) : intervalle des données ('hourly' ou 'daily', par défaut 'hourly')

    Retourne :
        pd.DataFrame : un DataFrame avec les données météo
    """
    # Fonction utilitaire pour convertir les coordonnées avec ou sans suffixe (ex: '45.5W', '-45.5')
    def parse_coordinates(coord):
        # Vérifie si la coordonnée a un suffixe de direction (W, E, N, S)
        pattern = r"^([-+]?\d+(\.\d+)?)([NSEW]?)$"
        match = re.match(pattern, str(coord))
        if match:
            value = float(match.group(1))
            direction = match.group(3)
            
            if direction == 'W' or direction == 'S':
                value = -abs(value)  # Si direction est Ouest ou Sud, on inverse la valeur
            
            return value
        else:
            raise ValueError(f"Coordonnée invalide : {coord}")

    # Convertir les coordonnées
    latitude = parse_coordinates(coordinates[0])
    longitude = parse_coordinates(coordinates[1])

    # Setup de l'API client avec retry et cache
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    openmeteo = openmeteo_requests.Client(session=cache_session)

    url = "https://api.open-meteo.com/v1/forecast"
    
    # Paramètres de base
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": days if mode == 'historical' else None,  # Si historique, utiliser 'past_days'
        "forecast_days": days if mode == 'forecast' else None,  # Si forecast, utiliser 'forecast_days'
        "hourly" if interval.lower() == 'hourly' else "daily": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "showers", 
            "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", 
            "cloud_cover_high", "visibility", "wind_speed_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm", 
            "is_day"
        ]
    }

    # Faire l'appel API
    responses = openmeteo.weather_api(url, params=params)

    # Traiter la réponse pour le premier emplacement
    response = responses[0]  # On prend la première réponse si plusieurs lieux sont fournis

    # Initialisation du dictionnaire pour les données à retourner
    data = {}

    # Processus des données en fonction du mode sélectionné
    if mode == 'historical':
        # Traitement pour données historiques
        if interval == 'hourly':
            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            hourly_variables = [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "showers", 
                "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", 
                "cloud_cover_high", "visibility", "wind_speed_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm", 
                "is_day"
            ]
            
            for i, var in enumerate(hourly_variables):
                hourly_data[var] = [round(value, 2) for value in hourly.Variables(i).ValuesAsNumpy()]

            return pd.DataFrame(hourly_data)

        elif interval == 'daily':
            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )
            }
            daily_variables = [
                "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", 
                "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", 
                "precipitation_sum", "rain_sum", "showers_sum", "precipitation_hours", "precipitation_probability_max", 
                "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum"
            ]
            
            for i, var in enumerate(daily_variables):
                daily_data[var] = [round(value, 2) for value in daily.Variables(i).ValuesAsNumpy()]

            return pd.DataFrame(daily_data)

    # If Forecast is chosen
    elif mode == 'forecast':
        # Traitement pour prévisions
        if interval == 'hourly':
            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            hourly_variables = [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "showers", 
                "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", 
                "cloud_cover_high", "visibility", "wind_speed_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm", 
                "is_day"
            ]
            
            for i, var in enumerate(hourly_variables):
                hourly_data[var] = [round(value, 2) for value in hourly.Variables(i).ValuesAsNumpy()]

            return pd.DataFrame(hourly_data)

        elif interval == 'daily':
            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )
            }
            daily_variables = [
                "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", 
                "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", 
                "precipitation_sum", "rain_sum", "showers_sum", "precipitation_hours", "precipitation_probability_max", 
                "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum"
            ]
            
            for i, var in enumerate(daily_variables):
                daily_data[var] = [round(value, 2) for value in daily.Variables(i).ValuesAsNumpy()]

            return pd.DataFrame(daily_data)

def connect_postgresql(user: str, password : str, host, port):
    # Créer l'engine PostgreSQL
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
    conn = engine.connect()
    return conn
# Fonction pour créer une base de données si elle n'existe pas
def create_database(dbname, user, password, host, port):
    """Crée une base de données si elle n'existe pas déjà."""
    # Se connecter à la base 'postgres' pour créer d'autres DB
    conn = connect_postgresql(user=user, password=password, host =host, port=port)
    if conn is None:
        return  # Si la connexion échoue, on arrête la fonction

    conn.autocommit = True  # Désactiver la transaction active
    cur = conn.cursor()
    
    # Vérifier si la base de données existe déjà
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
    exists = cur.fetchone()

    if not exists:
        try:
            cur.execute(f"CREATE DATABASE {dbname}")
            print(f"✅ Base '{dbname}' créée.")
        except Exception as e:
            print(f"Erreur lors de la création de la base de données : {e}")
    else:
        print(f"ℹ️ La base '{dbname}' existe déjà.")

    cur.close()
    conn.close()

def get_station_metadata(station_id):
    return api.station(station_id=station_id)

def extract_lat_lon_from_station_list(location):

    # Expression régulière pour capturer la latitude et la longitude
    lat_match = re.search(r'([+-]?\d+\.\d+|\d+)([NS])', location)
    lon_match = re.search(r'([+-]?\d+\.\d+|\d+)([EW])', location)
    
    if lat_match and lon_match:
        # Extraction des valeurs
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        
        # Inverser la direction de la latitude et longitude si nécessaire
        if lat_match.group(2) == 'S':  # Si la latitude est au Sud
            lat = -lat
        if lon_match.group(2) == 'W':  # Si la longitude est à l'Ouest
            lon = -lon
        
        lat = round(lat, 2)
        lon = round(lon, 2)
        return lat, lon
    return None, None

def print_with_flush(message):

    sys.stdout.write(f'\r{message}  ')  # \r permet de revenir au début de la ligne
    sys.stdout.flush()  # Force l'affichage immédiat

def parse_buoy_json(buoy_metadata):
    # Vérifier la présence des clés requises
    if 'Name' not in buoy_metadata or 'Location' not in buoy_metadata:
        raise ValueError("Les clés 'Name' et 'Location' doivent être présentes dans les données.")

    Name = buoy_metadata['Name']

    # Trouver tout ce qui vient après le premier tiret
    name_parts = Name.split(' - ', 2)
    
    station_zone = name_parts[1].strip().lower()

    station_id = Name.split(' ')[1]
    # Extraction des coordonnées depuis "Location"
    location_parts = buoy_metadata["Location"].split()
    if len(location_parts) < 4:
        raise ValueError("Format de 'Location' invalide")

    lat_buoy = f"{float(location_parts[0]):.2f}{location_parts[1]}"
    lon_buoy = f"{float(location_parts[2]):.2f}{location_parts[3]}"

    return station_id, station_zone, lat_buoy, lon_buoy

def fetch_and_add_data(table_dict, conn, schema, as_df=False):
    for station_id, tables in table_dict.items():
        # Vérifie que 'tables' est un dictionnaire
        if isinstance(tables, dict):
            bronze_marine_table = tables.get("bronze marine table name")
            bronze_meteo_table = tables.get("bronze meteo table name")

            try:
                if bronze_marine_table:
                    query = text(f'SELECT * FROM "{schema}"."{bronze_marine_table}"')
                    marine_data = pd.read_sql(query, conn)
                    # Conversion en JSON-compatible si nécessaire
                    if not as_df:
                        tables["silver marine data"] = marine_data.to_dict(orient='records')  # convert to dict
                    else:
                        tables["silver marine data"] = marine_data  # keep as DataFrame if needed

                if bronze_meteo_table:
                    query = text(f'SELECT * FROM "{schema}"."{bronze_meteo_table}"')
                    meteo_data = pd.read_sql(query, conn)
                    # Conversion en JSON-compatible si nécessaire
                    if not as_df:
                        tables["silver meteo data"] = meteo_data.to_dict(orient='records')  # convert to dict
                    else:
                        tables["silver meteo data"] = meteo_data  # keep as DataFrame if needed

            except Exception as e:
                print(f"Erreur pendant l'exécution pour station {station_id}: {e}")
                conn.rollback()  # En cas d'erreur, on annule la transaction en cours
        else:
            print(f"Warning: Element {station_id} is not a dictionary {type(station_id)},  skipping.")
    
    return table_dict

def auto_convert(df):
    warnings.filterwarnings("ignore", category=UserWarning)

    for col in df.columns:
        # Si la colonne est de type 'object' ou 'str', tenter la conversion datetime
        if df[col].dtype == 'object' or df[col].dtype == 'str':
            try:
                # Utiliser la fonction convert_to_datetime pour convertir les valeurs
                df[col] = df[col].apply(lambda x: convert_to_datetime(x))
            except Exception as e:
                pass

        # Si la colonne est encore de type 'object' ou 'str', tenter la conversion numérique
        if df[col].dtype == 'object' or df[col].dtype == 'str':
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception as e:
                pass

    return df

def convert_coordinates(lat, lon):
    # Conversion de la latitude
    lat_value = float(lat[:-1])  # On enlève la lettre 'n' ou 's' et on garde la valeur numérique
    if lon[-1].lower() == 's':  # Si la latitude est dans l'hémisphère sud
        lat_value = -lat_value

    # Conversion de la longitude
    lon_value = float(lon[:-1])  # On enlève la lettre 'e' ou 'w' et on garde la valeur numérique
    if lon[-1].lower() == 'w':  # Si la longitude est dans l'hémisphère ouest
        lon_value = -lon_value

    return round(lat_value, 2), round(lon_value, 2)

def load_data_in_table(conn, schema, table_name, df, key_column):
    # Ouvrir une nouvelle transaction
    try:
        inspector = inspect(conn)

        # Vérifier et créer le schéma si nécessaire
        if schema not in inspector.get_schema_names():
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            conn.commit()
            print(f'Schema "{schema}" created.')

        # Vérifier si la table existe
        if not inspector.has_table(table_name, schema=schema):
            print(f"Table '{table_name}' does not exist. Creating...")

            # Définir dynamiquement les types SQL
            type_mapping = {
                'int64': 'INTEGER',
                'float64': 'FLOAT',
                'object': 'TEXT',
                'bool': 'BOOLEAN'
            }
            columns_sql = ", ".join([f'"{col}" {type_mapping.get(str(df[col].dtype), "TEXT")}' for col in df.columns])

            create_table_query = f'''
            CREATE TABLE "{schema}"."{table_name}" (
                id SERIAL PRIMARY KEY,
                {columns_sql}
            );'''
            conn.execute(text(create_table_query))
            conn.commit()
            print(f"Table '{table_name}' created in schema '{schema}'.")

        # Vérifier et éviter les doublons
        existing_values_set = set()
        if key_column:
            try:
                query = f'SELECT DISTINCT "{key_column}" FROM "{schema}"."{table_name}"'
                existing_values = pd.read_sql(query, conn)
                existing_values_set = set(existing_values[key_column])
            except Exception as e:
                print(f"Error retrieving existing values: {e}")

        new_data = df if key_column is None else df[~df[key_column].isin(existing_values_set)]

        rows_before_insert = len(pd.read_sql(f'SELECT * FROM "{schema}"."{table_name}"', conn))

        rows_inserted = 0  # Initialiser à 0, afin qu'il y ait toujours une valeur
        if not new_data.empty:
            try:
                # Utiliser bulk insert pour plus de rapidité
                new_data.to_sql(table_name, conn, schema=schema, if_exists='append', index=False, method='multi')

                # Compter les lignes réellement insérées
                rows_inserted = len(new_data)
                print(f"New data inserted successfully!")
            except Exception as e:
                print(f"Error inserting new data: {e}\n")
        else:
            print("No new data to insert.\n")

        # Compter le nombre de lignes dans la table après l'insertion
        rows_after_insert = len(pd.read_sql(f'SELECT * FROM "{schema}"."{table_name}"', conn))

        # Affichage des résultats
        if rows_inserted == 0:
            print(f"No new data was inserted.")
        
        print(f"Rows in table before insertion: {rows_before_insert}")
        print(f"Rows inserted: {rows_inserted}")
        print(f"Rows in table after insertion: {rows_after_insert}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()  # Annuler la transaction en cas d'erreur
    finally:
        conn.commit()  # Toujours valider la transaction

def drop_tables(conn, schema_name, drop_schema=False, table_name_filter=None):
    try:
        # Vérifier si le schéma existe
        inspector = inspect(conn)
        schemas = inspector.get_schema_names()
        
        if schema_name not in schemas:
            print(f"Schema '{schema_name}' does not exist.")
            return
        
        # Obtenir la liste des tables du schéma
        print(f"Fetching tables from schema '{schema_name}'...")
        
        query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'"
        tables = pd.read_sql(query, conn)
        
        if tables.empty:
            print(f"No tables found in schema '{schema_name}'.")
            return
        
        # Si un filtre sur les tables est fourni, l'appliquer
        if table_name_filter:
            tables = tables[tables['table_name'].str.contains(table_name_filter, case=False, na=False)]
            if tables.empty:
                print(f"No tables found matching the filter '{table_name_filter}' in schema '{schema_name}'.")
                return
            print(f"Tables matching the filter '{table_name_filter}':\n{tables}")
        
        # Gérer les verrous : avant de supprimer, vérifier les processus en attente de verrou
        print("Checking for existing locks...")
        query_locks = """
            SELECT
                pg_stat_activity.pid,
                pg_stat_activity.state,
                pg_locks.mode,
                pg_class.relname,
                pg_stat_activity.query
            FROM
                pg_stat_activity
            JOIN
                pg_locks ON pg_stat_activity.pid = pg_locks.pid
            JOIN
                pg_class ON pg_locks.relation = pg_class.oid
            WHERE
                pg_stat_activity.state = 'idle in transaction';
        """
        locks = pd.read_sql(query_locks, conn)
        
        if not locks.empty:
            print(f"Found active locks:\n{locks}")
            print("Waiting 5 seconds before continuing...")
            time.sleep(5)  # Attendre 5 secondes avant de tenter la suppression des tables
        
        # Supprimer les tables une par une dans des transactions distinctes
        for table in tables['table_name']:
            try:
                # Commencer une transaction distincte pour chaque table
                print(f"\nDropping table '{table}'...")
                # Begin a transaction for each table drop
                with conn.begin():
                    conn.execute(text(f'DROP TABLE IF EXISTS "{schema_name}"."{table}" CASCADE'))
                    print(f"Table '{table}' dropped.")
                    time.sleep(1)  # Délai d'une seconde entre chaque suppression
            except Exception as e:
                print(f"Error dropping table '{table}': {e}")
        
        # Si l'argument drop_schema est True, supprimer également le schéma
        if drop_schema:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
            print(f"\nSchema '{schema_name}' and all its objects have been dropped.")
        else:
            print(f"\nTables dropped from schema '{schema_name}', but schema not removed.")

    except Exception as e:
        print(f"Error dropping tables in schema '{schema_name}': {e}")

def list_tables_info(conn, schema):
    try:
        # Obtenir une liste des tables dans le schéma
        inspector = inspect(conn)
        tables = inspector.get_table_names(schema=schema)
        
        if not tables:
            print(f"Aucune table trouvée dans le schéma '{schema}'.")
            return
        
        print(f"Tables dans le schéma '{schema}':\n")
        
        # Parcourir chaque table et obtenir le nombre de lignes
        for table in tables:
            query = f"SELECT COUNT(*) FROM \"{schema}\".\"{table}\""
            row_count = pd.read_sql(query, conn).iloc[0, 0]
            print(f"Table: {table}\nNombre de lignes: {row_count}")
    
    except Exception as e:
        print(f"Erreur lors de la récupération des informations des tables : {e}")

def count_files_in_directory(output_dir):
    try:
        # Vérifier si le dossier existe
        if not os.path.exists(output_dir):
            print(f"Le dossier {output_dir} n'existe pas.")
            return
        
        # Liste des fichiers dans le dossier
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        
        # Si le dossier est vide
        if not files:
            print(f"Aucun fichier trouvé dans le dossier {output_dir}.")
            return
        
        print(f"Nombre de fichiers dans le dossier '{output_dir}': {len(files)}\n")
        
        # Analyser chaque fichier
        for file in files:
            file_path = os.path.join(output_dir, file)
            file_name, file_extension = os.path.splitext(file)
            
            # Vérifier si c'est un fichier CSV
            if file_extension.lower() == '.csv':
                try:
                    # Lire le fichier CSV avec pandas
                    df = pd.read_csv(file_path)
                    num_rows, num_cols = df.shape
                    
                    # Afficher les informations sur le fichier
                    print(f"Nom du fichier: {file_name}")
                    print(f"Format: {file_extension}")
                    print(f"Nombre de lignes: {num_rows}, Nombre de colonnes: {num_cols}\n")
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier {file}: {e}")
            else:
                print(f"Fichier {file} n'est pas un fichier CSV.\n")
    
    except Exception as e:
        print(f"Erreur dans la fonction count_files_in_directory: {e}")

def show_popup(text):
    root = tk.Tk()
    root.title("Notification")
    
    # Empêcher la redimension de la fenêtre
    root.resizable(False, False)

    # Centrer le texte avec un peu de padding
    label = tk.Label(root, text=text, padx=20, pady=20, font=("Arial", 12))
    label.pack()

    # Fermer la fenêtre après 4 secondes
    root.after(4000, root.destroy)
    
    # Afficher la fenêtre
    root.mainloop()

def rename_columns(df, rename_spec):
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Si rename_spec est un dictionnaire, on le transforme en liste de dictionnaires pour un traitement uniforme
    if isinstance(rename_spec, dict):
        rename_spec = [rename_spec]  # Convertir en liste de dictionnaires pour uniformité
    
    # Process each dictionary in the list of rename_spec
    if isinstance(rename_spec, list):
        for rename_dict in rename_spec:
            # Filtrer les colonnes qui existent à la fois dans le DataFrame et rename_spec
            existing_columns = {col: rename_dict[col] for col in rename_dict if col in df.columns}
            
            # Renommer les colonnes uniquement si elles existent dans le DataFrame
            if existing_columns:
                df.rename(columns=existing_columns, inplace=True)
            else:
                print(f"⚠️ Aucune colonne à renommer pour ce spécification : {rename_dict}")
    
    # Retourner le DataFrame modifié
    return df
