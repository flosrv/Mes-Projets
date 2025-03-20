from imports import *
import warnings

path_postgresql_creds = r"C:\Users\f.gionnane\Documents\Data Engineering\Credentials\postgresql_creds.json"
with open(path_postgresql_creds, 'r') as file:
    content = json.load(file)
    user = content["user"]
    password = content["password"]
    host = content["host"]
    port = content["port"]

db = "MyProjects"
schema = "End_To_End_Oceanography_ML"

# Créer l'engine PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
conn = engine.connect()

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

def process_and_resample(df, column_name, resample_interval='h'):
    try:
        # Vérification si la colonne existe dans le DataFrame
        if column_name not in df.columns:
            print(f"Erreur : La colonne '{column_name}' n'existe pas dans le DataFrame.")
            return df

        # Renommer la colonne spécifiée en 'Datetime' et la convertir en datetime
        df.rename(columns={column_name: 'Datetime'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='raise')

        # Définir la colonne 'Datetime' comme index
        df.set_index('Datetime', inplace=True)

        # Valider l'intervalle de resampling
        valid_intervals = ['h', 'd', 'm', 'w', 'M', 'Y']
        if resample_interval not in valid_intervals:
            print(f"Erreur : L'intervalle de resampling '{resample_interval}' n'est pas valide. Utilisez l'un des suivants : {', '.join(valid_intervals)}.")
            return df

        # Resampler selon l'intervalle choisi et prendre la moyenne
        df_resampled = df.resample(resample_interval).mean().round(2)

        # Obtenir les dates min et max directement depuis l'index
        min_date, max_date = df_resampled.index.min(), df_resampled.index.max()

        # Réinitialiser l'index et remettre la colonne 'Datetime' comme colonne normale
        df_resampled.reset_index(inplace=True)

        # Retourner la DataFrame résultante
        return df_resampled
    
    except ValueError as e:
        print(f"Erreur de conversion des dates : {e}")
        return df
    except KeyError as e:
        print(f"Erreur : La colonne spécifiée n'a pas été trouvée dans le DataFrame. ({e})")
        return df
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return df
#
def handle_null_values(df):
    # Calcul des pourcentages de valeurs manquantes
    missing_percent = round((df.isnull().sum() / len(df)) * 100, 2)
    missing_percent_str = missing_percent.astype(str) + '%'
    

    # Gestion des valeurs manquantes
    for column in df.columns:
        null_percentage = missing_percent[column]
        
        if null_percentage == 100:
            # Si 100% des valeurs sont manquantes, on supprime la colonne
            print(f"Supprime la colonne : {column} (100% de valeurs manquantes)")
            df = df.drop(column, axis=1)
        elif null_percentage > 50:
            # Si plus de 50% des valeurs sont manquantes, on supprime la colonne
            print(f"Supprime la colonne : {column} ({null_percentage}% de valeurs manquantes)")
            df = df.drop(column, axis=1)
        elif null_percentage > 0:
            # Si moins de 50% des valeurs sont manquantes, on impute avec la médiane
            print(f"Impute la colonne : {column} avec la médiane ({null_percentage}% de valeurs manquantes)")
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
       
    # Retourner le DataFrame modifié
    return df
#
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

def connect_postgresql(user: str, password : str, host =host, port=port):
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

    # Formatage du nom de la table avec des underscores pour remplacer les points
    table_name = f"station_{station_id}_{station_zone}_{lat_buoy}_{lon_buoy}"
    marine_data_table_name = re.sub(r'[^a-zA-Z0-9_-]', '', table_name.replace('.', '_').replace(' ', '_')).lower()

    return station_id, station_zone, lat_buoy, lon_buoy, marine_data_table_name

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

        if df[col].dtype == 'object' or df[col].dtype == 'str':
            try:

                df[col] = pd.to_datetime(df[col], errors='raise') 
            except Exception as e:
                pass


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
    # Vérifier si le schéma existe, sinon le créer
    try:
        result = conn.execute(text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema}'"))
        if not result.fetchone():
            try:
                conn.execute(text(f"CREATE SCHEMA \"{schema}\""))
                print(f'\nSchema "{schema}" created.')
            except Exception as e:
                print(f"Error creating schema: {e}")
        else:
            print(f'\nSchema "{schema}" already exists.')
    except Exception as e:
        print(f"Error checking schema: {e}")

    # Vérifier si la table existe
    try:
        result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname = '{schema}' AND tablename = '{table_name}')"))
        table_exists = result.fetchone()[0]
    except Exception as e:
        print(f"Error checking table existence: {e}")
        table_exists = False  # Considérer que la table n'existe pas si une erreur se produit

    if not table_exists:
        print(f"Table '{table_name}' does not exist. Creating...")
        try:
            # Générer le schéma SQL en fonction des colonnes du DataFrame
            columns_sql = ", ".join([f'"{col}" TEXT' for col in df.columns])  # Supposition : types TEXT par défaut
            create_table_query = f"""
            CREATE TABLE \"{schema}\".\"{table_name}\" (
                id SERIAL PRIMARY KEY,
                {columns_sql}
            );
            """
            conn.execute(text(create_table_query))
            conn.commit()  # Commit explicite
            print(f"Table '{table_name}' created in schema '{schema}'.")
        except Exception as e:
            print(f"Error creating table: {e}")

        # Insérer les données du DataFrame
        try:
            df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False)
            print("Data inserted successfully.\n")
        except Exception as e:
            print(f"Error inserting data: {e}")
    else:
        print(f"Table '{table_name}' already exists.")

        # Vérifier si la table est vide
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM \"{schema}\".\"{table_name}\""))
            row_count = result.fetchone()[0]
        except Exception as e:
            print(f"Error checking row count: {e}")
            row_count = 0  # Par défaut, considérer que la table est vide en cas d'erreur

        if row_count == 0:
            print("Table is empty, inserting data...")
            try:
                df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False)
                print("Data inserted successfully.")
            except Exception as e:
                print(f"Error inserting data: {e}")
        else:
            print("Table is not empty, avoiding duplicates...")

            # Récupérer les valeurs existantes de la colonne de référence
            try:
                existing_values = pd.read_sql(f'SELECT DISTINCT "{key_column}" FROM "{schema}"."{table_name}"', conn)
                existing_values_set = set(existing_values[key_column])
            except Exception as e:
                print(f"Error retrieving existing values: {e}")
                existing_values_set = set()

            # Filtrer les nouvelles données pour éviter les doublons
            new_data = df[~df[key_column].isin(existing_values_set)]

            if not new_data.empty:
                try:
                    new_data.to_sql(table_name, conn, schema=schema, if_exists='append', index=False)
                    print("New data inserted successfully.")
                except Exception as e:
                    print(f"Error inserting new data: {e}")
            else:
                print("No new data to insert.")
