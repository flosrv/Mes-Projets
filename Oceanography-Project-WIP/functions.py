from imports import *

Base = declarative_base()

CATALOG_TABLE = "pg_catalog.pg_tables"

def get_buoy_url(station_id):
            station_id_str = str(station_id)
            url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_id_str}"
            return url

def load_data_in_table(engine:sqlalchemy.engine.base.Engine, schema:str, 
                       table_name:str, df:pd.DataFrame, key_column:str):

    # Ouvrir une nouvelle transaction
    try:
        with engine.connect() as conn:
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
                    'int64': Integer(),
                    'float64': Float(),
                    'object':  String(),
                    'bool': Boolean()
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

def show_first_row(df):
    # Récupérer la première ligne du DataFrame
    first_row = df.iloc[0]
    
    # Formater l'affichage des valeurs et des types
    formatted_output = "\n".join(
        f"{col:<30} {value}  ({df[col].dtype})" for col, value in first_row.items()
    )
    
    # Afficher le résultat formaté
    print(formatted_output)

def show_null_counts(df):
    row_count = df.shape[0]
    null_counts = df.isnull().sum()
    
    formatted_output = "\n".join(
        f"{col:<40}{count:<4}/ {row_count}" for col, count in null_counts.items()
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
        
    except ValueError as e:
        # Si la conversion échoue, on retourne la valeur d'origine (sans la modifier)
        return date_value  # Retourne la valeur d'origine sans la modifier

def handle_null_values(df):
    # Calcul du pourcentage de valeurs manquantes par colonne
    missing_percent = (df.isnull().sum() / len(df)) * 100

    # Listes pour regrouper les colonnes selon l'action à prendre
    dropped_columns_100 = []
    dropped_columns_above_50 = []
    imputed_columns = []
    skipped_columns = []

    # Gestion des valeurs manquantes
    for column in df.columns:
        null_percentage = missing_percent[column]  # Calcul du pourcentage de valeurs manquantes

        if isinstance(null_percentage, (int, float)):
            null_percentage = float(null_percentage)  # Assurez-vous que c'est un float

            if null_percentage == 100:
                dropped_columns_100.append(column)
                df = df.drop(columns=[column])
            elif null_percentage > 50:
                # Si la colonne est numérique, imputer les valeurs manquantes par la médiane
                if df[column].dtype in ['float64', 'int64']:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    imputed_columns.append(column)  # Ajout à la liste des colonnes imputées
                else:
                    dropped_columns_above_50.append(column)
                    df = df.drop(columns=[column])  # Si ce n'est pas numérique, supprimer la colonne
            elif null_percentage > 0:
                if df[column].dtype in ['float64', 'int64']:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    imputed_columns.append(column)
                else:
                    skipped_columns.append(column)

    # Affichage des logs
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

# Extraction des valeurs avec sécurité
def safe_get(dico, key):
    value = dico.get(key, "N/A")  # Valeur par défaut si clé absente
    print(f"📊 {key} : {value}")
    return value

# 🔹 Nettoyer les valeurs numériques (supprimer tout sauf chiffres et ".")
def clean_numeric(value):
    if value is None:
        return None  # ⚠️ Évite de faire `.replace()` sur None !
    return re.sub(r"[^\d.]", "", value).strip()  # Supprime tout sauf chiffres et "."

def parse_buoy_json(buoy_metadata):
    print("\n🔍 Début du parsing de la bouée...")

    # Vérifier la présence des clés requises
    if not isinstance(buoy_metadata, dict):
        raise ValueError("❌ Les données fournies ne sont pas un dictionnaire valide.")

    if 'Name' not in buoy_metadata or 'Location' not in buoy_metadata:
        raise ValueError("❌ Les clés 'Name' et 'Location' doivent être présentes dans les données.")

    Name = buoy_metadata['Name']

    # Trouver tout ce qui vient après le premier tiret
    name_parts = Name.split(' - ', 2)
    station_zone = name_parts[1].strip().lower() if len(name_parts) > 1 else "Unknown"
    print(f"🌍 Zone de la station : {station_zone}")

    # Extraction de l'ID
    name_split = Name.split()
    station_id = name_split[1] if len(name_split) > 1 else "Unknown"
    print(f"🆔 Station ID : {station_id}")

    # Extraction des coordonnées depuis "Location"
    location_parts = buoy_metadata["Location"].split()

    if len(location_parts) < 4:
        raise ValueError("❌ Format de 'Location' invalide (au moins 4 éléments attendus).")

    try:
        lat_buoy = f"{float(location_parts[0]):.2f}{location_parts[1]}"
        lon_buoy = f"{float(location_parts[2]):.2f}{location_parts[3]}"
        print(f"✅ Coordonnées extraites : Latitude = {lat_buoy}, Longitude = {lon_buoy}")
    except ValueError:
        raise ValueError("❌ Impossible de convertir les coordonnées en nombres.")

    # Extraire les autres valeurs avec des valeurs par défaut si non présentes
    Water_depth = buoy_metadata.get("Water depth", "N/A")
    print(f"🌊 Water Depth : {Water_depth}")
    
    sea_temp_depth = clean_numeric(buoy_metadata.get("Sea temp depth"))
    print(f"🌡️ Sea Temp Depth : {sea_temp_depth}")
    
    Barometer_elevation = clean_numeric(buoy_metadata.get("Barometer elevation"))
    print(f"🌬️ Barometer Elevation : {Barometer_elevation}")
    
    Anemometer_height = clean_numeric(buoy_metadata.get("Anemometer height"))
    print(f"💨 Anemometer Height : {Anemometer_height}")
    
    Air_temp_height = clean_numeric(buoy_metadata.get("Air temp height"))
    print(f"🌤️ Air Temp Height : {Air_temp_height}")

    # Récupérer l'URL de la bouée
    url = get_buoy_url(station_id)
    print(f"🔗 URL de la bouée : {url}")

    # Création du dictionnaire final
    data = {
        "station_zone": station_zone,
        "lat_buoy": lat_buoy,
        "lon_buoy": lon_buoy,
        "Water_depth": Water_depth,
        "sea_temp_depth": sea_temp_depth,
        "Barometer_elevation": Barometer_elevation,
        "Anemometer_height": Anemometer_height,
        "Air_temp_height": Air_temp_height,
        "url": url
    }

    print("✅ Parsing terminé !\n")
    return data


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

def drop_tables(conn, schema_name, drop_schema=False, table_name_filter=None):
    try:
        # Créer un inspecteur pour vérifier les schémas
        inspector = inspect(conn)
        schemas = inspector.get_schema_names()
        
        if schema_name not in schemas:
            print(f"Schema '{schema_name}' does not exist.")
            return
        
        # Créer un objet metadata et charger toutes les tables
        metadata = MetaData(bind=conn)
        metadata.reflect(schema=schema_name)
        
        # Obtenir la liste des tables du schéma
        print(f"Fetching tables from schema '{schema_name}'...")
        tables = metadata.tables
        
        # Filtrer les tables si un filtre de nom est fourni
        if table_name_filter:
            filtered_tables = {name: table for name, table in tables.items() if table_name_filter.lower() in name.lower()}
            if not filtered_tables:
                print(f"No tables found matching the filter '{table_name_filter}' in schema '{schema_name}'.")
                return
            print(f"Tables matching the filter '{table_name_filter}':\n{filtered_tables.keys()}")
        else:
            filtered_tables = tables
            print(f"Tables found in schema '{schema_name}':\n{filtered_tables.keys()}")
        
        # Vérifier les verrous existants avant de continuer
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
        with conn.begin():  # Utilisation du bloc `with conn` pour garantir la gestion des transactions
            for table_name, table in filtered_tables.items():
                try:
                    # Supprimer la table
                    print(f"\nDropping table '{table_name}'...")
                    conn.execute(text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}" CASCADE'))
                    print(f"Table '{table_name}' dropped.")
                    time.sleep(1)  # Délai d'une seconde entre chaque suppression
                except Exception as e:
                    print(f"Error dropping table '{table_name}': {e}")

            # Si l'argument drop_schema est True, supprimer également le schéma
            if drop_schema:
                print(f"Dropping schema '{schema_name}'...")
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
                print(f"Schema '{schema_name}' and all its objects have been dropped.")
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

def rename_columns(df, columns):

    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Si rename_spec est un dictionnaire, on le transforme en liste de dictionnaires pour un traitement uniforme
    if isinstance(columns, dict):
        columns = [columns]  # Convertir en liste de dictionnaires pour uniformité
    
    # Process each dictionary in the list of rename_spec
    if isinstance(columns, list):
        for rename_dict in columns:
            # Filtrer les colonnes qui existent à la fois dans le DataFrame et rename_spec
            existing_columns = {col: rename_dict[col] for col in rename_dict if col in df.columns}
            
            # Renommer les colonnes uniquement si elles existent dans le DataFrame
            if existing_columns:
                df.rename(columns=existing_columns, inplace=True)
            else:
                print(f"⚠️ Aucune colonne à renommer pour ce spécification : {rename_dict}")
    
    # Retourner le DataFrame modifié
    return df

def get_day_time(col):
    # Extraire l'heure et le mois directement de l'objet Timestamp
    hour = col.hour
    
    # Initialiser une variable pour le moment de la journée
    if 0 <= hour < 6:
        daytime = 'Night'
    elif 6 <= hour < 12:
        daytime = 'Morning'
    elif 12 <= hour < 18:
        daytime = 'Afternoon'
    elif 18 <= hour < 24:
        daytime = 'Evening'
    
    # Extraire le mois directement
    month = col.month
    
    # Retourner un tuple avec le moment de la journée et le mois
    return daytime, month

def process_datetime_column(df, column):
    # Convertir la colonne en chaîne de caractères au tout début
    df[column] = df[column].astype(str)
    print(f"📌 La colonne '{column}' est maintenant convertie en chaîne de caractères.")

    # Essayer de convertir la colonne en datetime en forçant la conversion avec errors='coerce'
    try:
        df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
        print(f"📌 Conversion réussie de '{column}' en datetime.")

        # Renommer la colonne AVANT d'appliquer floor()
        df.rename(columns={column: 'Datetime'}, inplace=True)
        print('✅ Successfully renamed column to "Datetime"!')

        # Appliquer l'arrondi sur la nouvelle colonne renommée et supprimer le fuseau horaire
        df['Datetime'] = df['Datetime'].dt.floor('H').dt.tz_localize(None)
        
    except Exception as e:
        print(f"🚨 ERREUR lors de la conversion de '{column}' : {e}")
    
    return df  # Toujours retourner le DataFrame modifié

def clean_dataframe(df, cols_to_convert, verbose=False):

    # Faire une copie du DataFrame pour éviter les modifications sur une vue
    df = df.copy()

    # Supprimer les colonnes 100% vides
    df.dropna(axis=1, how="all", inplace=True)

    # Conversion des colonnes en float
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
            # Remplacement des NaN par des valeurs adaptées
            df.fillna(df.median(numeric_only=True), inplace=True)

    # Cas spécifiques : visibilité et températures
    if "visibility" in df.columns:
        df["visibility"].fillna(df["visibility"].mean(), inplace=True)

    if "water_level_above_mean" in df.columns:
        df["water_level_above_mean"].fillna(0, inplace=True)

    if "air_temperature" in df.columns:
        df["air_temperature"].fillna(df["air_temperature"].median(), inplace=True)

    if "water_temperature" in df.columns:
        df["water_temperature"].fillna(df["water_temperature"].median(), inplace=True)

    # Vérification finale
    if verbose:
        if df.isnull().sum().sum() > 0:
            print("⚠️ Il reste encore des NaN !")
        else:
            print("✅ Toutes les valeurs ont été remplacées avec succès !")
        print(df.dtypes)

    return df

def convert_columns_to_numeric(df, cols_to_convert):


    df = df.copy()  # Ne pas modifier l'original

    print("Columns before conversion:", df.columns)

    for col in cols_to_convert:
        if col in df.columns:
            try:
                # Essayer de convertir la colonne en numérique
                df[col] = pd.to_numeric(df[col], errors="raise")  # 'raise' lève une erreur si la conversion échoue
            except ValueError:
                print(f"⚠️ Impossible de convertir la colonne {col}, elle reste inchangée.")

    # Si 'is_day' existe, convertissons-le en entier proprement
    if "is_day" in df.columns:
        try:
            df["is_day"] = pd.to_numeric(df["is_day"], errors="raise").astype(int)
        except ValueError:
            print("⚠️ Impossible de convertir 'is_day' en entier, elle reste inchangée.")

    print("Columns after conversion:", df.columns)
    return df













