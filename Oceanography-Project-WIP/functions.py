def drop_columns_if_exist(df, columns_to_drop):
    existing_columns = []
    for col in columns_to_drop:
        if col in df.columns:
            existing_columns.append(col)
            print(f"Colonne '{col}' Supprimée")
        else: 
            print(f"Colonne '{col}' Non Trouvée")
    return df.drop(columns=existing_columns)

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

        # Valider l'intervalle de resampling
        valid_intervals = ['h', 'd', 'm', 'w', 'M', 'Y']
        if resample_interval not in valid_intervals:
            print(f"Erreur : L'intervalle de resampling '{resample_interval}' n'est pas valide. Utilisez l'un des suivants : {', '.join(valid_intervals)}.")
            return df

        # Resampler selon l'intervalle choisi et prendre la moyenne
        df_resampled = df.resample(resample_interval).mean().round(2)

        # Obtenir les dates min et max directement depuis l'index
        min_date, max_date = df_resampled.index.min(), df_resampled.index.max()

        # Afficher les dates min et max
        print(f"Plage de dates après resampling ({resample_interval}): \nMin: {min_date}\nMax: {max_date}  ")
        
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

# Fonction pour se connecter à PostgreSQL via psycopg2 pour la gestion de la DB
def connect_psycopg2(dbname, user, password, host="localhost", port="5432"):
    """Établit la connexion à PostgreSQL avec psycopg2 pour la base de données spécifiée."""
    try:
        conn = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        print(f"Connexion réussie à la base {dbname}.")
        return conn
    except Exception as e:
        print(f"Erreur de connexion : {e}")
        return None

# Fonction pour créer une base de données si elle n'existe pas
def create_database(dbname, user, password, host, port):
    """Crée une base de données si elle n'existe pas déjà."""
    # Se connecter à la base 'postgres' pour créer d'autres DB
    conn = connect_psycopg2(user=user, password=password, host =host, port=port)
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

# Fonction pour se connecter à une base spécifique via SQLAlchemy
def connect_postgresql(dbname, user, password, host, port):
    """Se connecte à la base Postgresql via SQLAlchemy."""
    db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    try:
        engine = create_engine(db_url)
        conn = engine.connect()
        print(f"Connexion réussie à la base {dbname} via SQLAlchemy.")
        return conn
    except Exception as e:
        print(f"Erreur de connexion avec SQLAlchemy : {e}")
        return None

# Fonction pour créer un schéma dans la base
def create_schema(conn, schema_name):
    """Crée un schéma dans la base si ce n'est pas déjà fait."""
    if conn is not None:
        try:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            print(f"✅ Schéma '{schema_name}' créé.")
        except Exception as e:
            print(f"Erreur lors de la création du schéma : {e}")